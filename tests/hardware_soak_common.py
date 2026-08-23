#!/usr/bin/env python3
"""Shared helpers for the hardware endurance and model-switch stress tests.

Both ``soak_api.py`` (sustained per-model endurance) and
``switch_stress_api.py`` (bounded model-switch stress) drive a separately
started API and enforce the same zero-tolerance policy: any HTTP failure, any
token mismatch, or any amdxdna/XRT/NPU kernel error fails the gate.
"""

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

import powercap  # noqa: E402


BASE_URL = os.environ.get("HAWKPOINT_API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ["HAWKPOINT_API_KEY"]
API_PID = int(os.environ.get("HAWKPOINT_API_PID", "0"))

# Any line naming the NPU stack together with a fault word fails the gate.
ERROR_PATTERN = re.compile(
    r"(amdxdna|xrt|npu|aie2|hwctx).*"
    r"(reset|eagain|eio|timeout|timed out|fatal|error|failed)",
    re.IGNORECASE,
)

# Client-side exceptions that count as a hard request failure.
REQUEST_ERRORS = (HTTPError, URLError, OSError, ValueError, KeyError, RuntimeError)


def request(path, payload=None, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(
        BASE_URL + path,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    started = time.perf_counter()
    with urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def load_expected():
    """Return the fixed per-model token references, keyed by model id."""
    expected_path = os.environ.get("HAWKPOINT_EXPECTED_TOKENS")
    if not expected_path:
        return {}
    return json.loads(Path(expected_path).read_text())


def installed_models(expected):
    models_response, _ = request("/v1/models")
    models = [item["id"] for item in models_response["data"]]
    if not models:
        raise SystemExit("no installed models")
    missing = sorted(set(models) - set(expected))
    if missing:
        raise SystemExit(
            f"expected token references missing for: {', '.join(missing)}"
        )
    return models


def complete_once(model, expected_tokens):
    """Issue one completion and validate it. Raises on any disagreement.

    Returns ``(elapsed, ttft_seconds, decode_tokens_per_second)``.
    """
    result, elapsed = request(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly OK."}],
            "max_tokens": len(expected_tokens),
        },
    )
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError("response contained no choices")
    stats = result.get("x_hawkpoint_stats") or {}
    actual_tokens = stats.get("generated_token_ids")
    if actual_tokens != expected_tokens:
        raise RuntimeError(
            f"token mismatch: expected {expected_tokens}, got {actual_tokens}"
        )
    return elapsed, float(stats["ttft_seconds"]), float(
        stats["decode_tokens_per_second"]
    )


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(values):
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def process_tree(root):
    pending = [root] if root else []
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            pending.extend(int(value) for value in children.read_text().split())
        except (FileNotFoundError, PermissionError, ValueError):
            pass
    return seen


def rss_mib(root):
    total_kib = 0
    for pid in process_tree(root):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ValueError):
            pass
    return total_kib / 1024


def energy_uj():
    """Sum the top-level powercap zones only (see tests/powercap.py)."""
    return powercap.energy_uj()


def kernel_errors(since_epoch, api_log):
    """Return amdxdna/XRT/NPU error lines from the kernel log and API log."""
    if os.environ.get("HAWKPOINT_SKIP_KERNEL_LOG") == "1":
        output = ""
    else:
        try:
            output = subprocess.check_output(
                ["journalctl", "-k", "--since", f"@{since_epoch}", "--no-pager"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            output = ""
    api_log = Path(api_log)
    if api_log.exists():
        output += "\n" + api_log.read_text(errors="replace")
    matches = [line for line in output.splitlines() if ERROR_PATTERN.search(line)]
    return matches[-200:]


class ResourceMonitor:
    """Sample the API process-tree RSS on a background thread."""

    def __init__(self, root_pid):
        self.root_pid = root_pid
        self.start_ram = rss_mib(root_pid)
        self.peak_ram = self.start_ram
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(1.0):
            self.peak_ram = max(self.peak_ram, rss_mib(self.root_pid))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        return False

    @property
    def end_ram(self):
        return rss_mib(self.root_pid)


def write_report(report_path, report):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))


__all__ = [
    "API_PID",
    "REQUEST_ERRORS",
    "ResourceMonitor",
    "complete_once",
    "defaultdict",
    "energy_uj",
    "powercap",
    "installed_models",
    "kernel_errors",
    "load_expected",
    "request",
    "summarize",
    "write_report",
]
