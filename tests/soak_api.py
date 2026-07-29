#!/usr/bin/env python3
"""Measured, model-switching endurance test for a separately started API."""

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.environ.get("HAWKPOINT_API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ["HAWKPOINT_API_KEY"]
COMPLETIONS = int(os.environ.get("HAWKPOINT_SOAK_COMPLETIONS", "1000"))
REPORT = Path(os.environ.get("HAWKPOINT_SOAK_REPORT", "soak-report.json"))
API_LOG = Path(os.environ.get("HAWKPOINT_API_LOG", "hawkpoint-api.log"))
API_PID = int(os.environ.get("HAWKPOINT_API_PID", "0"))
EXPECTED_PATH = os.environ.get("HAWKPOINT_EXPECTED_TOKENS")
ERROR_PATTERN = re.compile(
    r"(amdxdna|xrt|npu).*(reset|eagain|eio|timeout|fatal|error)",
    re.IGNORECASE,
)


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


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


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


def gpu_vram_mib():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return sum(float(line.strip()) for line in output.splitlines() if line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def energy_uj():
    values = []
    for path in Path("/sys/class/powercap").glob("**/energy_uj"):
        try:
            values.append(int(path.read_text()))
        except (OSError, ValueError):
            pass
    return sum(values) if values else None


def kernel_errors(since_epoch):
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
    if API_LOG.exists():
        output += "\n" + API_LOG.read_text(errors="replace")
    matches = [line for line in output.splitlines() if ERROR_PATTERN.search(line)]
    return matches[-200:]


def summarize(values):
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


expected = {}
if EXPECTED_PATH:
    expected = json.loads(Path(EXPECTED_PATH).read_text())

models_response, _ = request("/v1/models")
models = [item["id"] for item in models_response["data"]]
if not models:
    raise SystemExit("no installed models")
missing = sorted(set(models) - set(expected))
if missing:
    raise SystemExit(f"expected token references missing for: {', '.join(missing)}")

started_epoch = int(time.time())
start_ram = rss_mib(API_PID)
start_energy = energy_uj()
peak_ram = start_ram
peak_vram = gpu_vram_mib()
monitor_stop = threading.Event()


def monitor_resources():
    global peak_ram, peak_vram
    while not monitor_stop.wait(1.0):
        peak_ram = max(peak_ram, rss_mib(API_PID))
        vram = gpu_vram_mib()
        if vram is not None:
            peak_vram = vram if peak_vram is None else max(peak_vram, vram)


monitor = threading.Thread(target=monitor_resources, daemon=True)
monitor.start()
latencies = []
ttfts = []
token_rates = []
per_model = defaultdict(lambda: {"success": 0, "failure": 0})
failures = []
switch_errors = 0
previous_model = None

try:
    for index in range(COMPLETIONS):
        model = models[index % len(models)]
        switched = previous_model is not None and previous_model != model
        try:
            result, elapsed = request(
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "Reply with exactly OK."}
                    ],
                    "max_tokens": len(expected[model]),
                },
            )
            choices = result.get("choices") or []
            if not choices:
                raise RuntimeError("response contained no choices")
            stats = result.get("x_hawkpoint_stats") or {}
            actual_tokens = stats.get("generated_token_ids")
            if actual_tokens != expected[model]:
                raise RuntimeError(
                    f"token mismatch: expected {expected[model]}, got {actual_tokens}"
                )
            ttft = float(stats["ttft_seconds"])
            token_rate = float(stats["decode_tokens_per_second"])
            latencies.append(elapsed)
            ttfts.append(ttft)
            token_rates.append(token_rate)
            per_model[model]["success"] += 1
        except (HTTPError, URLError, OSError, ValueError, KeyError, RuntimeError) as exc:
            per_model[model]["failure"] += 1
            if switched:
                switch_errors += 1
            failures.append(
                {"index": index, "model": model, "error": str(exc)[:500]}
            )
        previous_model = model
        if index and index % 100 == 0:
            print(f"completed {index}/{COMPLETIONS}", flush=True)
finally:
    monitor_stop.set()
    monitor.join(timeout=2)

end_ram = rss_mib(API_PID)
end_energy = energy_uj()
xrt_errors = kernel_errors(started_epoch)
report = {
    "schema_version": 1,
    "started_epoch": started_epoch,
    "completed_epoch": int(time.time()),
    "requested_completions": COMPLETIONS,
    "successful_requests": len(latencies),
    "failed_requests": len(failures),
    "model_switch_errors": switch_errors,
    "models": dict(per_model),
    "latency_seconds": summarize(latencies),
    "ttft_seconds": summarize(ttfts),
    "decode_tokens_per_second": summarize(token_rates),
    "ram_mib": {
        "start": start_ram,
        "end": end_ram,
        "peak": peak_ram,
        "growth": end_ram - start_ram,
    },
    "gpu_vram_peak_mib": peak_vram,
    "energy_joules": (
        (end_energy - start_energy) / 1_000_000
        if start_energy is not None and end_energy is not None and end_energy >= start_energy
        else None
    ),
    "xrt_npu_error_count": len(xrt_errors),
    "xrt_npu_errors": xrt_errors,
    "expected_token_ids": expected,
    "failures": failures[-200:],
}
REPORT.parent.mkdir(parents=True, exist_ok=True)
temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
os.replace(temporary, REPORT)
print(json.dumps(report, indent=2, sort_keys=True))

if failures or xrt_errors:
    raise SystemExit(
        f"FAIL failures={len(failures)} xrt_npu_errors={len(xrt_errors)}"
    )
print(f"PASS {COMPLETIONS} measured completions across {len(models)} model(s)")
