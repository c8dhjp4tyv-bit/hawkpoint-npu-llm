#!/usr/bin/env python3
"""Teacher-forced top-k logit agreement across CPU/GPU/NPU placements.

Ollama's HTTP API does not expose logits, so byte-for-byte response equality was
the only cross-placement check available -- and different backends (CPU, CUDA,
XDNA) produce numerically different greedy decodes, so that check can never pass.

This module compares placements at the logit level instead. Each placement is
driven directly through ``llama-server`` (the same runtime ollama launches, but
its native ``/completion`` endpoint returns per-token ``top_logprobs``). Every
placement is teacher-forced onto ONE fixed token prefix -- the CPU placement's
greedy decode -- and at each position we compare:

  * top-1 token agreement,
  * top-k overlap,
  * the reference logit margin (top-1 minus top-2 logprob).

A top-1 mismatch fails the gate only when the reference margin is at or above a
tolerance threshold. Low-margin (genuinely ambiguous) positions tolerate a
different top-1; high-margin mismatches fail. Exact response hashes stay in the
report for information only and never fail the release on their own.

The placement configuration mirrors the ollama-xdna launch policy
(``llm/llama_server.go``): ``-ngl`` selects offloaded layers and, when
``GGML_XDNA_XCLBIN``/``GGML_XDNA_INSTS`` are set, ``--fit off`` plus the XDNA
backend route those layers onto the NPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_XDNA_DIR = "/usr/local/lib/ollama/xdna"
DEFAULT_OLLAMA_LIB = "/usr/local/lib/ollama"


# Placement -> llama-server layer offload + XDNA routing, matching the patched
# ollama launch policy. ``xdna_layers`` is the OLLAMA_XDNA_GPU_LAYERS default.
PLACEMENTS = {
    "cpu_only": {"ngl": 0, "xdna": False},
    "gpu_only": {"ngl": 999, "xdna": False},
    # Exercise the XDNA backend without GPU layer offload. This catches a
    # backend that only works accidentally when CUDA is also present.
    "xdna_only": {"ngl": 0, "xdna": True},
    "cpu_gpu": {"ngl": 8, "xdna": False},
    "cpu_gpu_npu": {"ngl": 8, "xdna": True},
}


class PlacementError(RuntimeError):
    """A placement failed to run on its intended backend."""


def _api(port, path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _wait_ready(process, port, seconds=180):
    for _ in range(seconds):
        if process.poll() is not None:
            raise PlacementError("llama-server exited during startup")
        try:
            _api(port, "/health", timeout=2)
            return
        except (OSError, HTTPError, URLError):
            time.sleep(1)
    raise PlacementError("llama-server was not ready in time")


def _server_env(placement, xdna_dir, ollama_lib):
    cuda_dir = os.path.join(ollama_lib, "cuda_v13")
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": os.pathsep.join([ollama_lib, cuda_dir]),
    }
    for name in ("GGML_XDNA_XCLBIN", "GGML_XDNA_INSTS", "GGML_BACKEND_PATH",
                 "GGML_XDNA_MODEL_ARCH"):
        env.pop(name, None)
    if placement["xdna"]:
        # NPU placement: load the XDNA backend so its layers dispatch to the NPU.
        env.update(
            {
                "GGML_XDNA_XCLBIN": os.path.join(xdna_dir, "experts.xclbin"),
                "GGML_XDNA_INSTS": os.path.join(xdna_dir, "insts.bin"),
                "GGML_BACKEND_PATH": os.path.join(xdna_dir, "libggml-xdna.so"),
                "GGML_XDNA_MODEL_ARCH": "qwen2",
            }
        )
    elif placement["ngl"] > 0:
        # CUDA placements: the CUDA ggml backend lives in a subdirectory that
        # ggml does not auto-discover, so point GGML_BACKEND_PATH at it.
        env["GGML_BACKEND_PATH"] = os.path.join(cuda_dir, "libggml-cuda.so")
    return env


def _server_args(server_bin, model, port, placement):
    args = [server_bin, "-m", model, "--port", str(port), "-ngl",
            str(placement["ngl"]), "--no-webui"]
    if placement["xdna"]:
        args += ["--fit", "off"]
    return args


def _assert_backend(name, placement, server):
    """Fail unless the placement actually ran on its intended backend.

    Guards against a placement silently falling back (an XDNA or CUDA failure
    quietly running on CPU) being accepted as agreement. NPU dispatch is proven
    from the XDNA offload log; CUDA use is proven from held GPU memory.
    """
    log_text = server.log_text()
    gpu_mib = server.gpu_memory_mib()
    if placement["xdna"]:
        if "XDNA dense Qwen offload count" not in log_text:
            raise PlacementError(
                f"{name}: no XDNA dispatch evidence in llama-server log"
            )
    elif placement["ngl"] > 0:
        if "no usable GPU found" in log_text:
            raise PlacementError(
                f"{name}: CUDA backend not loaded; fell back to CPU"
            )
        if gpu_mib is not None and gpu_mib <= 0:
            raise PlacementError(
                f"{name}: expected CUDA use but the process held no GPU memory"
            )
    else:  # cpu_only
        if gpu_mib is not None and gpu_mib > 0:
            raise PlacementError(
                f"{name}: expected pure CPU but the process held GPU memory"
            )
    return gpu_mib


class LlamaServer:
    def __init__(self, name, placement, model, port, log_path, server_bin,
                 xdna_dir, ollama_lib):
        self.name = name
        self.placement = placement
        self.log_path = log_path
        self.port = port
        self._log = open(log_path, "w")
        self._process = subprocess.Popen(
            _server_args(server_bin, model, port, placement),
            env=_server_env(placement, xdna_dir, ollama_lib),
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    def __enter__(self):
        _wait_ready(self._process, self.port)
        return self

    def __exit__(self, *exc):
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        self._log.close()
        return False

    def log_text(self):
        with open(self.log_path, errors="replace") as handle:
            return handle.read()

    def gpu_memory_mib(self):
        """GPU memory (MiB) this server's process holds, via nvidia-smi.

        Returns 0.0 when the process holds no GPU memory and None when
        nvidia-smi is unavailable (so callers can fall back to log evidence).
        """
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                if int(parts[0]) == self._process.pid:
                    return float(parts[1])
        return 0.0

    def tokenize(self, text):
        return _api(self.port, "/tokenize", {"content": text})["tokens"]

    def topk_at(self, prefix_tokens, k):
        """top-k logprobs for the next token after ``prefix_tokens``."""
        result = _api(
            self.port,
            "/completion",
            {
                "prompt": prefix_tokens,
                "n_predict": 1,
                "n_probs": k,
                "temperature": 0,
                "seed": 1,
                "cache_prompt": False,
            },
        )
        entry = result["completion_probabilities"][0]
        return [
            {"id": candidate["id"], "logprob": candidate["logprob"]}
            for candidate in entry["top_logprobs"]
        ]

    def greedy_token_ids(self, prompt, count):
        result = _api(
            self.port,
            "/completion",
            {
                "prompt": prompt,
                "n_predict": count,
                "n_probs": 1,
                "temperature": 0,
                "seed": 1,
                "cache_prompt": False,
            },
        )
        return [entry["id"] for entry in result["completion_probabilities"]]


def teacher_forced_topk(server, base_tokens, reference_tokens, k):
    """Per-position top-k for every teacher-forced position.

    Position ``i`` conditions on ``base_tokens + reference_tokens[:i]`` so every
    placement is scored on the identical prefix.
    """
    rows = []
    for index in range(len(reference_tokens)):
        prefix = list(base_tokens) + list(reference_tokens[:index])
        rows.append(server.topk_at(prefix, k))
    return rows


def _margin(topk_row):
    if len(topk_row) < 2:
        return float("inf")
    return topk_row[0]["logprob"] - topk_row[1]["logprob"]


def evaluate_agreement(per_placement_topk, reference_name, margin_threshold):
    """Compare each placement's teacher-forced top-k against the reference.

    Returns ``(report, gate_failures)``. A high-margin top-1 mismatch (reference
    margin >= threshold) is a failure; a low-margin mismatch is tolerated.
    """
    reference = per_placement_topk[reference_name]
    positions = len(reference)
    report = {"reference": reference_name, "margin_threshold": margin_threshold,
              "positions": positions, "placements": {}}
    gate_failures = []

    for name, rows in per_placement_topk.items():
        top1_matches = 0
        tolerated = []
        violations = []
        topk_overlap = []
        for index in range(positions):
            ref_row = reference[index]
            row = rows[index]
            ref_top1 = ref_row[0]["id"]
            top1 = row[0]["id"]
            ref_ids = {c["id"] for c in ref_row}
            ids = {c["id"] for c in row}
            topk_overlap.append(len(ref_ids & ids))
            if top1 == ref_top1:
                top1_matches += 1
                continue
            ref_margin = _margin(ref_row)
            record = {"position": index, "reference_top1": ref_top1,
                      "placement_top1": top1,
                      "reference_margin": round(ref_margin, 4)}
            if ref_margin >= margin_threshold:
                violations.append(record)
            else:
                tolerated.append(record)
        report["placements"][name] = {
            "top1_matches": top1_matches,
            "top1_total": positions,
            "tolerated_mismatches": tolerated,
            "high_margin_violations": violations,
            "min_topk_overlap": min(topk_overlap) if topk_overlap else None,
            "mean_topk_overlap": (
                sum(topk_overlap) / len(topk_overlap) if topk_overlap else None
            ),
        }
        if violations:
            gate_failures.append(f"{name}_top1_divergence")
    return report, gate_failures


def run_placement_agreement(
    model,
    *,
    prompt="Explain why the sky is blue.",
    positions=16,
    top_k=5,
    margin_threshold=1.0,
    reference="cpu_only",
    server_bin=os.path.join(DEFAULT_OLLAMA_LIB, "llama-server"),
    xdna_dir=DEFAULT_XDNA_DIR,
    ollama_lib=DEFAULT_OLLAMA_LIB,
    log_dir=".",
    base_port=11700,
):
    """Drive every placement through llama-server and evaluate agreement.

    ``model`` is a path to the GGUF blob. Returns a report dict with
    ``gate_failures`` (empty means the logit-agreement gate passed).
    """
    os.makedirs(log_dir, exist_ok=True)
    per_placement_topk = {}
    determinism = {}
    backends = {}

    # Establish the reference token prefix from the reference placement first.
    reference_placement = PLACEMENTS[reference]
    with LlamaServer(
        reference, reference_placement, model, base_port,
        os.path.join(log_dir, f"logit-{reference}.log"),
        server_bin, xdna_dir, ollama_lib,
    ) as server:
        base_tokens = server.tokenize(prompt)
        reference_tokens = server.greedy_token_ids(prompt, positions)
        rows = teacher_forced_topk(server, base_tokens, reference_tokens, top_k)
        # Within-placement determinism: a second pass must be identical.
        rows_again = teacher_forced_topk(
            server, base_tokens, reference_tokens, top_k
        )
        # Assert the backend only after the model has actually run, since the
        # offload/dispatch evidence is logged during the first load.
        _assert_backend(reference, reference_placement, server)
        determinism[reference] = _rows_equal(rows, rows_again)
        per_placement_topk[reference] = rows
        backends[reference] = "cpu"

    for name, placement in PLACEMENTS.items():
        if name == reference:
            continue
        with LlamaServer(
            name, placement, model, base_port + list(PLACEMENTS).index(name),
            os.path.join(log_dir, f"logit-{name}.log"),
            server_bin, xdna_dir, ollama_lib,
        ) as server:
            rows = teacher_forced_topk(
                server, base_tokens, reference_tokens, top_k
            )
            rows_again = teacher_forced_topk(
                server, base_tokens, reference_tokens, top_k
            )
            _assert_backend(name, placement, server)
            determinism[name] = _rows_equal(rows, rows_again)
            per_placement_topk[name] = rows
            backends[name] = "xdna" if placement["xdna"] else (
                "cuda" if placement["ngl"] > 0 else "cpu"
            )

    report, gate_failures = evaluate_agreement(
        per_placement_topk, reference, margin_threshold
    )
    report["determinism"] = determinism
    report["backends"] = backends
    report["reference_token_ids"] = reference_tokens
    for name, is_deterministic in determinism.items():
        if not is_deterministic:
            gate_failures.append(f"{name}_nondeterministic")
    report["gate_failures"] = gate_failures
    return report


def _rows_equal(rows_a, rows_b):
    if len(rows_a) != len(rows_b):
        return False
    for row_a, row_b in zip(rows_a, rows_b):
        if [c["id"] for c in row_a] != [c["id"] for c in row_b]:
            return False
    return True
