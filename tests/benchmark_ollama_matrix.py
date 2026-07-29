#!/usr/bin/env python3
"""Measure CPU/GPU/NPU Ollama placements with one pinned model and prompt."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from urllib.request import Request, urlopen

from verify_ollama_manifest import verify_manifest


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def process_tree(root):
    pending = [root]
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children")
            pending.extend(int(value) for value in children.read_text().split())
        except (OSError, ValueError):
            pass
    return seen


def rss_mib(root):
    total = 0
    for pid in process_tree(root):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except (OSError, ValueError):
            pass
    return total / 1024


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
        return sum(float(line) for line in output.splitlines() if line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def energy_uj():
    readings = []
    for path in Path("/sys/class/powercap").glob("**/energy_uj"):
        try:
            readings.append(int(path.read_text()))
        except (OSError, ValueError):
            pass
    return sum(readings) if readings else None


def api(port, path, payload=None, timeout=600):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def streaming_generate(port, payload, timeout=600):
    """Return the final Ollama event and wall-clock time to its first token."""
    body = json.dumps({**payload, "stream": True}).encode()
    request = Request(
        f"http://127.0.0.1:{port}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at = None
    pieces = []
    final = None
    with urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            piece = event.get("response", "")
            if piece:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(piece)
            if event.get("done"):
                final = event
    if final is None or first_token_at is None:
        raise RuntimeError("stream ended without a token or final metrics")
    return final, first_token_at - started, "".join(pieces)


def wait_ready(process, port):
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError("Ollama server exited during startup")
        try:
            api(port, "/api/version", timeout=1)
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError("Ollama server was not ready in 120 seconds")


def summary(values):
    if not values:
        return {"median": None, "p95": None, "min": None, "max": None}
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def compare_placement_responses(results, expected_count):
    hashes = {
        item["placement"]: item["response_sha256"]
        for item in results
        if item["response_sha256"] is not None
    }
    return hashes, len(hashes) == expected_count and len(set(hashes.values())) == 1


def benchmark_mode(args, mode, index):
    port = args.base_port + index
    environment = {
        **os.environ,
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        "OLLAMA_MODELS": str(args.models_dir),
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_LLM_LIBRARY": mode["library"],
    }
    for name in ("GGML_BACKEND_PATH", "GGML_XDNA_XCLBIN", "GGML_XDNA_INSTS"):
        environment.pop(name, None)
    if mode["xdna"]:
        environment.update(
            {
                "GGML_BACKEND_PATH": str(args.xdna_dir / "libggml-xdna.so"),
                "GGML_XDNA_XCLBIN": str(args.xdna_dir / "experts.xclbin"),
                "GGML_XDNA_INSTS": str(args.xdna_dir / "insts.bin"),
                "OLLAMA_XDNA_GPU_LAYERS": str(mode["num_gpu"]),
            }
        )
    log_path = args.output_dir / f"{mode['id']}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [str(args.ollama_bin), "serve"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(process, port)
            api(
                port,
                "/api/pull",
                {"model": args.model, "stream": False},
                timeout=1200,
            )
            verify_manifest(
                args.models_dir,
                args.model,
                args.model_manifest_sha256,
            )
            payload = {
                "model": args.model,
                "prompt": args.prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "num_gpu": mode["num_gpu"],
                    "num_predict": args.tokens,
                    "temperature": 0,
                    "seed": 1,
                },
            }
            warmup_started = time.monotonic()
            warmup_requests = 0
            while (
                warmup_requests == 0
                or time.monotonic() - warmup_started < args.warmup_seconds
            ):
                streaming_generate(port, payload)
                warmup_requests += 1
            peak_ram = rss_mib(process.pid)
            peak_vram = gpu_vram_mib()
            stop = threading.Event()

            def monitor():
                nonlocal peak_ram, peak_vram
                while not stop.wait(1):
                    peak_ram = max(peak_ram, rss_mib(process.pid))
                    current = gpu_vram_mib()
                    if current is not None:
                        peak_vram = (
                            current if peak_vram is None else max(peak_vram, current)
                        )

            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            ttfts = []
            rates = []
            total_times = []
            response_hashes = []
            failures = []
            start_energy = energy_uj()
            started = time.time()
            for request_index in range(args.requests):
                try:
                    result, ttft, response_text = streaming_generate(port, payload)
                    ttfts.append(ttft)
                    rates.append(
                        result["eval_count"]
                        / max(result["eval_duration"] / 1_000_000_000, 1e-9)
                    )
                    total_times.append(result["total_duration"] / 1_000_000_000)
                    response_hashes.append(
                        hashlib.sha256(response_text.encode()).hexdigest()
                    )
                except Exception as exc:
                    failures.append(
                        {"request": request_index, "error": str(exc)[:500]}
                    )
            completed = time.time()
            end_energy = energy_uj()
            stop.set()
            monitor_thread.join(timeout=2)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    log_text = log_path.read_text(errors="replace")
    if mode["xdna"] and "XDNA dense Qwen offload count:" not in log_text:
        failures.append({"request": -1, "error": "no XDNA dispatch evidence"})
    if len(set(response_hashes)) > 1:
        failures.append(
            {
                "request": -1,
                "error": "deterministic response changed between requests",
            }
        )
    return {
        "placement": mode["id"],
        "requests": args.requests,
        "successful_requests": len(ttfts),
        "failures": failures,
        "ttft_seconds": summary(ttfts),
        "tokens_per_second": summary(rates),
        "total_seconds": summary(total_times),
        "peak_ram_mib": peak_ram,
        "peak_gpu_vram_mib": peak_vram,
        "energy_joules": (
            (end_energy - start_energy) / 1_000_000
            if start_energy is not None
            and end_energy is not None
            and end_energy >= start_energy
            else None
        ),
        "wall_seconds": completed - started,
        "warmup_seconds": time.monotonic() - warmup_started,
        "warmup_requests": warmup_requests,
        "num_gpu": mode["num_gpu"],
        "xdna_enabled": mode["xdna"],
        "response_sha256": response_hashes[0] if response_hashes else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-bin", type=Path, required=True)
    parser.add_argument("--xdna-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--gpu-library", default="cuda_v13")
    parser.add_argument("--partial-gpu-layers", type=int, default=8)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--warmup-seconds", type=int, default=300)
    parser.add_argument("--base-port", type=int, default=11500)
    parser.add_argument("--prompt", default="Explain why the sky is blue.")
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be positive")
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.warmup_seconds < 0:
        parser.error("--warmup-seconds cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    modes = [
        {"id": "cpu_only", "library": "cpu", "num_gpu": 0, "xdna": False},
        {
            "id": "gpu_only",
            "library": args.gpu_library,
            "num_gpu": 999,
            "xdna": False,
        },
        {
            "id": "cpu_gpu",
            "library": args.gpu_library,
            "num_gpu": args.partial_gpu_layers,
            "xdna": False,
        },
        {
            "id": "cpu_gpu_npu",
            "library": args.gpu_library,
            "num_gpu": args.partial_gpu_layers,
            "xdna": True,
        },
    ]
    results = []
    for index, mode in enumerate(modes):
        try:
            results.append(benchmark_mode(args, mode, index))
        except Exception as exc:
            results.append(
                {
                    "placement": mode["id"],
                    "requests": args.requests,
                    "successful_requests": 0,
                    "failures": [{"request": -1, "error": str(exc)[:1000]}],
                    "ttft_seconds": summary([]),
                    "tokens_per_second": summary([]),
                    "total_seconds": summary([]),
                    "peak_ram_mib": None,
                    "peak_gpu_vram_mib": None,
                    "energy_joules": None,
                    "wall_seconds": None,
                    "warmup_seconds": None,
                    "warmup_requests": 0,
                    "num_gpu": mode["num_gpu"],
                    "xdna_enabled": mode["xdna"],
                    "response_sha256": None,
                }
            )
    placement_hashes, cross_placement_agreement = compare_placement_responses(
        results,
        len(modes),
    )
    gate_failures = [
        item["placement"]
        for item in results
        if item["failures"] or item["successful_requests"] != args.requests
    ]
    if not cross_placement_agreement:
        gate_failures.append("cross_placement_response_mismatch")
    report = {
        "schema_version": 1,
        "model": args.model,
        "model_manifest_sha256": args.model_manifest_sha256,
        "prompt": args.prompt,
        "generated_tokens": args.tokens,
        "warmup_seconds_per_placement": args.warmup_seconds,
        "placement_response_sha256": placement_hashes,
        "cross_placement_response_agreement": cross_placement_agreement,
        "gate_failures": gate_failures,
        "placements": results,
    }
    json_path = args.output_dir / "ollama-placement-matrix.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = [
        "| Placement | median TTFT | p95 TTFT | median token/s | RAM MiB | VRAM MiB | Energy J | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        ttft = item["ttft_seconds"]["median"]
        p95 = item["ttft_seconds"]["p95"]
        rate = item["tokens_per_second"]["median"]
        markdown.append(
            "| {placement} | {ttft} | {p95} | {rate} | "
            "{ram} | {vram} | {energy} | {failures} |".format(
                placement=item["placement"],
                ttft=f"{ttft:.3f}" if ttft is not None else "failed",
                p95=f"{p95:.3f}" if p95 is not None else "failed",
                rate=f"{rate:.3f}" if rate is not None else "failed",
                ram=(
                    f"{item['peak_ram_mib']:.1f}"
                    if item["peak_ram_mib"] is not None
                    else "unavailable"
                ),
                vram=(
                    f"{item['peak_gpu_vram_mib']:.1f}"
                    if item["peak_gpu_vram_mib"] is not None
                    else "unavailable"
                ),
                energy=(
                    f"{item['energy_joules']:.1f}"
                    if item["energy_joules"] is not None
                    else "unavailable"
                ),
                failures=len(item["failures"]),
            )
        )
    (args.output_dir / "ollama-placement-matrix.md").write_text(
        "\n".join(
            markdown
            + [
                "",
                "Cross-placement response SHA-256 agreement: "
                + ("PASS" if cross_placement_agreement else "FAIL"),
            ]
        )
        + "\n"
    )
    print("\n".join(markdown))
    if gate_failures:
        raise SystemExit("benchmark gate failed for: " + ", ".join(gate_failures))


if __name__ == "__main__":
    main()
