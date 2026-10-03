#!/usr/bin/env python3
"""Benchmark OpenAI-compatible API server throughput, latency, and TTFT."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class RequestBenchmarkResult:
    status_code: int
    ttft_seconds: float | None
    total_latency_seconds: float
    generated_tokens: int
    prompt_tokens: int
    tokens_per_second: float
    error: str | None = None


def fetch_metrics(base_url: str, api_key: str | None = None) -> dict[str, float]:
    """Fetch and parse current Prometheus metrics from /metrics."""
    url = f"{base_url.rstrip('/')}/metrics"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    metrics: dict[str, float] = {}
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            for line in resp.read().decode("utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        metrics[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
    except Exception:
        pass
    return metrics


def execute_single_request(
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool = True,
    endpoint: str = "/v1/chat/completions",
) -> RequestBenchmarkResult:
    """Execute one benchmark request and measure latency and tokens."""
    url = f"{base_url.rstrip('/')}{endpoint}"
    if endpoint == "/v1/completions":
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": stream,
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": stream,
        }

    if stream:
        payload["stream_options"] = {"include_usage": True}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    start_time = time.monotonic()
    first_token_time: float | None = None
    generated_tokens = 0
    prompt_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status_code = resp.status
            if stream:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line.removeprefix("data: ")
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        delta = choice.get("delta") or choice
                        content = delta.get("content") or delta.get("text")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.monotonic()
                            generated_tokens += 1
                    if "usage" in data and data["usage"]:
                        u = data["usage"]
                        if "completion_tokens" in u:
                            generated_tokens = u["completion_tokens"]
                        if "prompt_tokens" in u:
                            prompt_tokens = u["prompt_tokens"]
            else:
                body = json.loads(resp.read().decode("utf-8"))
                usage = body.get("usage", {})
                generated_tokens = usage.get("completion_tokens", max_tokens)
                prompt_tokens = usage.get("prompt_tokens", 0)

        end_time = time.monotonic()
        total_latency = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_latency
        decode_duration = (end_time - first_token_time) if first_token_time else total_latency
        tps = (generated_tokens / decode_duration) if decode_duration > 0 else 0.0

        return RequestBenchmarkResult(
            status_code=status_code,
            ttft_seconds=ttft if stream else None,
            total_latency_seconds=total_latency,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_second=tps,
        )

    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return RequestBenchmarkResult(
            status_code=exc.code,
            ttft_seconds=None,
            total_latency_seconds=time.monotonic() - start_time,
            generated_tokens=0,
            prompt_tokens=0,
            tokens_per_second=0.0,
            error=f"HTTP {exc.code}: {err_body[:200]}",
        )
    except Exception as exc:
        return RequestBenchmarkResult(
            status_code=0,
            ttft_seconds=None,
            total_latency_seconds=time.monotonic() - start_time,
            generated_tokens=0,
            prompt_tokens=0,
            tokens_per_second=0.0,
            error=str(exc),
        )


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * (pct / 100.0)
    floor_idx = int(idx)
    ceil_idx = min(floor_idx + 1, len(sorted_data) - 1)
    weight = idx - floor_idx
    return sorted_data[floor_idx] * (1.0 - weight) + sorted_data[ceil_idx] * weight


def run_benchmark(
    base_url: str,
    model: str,
    api_key: str | None = None,
    prompt: str = "Explain the architecture of a systolic neural processing unit in three sentences.",
    max_tokens: int = 32,
    num_requests: int = 10,
    concurrency: int = 1,
    stream: bool = True,
    endpoint: str = "/v1/chat/completions",
) -> dict:
    """Run concurrent benchmark requests and compute summary statistics."""
    initial_metrics = fetch_metrics(base_url, api_key)
    wall_start = time.monotonic()
    results: list[RequestBenchmarkResult] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                execute_single_request,
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                stream=stream,
                endpoint=endpoint,
            )
            for _ in range(num_requests)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    wall_duration = time.monotonic() - wall_start
    final_metrics = fetch_metrics(base_url, api_key)

    successful = [r for r in results if r.status_code == 200]
    latencies = [r.total_latency_seconds for r in successful]
    ttfts = [r.ttft_seconds for r in successful if r.ttft_seconds is not None]
    tps_list = [r.tokens_per_second for r in successful]
    total_tokens = sum(r.generated_tokens for r in successful)
    aggregate_tps = (total_tokens / wall_duration) if wall_duration > 0 else 0.0

    report = {
        "configuration": {
            "base_url": base_url,
            "model": model,
            "endpoint": endpoint,
            "num_requests": num_requests,
            "concurrency": concurrency,
            "stream": stream,
            "max_tokens": max_tokens,
        },
        "summary": {
            "total_requests": num_requests,
            "successful_requests": len(successful),
            "failed_requests": num_requests - len(successful),
            "success_rate_percent": (len(successful) / num_requests * 100) if num_requests else 0.0,
            "wall_time_seconds": round(wall_duration, 3),
            "total_generated_tokens": total_tokens,
            "aggregate_tokens_per_second": round(aggregate_tps, 2),
        },
        "latency_seconds": {
            "min": round(min(latencies), 4) if latencies else None,
            "p50": round(_percentile(latencies, 50), 4) if latencies else None,
            "p95": round(_percentile(latencies, 95), 4) if latencies else None,
            "p99": round(_percentile(latencies, 99), 4) if latencies else None,
            "max": round(max(latencies), 4) if latencies else None,
            "mean": round(statistics.mean(latencies), 4) if latencies else None,
        },
        "ttft_seconds": {
            "min": round(min(ttfts), 4) if ttfts else None,
            "p50": round(_percentile(ttfts, 50), 4) if ttfts else None,
            "p95": round(_percentile(ttfts, 95), 4) if ttfts else None,
            "p99": round(_percentile(ttfts, 99), 4) if ttfts else None,
            "max": round(max(ttfts), 4) if ttfts else None,
            "mean": round(statistics.mean(ttfts), 4) if ttfts else None,
        }
        if stream
        else None,
        "tokens_per_second": {
            "min": round(min(tps_list), 2) if tps_list else None,
            "p50": round(_percentile(tps_list, 50), 2) if tps_list else None,
            "p95": round(_percentile(tps_list, 95), 2) if tps_list else None,
            "max": round(max(tps_list), 2) if tps_list else None,
            "mean": round(statistics.mean(tps_list), 2) if tps_list else None,
        },
        "metrics_diff": {
            k: round(final_metrics.get(k, 0.0) - initial_metrics.get(k, 0.0), 2)
            for k in set(initial_metrics) | set(final_metrics)
            if final_metrics.get(k, 0.0) != initial_metrics.get(k, 0.0)
        },
        "individual_results": [asdict(r) for r in results],
    }
    return report


def format_text_report(report: dict) -> str:
    """Format benchmark results into human-readable terminal text."""
    c = report["configuration"]
    s = report["summary"]
    lat = report["latency_seconds"]
    ttft = report.get("ttft_seconds")
    tps = report["tokens_per_second"]

    lines = [
        "==================================================",
        f" HawkPoint NPU API Benchmark: {c['model']}",
        "==================================================",
        f"Endpoint:      {c['endpoint']} (stream={c['stream']})",
        f"Requests:      {s['successful_requests']}/{s['total_requests']} succeeded ({s['success_rate_percent']:.1f}%)",
        f"Concurrency:   {c['concurrency']}",
        f"Wall Time:     {s['wall_time_seconds']:.2f}s",
        f"Tokens Gen:    {s['total_generated_tokens']} tokens",
        f"Aggregate TPS: {s['aggregate_tokens_per_second']:.2f} tok/s",
        "--------------------------------------------------",
        "Latency (s):",
        f"  min: {lat['min']}  p50: {lat['p50']}  p95: {lat['p95']}  p99: {lat['p99']}  max: {lat['max']}",
    ]
    if ttft:
        lines.extend([
            "Time To First Token (TTFT, s):",
            f"  min: {ttft['min']}  p50: {ttft['p50']}  p95: {ttft['p95']}  max: {ttft['max']}",
        ])
    lines.extend([
        "Decode Throughput (tok/s):",
        f"  min: {tps['min']}  p50: {tps['p50']}  p95: {tps['p95']}  max: {tps['max']}  mean: {tps['mean']}",
        "==================================================",
    ])
    return "\n".join(lines)


def format_markdown_report(report: dict) -> str:
    """Format benchmark results into GitHub Markdown table."""
    c = report["configuration"]
    s = report["summary"]
    lat = report["latency_seconds"]
    ttft = report.get("ttft_seconds") or {}
    tps = report["tokens_per_second"]

    lines = [
        f"## API Benchmark Report: `{c['model']}`",
        "",
        "| Metric | Value |",
        "|:-------|:------|",
        f"| **Target Endpoint** | `{c['endpoint']}` |",
        f"| **Concurrency** | `{c['concurrency']}` |",
        f"| **Total / Success Requests** | `{s['successful_requests']} / {s['total_requests']}` ({s['success_rate_percent']:.1f}%) |",
        f"| **Wall Time** | `{s['wall_time_seconds']}s` |",
        f"| **Total Generated Tokens** | `{s['total_generated_tokens']}` |",
        f"| **Aggregate Throughput** | **`{s['aggregate_tokens_per_second']} tok/s`** |",
        f"| **TTFT (p50 / p95)** | `{ttft.get('p50', 'N/A')}s / {ttft.get('p95', 'N/A')}s` |",
        f"| **Latency (p50 / p95)** | `{lat.get('p50', 'N/A')}s / {lat.get('p95', 'N/A')}s` |",
        f"| **Decode TPS (p50 / Mean)** | `{tps.get('p50', 'N/A')} / {tps.get('mean', 'N/A')} tok/s` |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible NPU API server.")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of API server")
    parser.add_argument("--api-key", default=os.environ.get("HAWKPOINT_API_KEY"), help="API bearer token")
    parser.add_argument("--model", default="smollm2-135m-xdna1", help="Model identifier to benchmark")
    parser.add_argument("--endpoint", choices=["/v1/chat/completions", "/v1/completions"], default="/v1/chat/completions")
    parser.add_argument("-n", "--requests", type=int, default=10, help="Total number of requests (default: 10)")
    parser.add_argument("-c", "--concurrency", type=int, default=1, help="Concurrent client workers (default: 1)")
    parser.add_argument("--tokens", type=int, default=32, help="Max tokens per completion (default: 32)")
    parser.add_argument("--prompt", default="Explain why the sky is blue in two sentences.", help="Prompt text")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming response mode")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    parser.add_argument("-o", "--output", type=Path, help="Write benchmark report to file")
    parser.add_argument("--min-tps", type=float, help="Exit with code 2 if aggregate TPS is below this threshold")
    args = parser.parse_args(argv)

    if args.requests < 1:
        parser.error("--requests must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    report = run_benchmark(
        base_url=args.url,
        model=args.model,
        api_key=args.api_key,
        prompt=args.prompt,
        max_tokens=args.tokens,
        num_requests=args.requests,
        concurrency=args.concurrency,
        stream=not args.no_stream,
        endpoint=args.endpoint,
    )

    if args.format == "markdown":
        output_str = format_markdown_report(report)
    elif args.format == "json":
        output_str = json.dumps(report, indent=2)
    else:
        output_str = format_text_report(report)

    if args.output:
        args.output.write_text(output_str, encoding="utf-8")
        print(f"Benchmark report written to {args.output}")
    else:
        print(output_str)

    if args.min_tps is not None:
        actual_tps = report["summary"]["aggregate_tokens_per_second"]
        if actual_tps < args.min_tps:
            sys.stderr.write(f"FAILURE: Aggregate TPS {actual_tps:.2f} is below target {args.min_tps:.2f}\n")
            return 2

    if report["summary"]["failed_requests"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
