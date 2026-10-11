#!/usr/bin/env python3
"""Tests for scripts/benchmark_api.py."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.benchmark_api import (  # noqa: E402
    fetch_metrics,
    format_markdown_report,
    format_text_report,
    main,
    run_benchmark,
)


class MockBenchmarkAPIServer(BaseHTTPRequestHandler):
    tokens_served = 0

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            content = (
                f"# TYPE hawkpoint_tokens_generated_total counter\n"
                f'hawkpoint_tokens_generated_total{{model="test-m"}} {MockBenchmarkAPIServer.tokens_served}\n'
            )
            self.wfile.write(content.encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        stream = body.get("stream", False)
        MockBenchmarkAPIServer.tokens_served += 4

        if self.path in ("/v1/chat/completions", "/v1/completions"):
            self.send_response(200)
            if stream:
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk1 = {
                    "choices": [{"index": 0, "delta": {"content": "Hello"}}],
                }
                self.wfile.write(f"data: {json.dumps(chunk1)}\n\n".encode("utf-8"))
                self.wfile.flush()
                chunk2 = {
                    "choices": [{"index": 0, "delta": {"content": " world"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
                }
                self.wfile.write(f"data: {json.dumps(chunk2)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp = {
                    "id": "cmpl-1",
                    "choices": [{"index": 0, "message": {"content": "Hello world"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
                }
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()


def run_test_server():
    MockBenchmarkAPIServer.tokens_served = 0
    server = HTTPServer(("127.0.0.1", 0), MockBenchmarkAPIServer)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


def test_fetch_metrics():
    server, base_url = run_test_server()
    try:
        metrics = fetch_metrics(base_url)
        assert any("hawkpoint_tokens_generated_total" in k for k in metrics)
    finally:
        server.shutdown()


def test_run_benchmark_streaming():
    server, base_url = run_test_server()
    try:
        report = run_benchmark(
            base_url=base_url,
            model="test-m",
            prompt="Hello",
            max_tokens=4,
            num_requests=3,
            concurrency=2,
            stream=True,
        )
        assert report["summary"]["total_requests"] == 3
        assert report["summary"]["successful_requests"] == 3
        assert report["summary"]["failed_requests"] == 0
        assert report["summary"]["total_generated_tokens"] == 12
        assert report["latency_seconds"]["p50"] is not None
        assert report["ttft_seconds"]["p50"] is not None
        assert report["tokens_per_second"]["mean"] is not None
        assert any("hawkpoint_tokens_generated_total" in k for k in report["metrics_diff"])

        text_out = format_text_report(report)
        assert "HawkPoint NPU API Benchmark: test-m" in text_out
        assert "Time To First Token" in text_out

        md_out = format_markdown_report(report)
        assert "| **Aggregate Throughput** |" in md_out
    finally:
        server.shutdown()


def test_run_benchmark_non_streaming():
    server, base_url = run_test_server()
    try:
        report = run_benchmark(
            base_url=base_url,
            model="test-m",
            prompt="Hello",
            max_tokens=4,
            num_requests=2,
            concurrency=1,
            stream=False,
            endpoint="/v1/completions",
        )
        assert report["summary"]["successful_requests"] == 2
        assert report["ttft_seconds"] is None
    finally:
        server.shutdown()


def test_main_cli_and_reports():
    server, base_url = run_test_server()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "report.md"
            ret = main(
                [
                    "--url",
                    base_url,
                    "-n",
                    "2",
                    "-c",
                    "1",
                    "--format",
                    "markdown",
                    "-o",
                    str(out_file),
                    "--min-tps",
                    "1.0",
                ]
            )
            assert ret == 0
            assert out_file.exists()
            assert "| **Target Endpoint** |" in out_file.read_text(encoding="utf-8")
    finally:
        server.shutdown()


if __name__ == "__main__":
    test_fetch_metrics()
    test_run_benchmark_streaming()
    test_run_benchmark_non_streaming()
    test_main_cli_and_reports()
    print("PASS 4 API benchmark tests")
