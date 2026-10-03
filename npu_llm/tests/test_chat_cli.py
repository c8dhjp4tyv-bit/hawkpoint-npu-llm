#!/usr/bin/env python3
"""Tests for npu_llm/chat.py interactive and API client modes."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "npu_llm"))

from npu_llm.chat import is_local_api_available, list_api_models, main, stream_chat_api  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402


class MockAPIServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Quiet logs during test execution

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            return
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {"id": "smollm2-135m-xdna1", "object": "model"},
                            {"id": "qwen2.5-0.5b-xdna1", "object": "model"},
                        ],
                    }
                ).encode("utf-8")
            )
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            assert body.get("stream") is True

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # First SSE chunk
            chunk1 = {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "Hello"}}],
            }
            self.wfile.write(f"data: {json.dumps(chunk1)}\n\n".encode("utf-8"))
            self.wfile.flush()

            # Second SSE chunk with usage
            chunk2 = {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": " from NPU!"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }
            self.wfile.write(f"data: {json.dumps(chunk2)}\n\n".encode("utf-8"))
            self.wfile.flush()

            # Terminal [DONE] marker
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self.send_response(404)
        self.end_headers()


def run_mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockAPIServerHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


def test_list_api_models():
    server, base_url = run_mock_server()
    try:
        models = list_api_models(base_url, api_key="test-key")
        assert models == ["smollm2-135m-xdna1", "qwen2.5-0.5b-xdna1"]
    finally:
        server.shutdown()


def test_is_local_api_available():
    server, base_url = run_mock_server()
    try:
        assert is_local_api_available(base_url) is True
        assert is_local_api_available("http://127.0.0.1:1") is False
    finally:
        server.shutdown()


def test_stream_chat_api():
    server, base_url = run_mock_server()
    try:
        messages = [{"role": "user", "content": "Hi"}]
        tokens = []
        final_stats = None
        for text, stats in stream_chat_api(
            base_url,
            api_key="test-key",
            model="smollm2-135m-xdna1",
            messages=messages,
            max_tokens=16,
            sampling=SamplingParams.build(temperature=0.7),
        ):
            if text:
                tokens.append(text)
            if stats:
                final_stats = stats

        assert tokens == ["Hello", " from NPU!"]
        assert final_stats is not None
        assert final_stats["generated_tokens"] == 2
        assert final_stats["prompt_tokens"] == 4
        assert final_stats["ttft_seconds"] >= 0.0
        assert final_stats["decode_tokens_per_second"] >= 0.0
    finally:
        server.shutdown()


def test_main_cli_single_prompt(capsys=None):
    server, base_url = run_mock_server()
    try:
        ret = main(
            [
                "--prompt",
                "Hello test",
                "--api-url",
                base_url,
                "--api-key",
                "secret-token",
                "--max-new-tokens",
                "8",
            ]
        )
        assert ret == 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    test_list_api_models()
    test_is_local_api_available()
    test_stream_chat_api()
    test_main_cli_single_prompt()
    print("PASS 4 chat CLI tests")
