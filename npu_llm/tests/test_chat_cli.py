#!/usr/bin/env python3
"""Tests for npu_llm/chat.py interactive and API client modes."""

from __future__ import annotations

from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
import json
from pathlib import Path
import sys
import threading
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "npu_llm"))

from npu_llm.chat import (  # noqa: E402
    is_local_api_available,
    list_api_models,
    main,
    parse_logit_bias,
    stream_chat_api,
)
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


def test_parse_logit_bias():
    assert parse_logit_bias(None) is None
    assert parse_logit_bias(["1=5", "2=-100"]) == {"1": 5.0, "2": -100.0}
    for invalid in (["1"], ["=1"], ["1="], ["token=high"]):
        try:
            parse_logit_bias(invalid)
        except ValueError:
            continue
        raise AssertionError(f"{invalid!r} was accepted")


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


def test_native_chat_conversation_and_reset():
    calls = []

    class FakeNativeDecoder:
        def __init__(self, model_dir, npu_layers=None):
            assert model_dir == Path("native-model")
            assert npu_layers == 3

        def generate_messages(self, messages, max_new_tokens, sampling=None):
            assert max_new_tokens == 8
            assert sampling == SamplingParams()
            calls.append([dict(message) for message in messages])
            yield "native answer", None
            yield "", {"generated_tokens": 1}

    runtime = ModuleType("runtime.generate")
    runtime.NPUDecoder = FakeNativeDecoder
    output = StringIO()
    with (
        patch.dict(sys.modules, {"runtime.generate": runtime}),
        patch.dict("os.environ", {"HAWKPOINT_API_URL": ""}),
        patch("builtins.input", side_effect=["first", "second", "/reset", "last", "/exit"]),
        redirect_stdout(output),
    ):
        assert main(["--model", "native-model", "--npu-layers", "3", "--max-new-tokens", "8"]) == 0

    system = {"role": "system", "content": "You are a helpful AI assistant."}
    assert calls == [
        [system, {"role": "user", "content": "first"}],
        [system, {"role": "user", "content": "first"},
         {"role": "assistant", "content": "native answer"},
         {"role": "user", "content": "second"}],
        [system, {"role": "user", "content": "last"}],
    ]
    assert output.getvalue().count("SmolLM: native answer") == 3
    assert "Conversation reset." in output.getvalue()


if __name__ == "__main__":
    test_list_api_models()
    test_is_local_api_available()
    test_parse_logit_bias()
    test_stream_chat_api()
    test_main_cli_single_prompt()
    test_native_chat_conversation_and_reset()
    print("PASS 6 chat CLI tests")
