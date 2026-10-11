#!/usr/bin/env python3
"""Example client script demonstrating interaction with the HawkPoint NPU API server.

This example uses Python's standard library only (no external pip dependencies needed).
It shows:
1. Server discovery and installed model listing
2. Streaming chat completion with real-time SSE token parsing and TTFT metrics
3. Non-streaming chat completion
4. Text completion with prompt echoing
5. Server health and OpenMetrics inspection
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu_llm.http_client import open_api_request


def list_models(base_url: str, api_key: str | None = None) -> list[dict]:
    """Retrieve the list of installed NPU models."""
    url = f"{base_url.rstrip('/')}/v1/models"
    req = urllib.request.Request(url, headers={"User-Agent": "HawkPoint-Example-Client/1.0"})

    with open_api_request(req, api_key, timeout=10.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("data", [])


def stream_chat(
    base_url: str,
    prompt: str,
    model: str = "smollm2-135m-xdna1",
    api_key: str | None = None,
    temperature: float = 0.7,
) -> dict:
    """Stream chat completion tokens in real-time and calculate timing metrics."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "HawkPoint-Example-Client/1.0",
        },
    )

    start_time = time.perf_counter()
    ttft = None
    accumulated_text = []
    generated_tokens = 0
    usage = None

    print(f"\n[Prompt]: {prompt}")
    print("[Assistant]: ", end="", flush=True)

    with open_api_request(req, api_key, timeout=60.0) as resp:
        for line_bytes in resp:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if "error" in chunk:
                raise RuntimeError(f"API stream failed: {chunk['error']}")

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    print(content, end="", flush=True)
                    accumulated_text.append(content)
                    generated_tokens += 1

            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]

    total_time = time.perf_counter() - start_time
    print()  # newline after completion

    # Fallback token count from usage if available
    if usage and "completion_tokens" in usage:
        generated_tokens = usage["completion_tokens"]

    tps = generated_tokens / total_time if total_time > 0 else 0.0

    metrics = {
        "text": "".join(accumulated_text),
        "generated_tokens": generated_tokens,
        "total_time_seconds": total_time,
        "ttft_seconds": ttft or 0.0,
        "tokens_per_second": tps,
        "usage": usage,
    }
    print(f"\n--- Metrics: {generated_tokens} tokens | {tps:.2f} tok/s | TTFT: {ttft or 0.0:.3f}s ---")
    return metrics


def complete_text(
    base_url: str,
    prompt: str,
    model: str = "smollm2-135m-xdna1",
    api_key: str | None = None,
    echo: bool = False,
) -> str:
    """Execute non-streaming text completion via /v1/completions."""
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "echo": echo,
        "max_tokens": 32,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "HawkPoint-Example-Client/1.0",
        },
    )

    with open_api_request(req, api_key, timeout=30.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices", [])
        return choices[0]["text"] if choices else ""


def check_health(base_url: str) -> dict:
    """Query readiness and health probe endpoints."""
    url = f"{base_url.rstrip('/')}/ready"
    req = urllib.request.Request(url, headers={"User-Agent": "HawkPoint-Example-Client/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return {"status": "not_ready", "code": err.code}


class MockServerHandler(BaseHTTPRequestHandler):
    """Lightweight in-process HTTP handler for self-test validation."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": "smollm2-135m-xdna1"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/ready":
            body = json.dumps({"status": "ready", "device": "npu1"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(content_len).decode("utf-8"))

        if self.path == "/v1/chat/completions" and data.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            tokens = ["Hello", " from", " Hawk", "Point", " NPU!"]
            for tok in tokens:
                chunk = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": tok}, "index": 0}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
            usage_chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(tokens), "total_tokens": 10 + len(tokens)},
            }
            self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        if self.path == "/v1/completions":
            prompt = data.get("prompt", "")
            echo = data.get("echo", False)
            text = (prompt if echo else "") + " completed text"
            resp = {
                "id": "cmpl-mock",
                "object": "text_completion",
                "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            }
            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()


def run_self_test() -> bool:
    """Run an automated mock end-to-end self test."""
    print("Running self-test mode with local mock server...")
    server = HTTPServer(("127.0.0.1", 0), MockServerHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    try:
        # 1. Health
        health = check_health(base_url)
        assert health["status"] == "ready"

        # 2. Models
        models = list_models(base_url)
        assert len(models) == 1
        assert models[0]["id"] == "smollm2-135m-xdna1"

        # 3. Stream chat
        metrics = stream_chat(base_url, "Tell me a joke")
        assert "Hello from HawkPoint NPU!" in metrics["text"]
        assert metrics["generated_tokens"] == 5

        # 4. Text completion with echo
        out = complete_text(base_url, "Prefix:", echo=True)
        assert out == "Prefix: completed text"

        print("\nAll self-test assertions PASSED successfully!")
        return True
    finally:
        server.shutdown()
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HawkPoint NPU API interaction example client.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the API server")
    parser.add_argument("--api-key", default=os.environ.get("HAWKPOINT_API_KEY"), help="API bearer token")
    parser.add_argument("--prompt", default="Why is on-device NPU inference beneficial for privacy and latency?", help="Prompt")
    parser.add_argument("--model", default="smollm2-135m-xdna1", help="Model ID")
    parser.add_argument("--self-test", action="store_true", help="Run automated self-test using an internal mock server")
    args = parser.parse_args(argv)

    if args.self_test:
        success = run_self_test()
        return 0 if success else 1

    print(f"Checking server status at {args.url}...")
    health = check_health(args.url)
    print(f"Server health: {health.get('status', 'unknown')}")

    try:
        models = list_models(args.url, api_key=args.api_key)
        print(f"Installed models: {[m['id'] for m in models]}")
    except Exception as exc:
        print(f"Could not retrieve model list: {exc}")

    print("\nExecuting streaming chat completion...")
    try:
        stream_chat(args.url, args.prompt, model=args.model, api_key=args.api_key)
    except Exception as exc:
        print(f"Error during streaming chat: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
