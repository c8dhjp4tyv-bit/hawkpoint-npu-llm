#!/usr/bin/env python3
"""Small OpenAI-compatible HTTP server for the XDNA1 SmolLM2 runtime."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parent
MODEL_ID = "smollm2-135m-xdna1"


def _completion_id():
    return f"chatcmpl-{uuid.uuid4().hex}"


def _validate_messages(value):
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty array")
    messages = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role!r}")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        messages.append({"role": role, "content": content})
    return messages


class CompletionEngine:
    """Serialize access to one physical NPU and expose completion helpers."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.lock = threading.Lock()

    def generate(self, messages, max_tokens):
        with self.lock:
            yield from self.decoder.generate_messages(messages, max_tokens)


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HawkPointNPU/1.0"

        def log_message(self, fmt, *args):
            sys.stderr.write(
                f"[{self.log_date_time_string()}] {self.address_string()} "
                f"{fmt % args}\n"
            )

        def _headers(self, status=200, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Allow-Headers",
                "Authorization, Content-Type",
            )
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def _json(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, message, status=400):
            self._json(
                {
                    "error": {
                        "message": str(message),
                        "type": "invalid_request_error",
                    }
                },
                status,
            )

        def do_OPTIONS(self):
            self._headers(204)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._json({"status": "ok", "model": MODEL_ID, "device": "npu1"})
                return
            if self.path.rstrip("/") == "/v1/models":
                self._json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": MODEL_ID,
                                "object": "model",
                                "created": 0,
                                "owned_by": "local",
                            }
                        ],
                    }
                )
                return
            self._error("not found", 404)

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._error("not found", 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                messages = _validate_messages(request.get("messages"))
                requested = int(request.get("max_tokens", 16))
                max_tokens = min(
                    max(1, requested),
                    engine.decoder.context_length - 1,
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._error(exc)
                return

            if request.get("stream", False):
                self._stream(messages, max_tokens)
            else:
                self._complete(messages, max_tokens)

        def _complete(self, messages, max_tokens):
            pieces = []
            stats = None
            try:
                for text, final_stats in engine.generate(messages, max_tokens):
                    pieces.append(text)
                    if final_stats is not None:
                        stats = final_stats
            except Exception as exc:
                self._error(exc, 500)
                return
            stats = stats or {}
            prompt_tokens = int(stats.get("prompt_tokens", 0))
            completion_tokens = int(stats.get("generated_tokens", 0))
            finish_reason = stats.get("finish_reason", "stop")
            self._json(
                {
                    "id": _completion_id(),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "".join(pieces),
                            },
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "x_hawkpoint_stats": stats,
                }
            )

        def _stream(self, messages, max_tokens):
            completion_id = _completion_id()
            created = int(time.time())
            self._headers(200, "text/event-stream")

            def send(payload):
                data = json.dumps(payload, separators=(",", ":"))
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()

            try:
                send(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                stats = None
                for text, final_stats in engine.generate(messages, max_tokens):
                    if text:
                        send(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_ID,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    if final_stats is not None:
                        stats = final_stats
                send(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": (stats or {}).get(
                                    "finish_reason", "stop"
                                ),
                            }
                        ],
                        "x_hawkpoint_stats": stats or {},
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description="Serve SmolLM2 on an AMD Hawk Point NPU"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/SmolLM2-135M-Instruct-xdna1-w8a16",
    )
    args = parser.parse_args()

    from runtime.generate import NPUDecoder

    print(f"Loading {MODEL_ID} from {args.model} on XDNA1...", flush=True)
    engine = CompletionEngine(NPUDecoder(args.model))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"OpenAI-compatible API: http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
