#!/usr/bin/env python3
"""Small OpenAI-compatible HTTP server for the XDNA1 SmolLM2 runtime."""

import argparse
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parent
try:
    from .model_catalog import DEFAULT_MODEL_ID, discover_models
except ImportError:
    from model_catalog import DEFAULT_MODEL_ID, discover_models


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
    """Lazily load selected models and serialize access to one physical NPU."""

    def __init__(self, models, decoder_factory=None):
        if not models:
            raise ValueError("no converted models were found")
        self.models = {}
        for model_id, record in models.items():
            if hasattr(record, "generate_messages"):
                record = {
                    "decoder": record,
                    "display_name": model_id,
                    "context_length": record.context_length,
                }
            self.models[model_id] = dict(record)
        self.default_model_id = (
            DEFAULT_MODEL_ID
            if DEFAULT_MODEL_ID in self.models
            else next(iter(self.models))
        )
        self.decoder_factory = decoder_factory
        self._active_model_id = None
        self._active_decoder = None
        self.lock = threading.Lock()

    def model_list(self):
        return [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "name": record.get("display_name", model_id),
            }
            for model_id, record in self.models.items()
        ]

    def has_model(self, model_id):
        return model_id in self.models

    def context_length(self, model_id):
        return int(self.models[model_id].get("context_length", 64))

    def _load(self, model_id):
        record = self.models[model_id]
        if "decoder" in record:
            return record["decoder"]
        if self._active_model_id == model_id:
            return self._active_decoder
        if self.decoder_factory is None:
            raise RuntimeError("no decoder factory configured")
        self._active_decoder = None
        self._active_model_id = None
        gc.collect()
        print(f"Loading {model_id} from {record['path']} on XDNA1...", flush=True)
        self._active_decoder = self.decoder_factory(record["path"])
        self._active_model_id = model_id
        return self._active_decoder

    def generate(self, model_id, messages, max_tokens):
        with self.lock:
            decoder = self._load(model_id)
            yield from decoder.generate_messages(messages, max_tokens)


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

        def _error(self, message, status=400, error_type="invalid_request_error"):
            self._json(
                {
                    "error": {
                        "message": str(message),
                        "type": error_type,
                    }
                },
                status,
            )

        def do_OPTIONS(self):
            self._headers(204)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._json(
                    {
                        "status": "ok",
                        "models": list(engine.models),
                        "device": "npu1",
                    }
                )
                return
            if self.path.rstrip("/") == "/v1/models":
                self._json(
                    {
                        "object": "list",
                        "data": engine.model_list(),
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
                model_id = request.get("model") or engine.default_model_id
                if not engine.has_model(model_id):
                    self._error(
                        f"model {model_id!r} is not installed",
                        404,
                        "model_not_found",
                    )
                    return
                requested = int(request.get("max_tokens", 16))
                max_tokens = min(
                    max(1, requested),
                    engine.context_length(model_id) - 1,
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._error(exc)
                return

            if request.get("stream", False):
                self._stream(model_id, messages, max_tokens)
            else:
                self._complete(model_id, messages, max_tokens)

        def _complete(self, model_id, messages, max_tokens):
            pieces = []
            stats = None
            try:
                for text, final_stats in engine.generate(
                    model_id, messages, max_tokens
                ):
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
                    "model": model_id,
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

        def _stream(self, model_id, messages, max_tokens):
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
                        "model": model_id,
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
                for text, final_stats in engine.generate(
                    model_id, messages, max_tokens
                ):
                    if text:
                        send(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
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
                        "model": model_id,
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
        help="serve one converted model directory instead of scanning --models-dir",
    )
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    offload = parser.add_mutually_exclusive_group()
    offload.add_argument(
        "--npu-layers",
        type=int,
        help="exact number of leading decoder layers to run on the NPU",
    )
    offload.add_argument(
        "--npu-percent",
        type=float,
        help="percentage of leading decoder layers to run on the NPU",
    )
    args = parser.parse_args()
    if args.npu_percent is not None and not 0 <= args.npu_percent <= 100:
        parser.error("--npu-percent must be between 0 and 100")
    if args.npu_layers is not None and args.npu_layers < 0:
        parser.error("--npu-layers cannot be negative")

    from runtime.generate import NPUDecoder

    if args.model:
        discovered = discover_models(args.model.parent)
        matching = {
            model_id: record
            for model_id, record in discovered.items()
            if record["path"].resolve() == args.model.resolve()
        }
        models = matching or {
            DEFAULT_MODEL_ID: {
                "path": args.model,
                "display_name": DEFAULT_MODEL_ID,
                "context_length": 64,
            }
        }
    else:
        models = discover_models(args.models_dir)
    if not models:
        parser.error(
            "no converted models found; run scripts/prepare_model.py first"
        )
    def decoder_factory(path):
        metadata = json.loads((Path(path) / "metadata.json").read_text())
        layers = int(metadata["layers"])
        npu_layers = args.npu_layers
        if args.npu_percent is not None:
            npu_layers = round(layers * args.npu_percent / 100.0)
        return NPUDecoder(path, npu_layers=npu_layers)

    engine = CompletionEngine(models, decoder_factory=decoder_factory)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print("Installed models: " + ", ".join(models), flush=True)
    print(f"OpenAI-compatible API: http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
