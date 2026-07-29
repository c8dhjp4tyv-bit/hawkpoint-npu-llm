#!/usr/bin/env python3
"""Small OpenAI-compatible HTTP server for the XDNA1 SmolLM2 runtime."""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import gc
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import multiprocessing
import os
from pathlib import Path
import socket
import ssl
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
        if len(content) > 32_768:
            raise ValueError("message content exceeds 32768 characters")
        messages.append({"role": role, "content": content})
    return messages


@dataclass(frozen=True)
class ServerConfig:
    api_key: str
    cors_origins: tuple = (
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    )
    max_body_bytes: int = 1_048_576
    request_timeout: float = 120.0
    queue_capacity: int = 2
    rate_limit_per_minute: int = 30


class RateLimiter:
    def __init__(self, requests_per_minute):
        self.limit = requests_per_minute
        self._requests = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client):
        if self.limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            history = self._requests[client]
            while history and now - history[0] >= 60:
                history.popleft()
            if len(history) >= self.limit:
                return False
            history.append(now)
            return True


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
        warmup = getattr(self._active_decoder, "warmup", None)
        if warmup is not None:
            print(f"Warming {model_id}...", flush=True)
            warmup()
        self._active_model_id = model_id
        return self._active_decoder

    def prewarm_default(self):
        with self.lock:
            self._load(self.default_model_id)

    @property
    def ready(self):
        return True

    def generate(self, model_id, messages, max_tokens, timeout=None):
        with self.lock:
            decoder = self._load(model_id)
            yield from decoder.generate_messages(messages, max_tokens)


class InferenceTimeout(TimeoutError):
    pass


def _worker_loop(models, decoder_factory, connection):
    engine = CompletionEngine(models, decoder_factory=decoder_factory)
    try:
        while True:
            command = connection.recv()
            if command["op"] == "close":
                return
            try:
                if command["op"] == "prewarm":
                    engine.prewarm_default()
                    connection.send(("done", None))
                    continue
                for text, stats in engine.generate(
                    command["model"],
                    command["messages"],
                    command["max_tokens"],
                ):
                    connection.send(("chunk", (text, stats)))
                connection.send(("done", None))
            except Exception as exc:
                logging.exception("inference worker failed")
                connection.send(("error", type(exc).__name__))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        connection.close()


class ProcessCompletionEngine:
    """Run the NPU/XRT context in a killable, restartable worker process."""

    def __init__(self, models, decoder_factory, timeout=120):
        catalog = CompletionEngine(models, decoder_factory=decoder_factory)
        self.models = catalog.models
        self.default_model_id = catalog.default_model_id
        self.decoder_factory = decoder_factory
        self.timeout = timeout
        self._context = multiprocessing.get_context("spawn")
        self._process = None
        self._connection = None
        self._lock = threading.Lock()
        self._ready = False
        self.worker_restarts = 0

    def model_list(self):
        return CompletionEngine(self.models).model_list()

    def has_model(self, model_id):
        return model_id in self.models

    def context_length(self, model_id):
        return int(self.models[model_id].get("context_length", 64))

    @property
    def ready(self):
        return self._ready and self._process is not None and self._process.is_alive()

    def _start(self):
        if self._process is not None and self._process.is_alive():
            return
        parent, child = self._context.Pipe()
        self._process = self._context.Process(
            target=_worker_loop,
            args=(self.models, self.decoder_factory, child),
            daemon=True,
            name="hawkpoint-npu-worker",
        )
        self._process.start()
        child.close()
        self._connection = parent
        self._ready = False

    def _terminate(self, count_restart=True):
        self._ready = False
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)
            self._process.close()
            self._process = None
        if count_restart:
            self.worker_restarts += 1

    def _receive(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._connection.poll(remaining):
            self._terminate()
            raise InferenceTimeout("inference worker exceeded its deadline")
        try:
            return self._connection.recv()
        except (EOFError, BrokenPipeError) as exc:
            self._terminate()
            raise RuntimeError("inference worker exited unexpectedly") from exc

    def prewarm_default(self):
        with self._lock:
            self._start()
            self._connection.send({"op": "prewarm"})
            kind, _ = self._receive(time.monotonic() + self.timeout)
            if kind != "done":
                self._terminate()
                raise RuntimeError("inference worker prewarm failed")
            self._ready = True

    def generate(self, model_id, messages, max_tokens, timeout=None):
        deadline = time.monotonic() + (timeout or self.timeout)
        with self._lock:
            self._start()
            self._connection.send(
                {
                    "op": "generate",
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                }
            )
            try:
                while True:
                    kind, payload = self._receive(deadline)
                    if kind == "chunk":
                        yield payload
                    elif kind == "done":
                        self._ready = True
                        return
                    else:
                        self._ready = False
                        raise RuntimeError("inference worker request failed")
            except GeneratorExit:
                self._terminate()
                raise

    def close(self):
        with self._lock:
            if self._process is None:
                return
            try:
                self._connection.send({"op": "close"})
                self._process.join(timeout=5)
            except (BrokenPipeError, OSError):
                pass
            finally:
                self._terminate(count_restart=False)


@dataclass(frozen=True)
class NPUDecoderFactory:
    npu_layers: int | None = None
    npu_percent: float | None = None

    def __call__(self, path):
        from runtime.generate import NPUDecoder

        metadata = json.loads((Path(path) / "metadata.json").read_text())
        layers = int(metadata["layers"])
        selected = self.npu_layers
        if self.npu_percent is not None:
            selected = round(layers * self.npu_percent / 100.0)
        return NPUDecoder(path, npu_layers=selected)


def make_handler(engine, config=None):
    config = config or ServerConfig(api_key="test-only")
    slots = threading.BoundedSemaphore(config.queue_capacity + 1)
    limiter = RateLimiter(config.rate_limit_per_minute)

    class Handler(BaseHTTPRequestHandler):
        server_version = "HawkPointNPU/1.0"

        def log_message(self, fmt, *args):
            sys.stderr.write(
                f"[{self.log_date_time_string()}] {self.address_string()} "
                f"{fmt % args}\n"
            )

        def _cors(self):
            origin = self.headers.get("Origin")
            if origin and origin in config.cors_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _headers(self, status=200, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self._cors()
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
            self._cors()
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

        def _authorized(self):
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {config.api_key}"
            if hmac.compare_digest(supplied, expected):
                return True
            self._error("missing or invalid bearer token", 401, "authentication_error")
            return False

        def do_OPTIONS(self):
            self._headers(204)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._json(
                    {
                        "status": "ok",
                        "service": "hawkpoint-npu-api",
                    }
                )
                return
            if self.path.rstrip("/") == "/ready":
                ready = engine.ready
                self._json(
                    {
                        "status": "ready" if ready else "not_ready",
                        "models": list(engine.models),
                        "device": "npu1",
                        "worker_restarts": getattr(engine, "worker_restarts", 0),
                    },
                    200 if ready else 503,
                )
                return
            if self.path.rstrip("/") == "/v1/models":
                if not self._authorized():
                    return
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
            if not self._authorized():
                return
            client = self.client_address[0]
            if not limiter.allow(client):
                self._error("rate limit exceeded", 429, "rate_limit_error")
                return
            try:
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    self._error("Content-Length is required", 411)
                    return
                length = int(raw_length)
                if length <= 0:
                    raise ValueError("request body cannot be empty")
                if length > config.max_body_bytes:
                    self._error("request body is too large", 413)
                    return
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
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

            if not slots.acquire(blocking=False):
                self._error("NPU request queue is full", 429, "server_overloaded")
                return
            self.connection.settimeout(config.request_timeout)
            self._deadline = time.monotonic() + config.request_timeout
            try:
                if request.get("stream", False):
                    self._stream(model_id, messages, max_tokens)
                else:
                    self._complete(model_id, messages, max_tokens)
            finally:
                slots.release()

        def _complete(self, model_id, messages, max_tokens):
            pieces = []
            stats = None
            try:
                for text, final_stats in engine.generate(
                    model_id,
                    messages,
                    max_tokens,
                    timeout=config.request_timeout,
                ):
                    if time.monotonic() > self._deadline:
                        raise TimeoutError
                    pieces.append(text)
                    if final_stats is not None:
                        stats = final_stats
            except (TimeoutError, socket.timeout):
                self._error("inference request timed out", 504, "timeout_error")
                return
            except Exception:
                logging.exception("inference request failed")
                self._error("internal inference error", 500, "server_error")
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
                    model_id,
                    messages,
                    max_tokens,
                    timeout=config.request_timeout,
                ):
                    if time.monotonic() > self._deadline:
                        raise TimeoutError
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
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass
            except Exception:
                logging.exception("streaming inference request failed")

    return Handler


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16


def main():
    parser = argparse.ArgumentParser(
        description="Serve SmolLM2 on an AMD Hawk Point NPU"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HAWKPOINT_API_KEY"),
        help="Bearer token (or set HAWKPOINT_API_KEY)",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        help="allowed browser origin; may be repeated",
    )
    parser.add_argument("--max-body-bytes", type=int, default=1_048_576)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--queue-capacity", type=int, default=2)
    parser.add_argument("--rate-limit-per-minute", type=int, default=30)
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        help="serve one converted model directory instead of scanning --models-dir",
    )
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--no-prewarm",
        action="store_true",
        help="defer model compilation until the first completion request",
    )
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
    if not args.api_key:
        parser.error("--api-key or HAWKPOINT_API_KEY is required")
    if args.max_body_bytes < 1024:
        parser.error("--max-body-bytes must be at least 1024")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.queue_capacity < 0:
        parser.error("--queue-capacity cannot be negative")
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("--tls-cert and --tls-key must be provided together")
    if args.npu_percent is not None and not 0 <= args.npu_percent <= 100:
        parser.error("--npu-percent must be between 0 and 100")
    if args.npu_layers is not None and args.npu_layers < 0:
        parser.error("--npu-layers cannot be negative")

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
    decoder_factory = NPUDecoderFactory(
        npu_layers=args.npu_layers,
        npu_percent=args.npu_percent,
    )
    engine = ProcessCompletionEngine(
        models,
        decoder_factory=decoder_factory,
        timeout=args.request_timeout,
    )
    if not args.no_prewarm:
        engine.prewarm_default()
    config = ServerConfig(
        api_key=args.api_key,
        cors_origins=tuple(args.cors_origins or ServerConfig.cors_origins),
        max_body_bytes=args.max_body_bytes,
        request_timeout=args.request_timeout,
        queue_capacity=args.queue_capacity,
        rate_limit_per_minute=args.rate_limit_per_minute,
    )
    server = BoundedHTTPServer(
        (args.host, args.port),
        make_handler(engine, config),
    )
    scheme = "http"
    if args.tls_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print("Installed models: " + ", ".join(models), flush=True)
    print(f"OpenAI-compatible API: {scheme}://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
