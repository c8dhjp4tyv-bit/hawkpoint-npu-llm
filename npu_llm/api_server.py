#!/usr/bin/env python3
"""Small OpenAI-compatible HTTP server for the XDNA1 SmolLM2 runtime."""

import argparse
from collections import defaultdict, deque
from contextlib import closing
from dataclasses import dataclass
import gc
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import ssl
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
try:
    from .model_catalog import DEFAULT_MODEL_ID, discover_models
except ImportError:
    from model_catalog import DEFAULT_MODEL_ID, discover_models

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.sampling import GREEDY, SamplingParams  # noqa: E402
from runtime.stopping import normalize_logprobs, normalize_stop  # noqa: E402


def _completion_id():
    """Create an opaque identifier for one chat completion."""
    return f"chatcmpl-{uuid.uuid4().hex}"


def _validate_messages(value):
    """Validate supported roles and bounded text-only message content."""
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty array")
    messages = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role!r}")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        if len(content) > 32_768:
            raise ValueError("message content exceeds 32768 characters")
        messages.append({"role": role, "content": content})
    return messages


def _validate_completion_options(request):
    """Validate protocol types before admitting any inference work."""
    model = request.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise ValueError("model must be a non-empty string")
    if "max_tokens" in request and "max_completion_tokens" in request:
        raise ValueError(
            "max_tokens and max_completion_tokens are mutually exclusive"
        )
    token_field = (
        "max_completion_tokens"
        if "max_completion_tokens" in request
        else "max_tokens"
    )
    requested = request.get(token_field, 16)
    if type(requested) is not int or requested <= 0:
        raise ValueError(f"{token_field} must be a positive integer")
    if type(request.get("n", 1)) is not int or request.get("n", 1) != 1:
        raise ValueError("only integer n=1 is supported")
    stream = request.get("stream", False)
    if type(stream) is not bool:
        raise ValueError("stream must be a boolean")
    options = request.get("stream_options")
    if options is not None:
        if not isinstance(options, dict):
            raise ValueError("stream_options must be an object")
        if not stream:
            raise ValueError("stream_options requires stream=true")
        if set(options) - {"include_usage"}:
            raise ValueError("unsupported stream_options field")
        if type(options.get("include_usage", False)) is not bool:
            raise ValueError("stream_options.include_usage must be a boolean")
    return requested, bool(options and options.get("include_usage"))


def _validate_output_options(request):
    """Validate ``stop`` and log-probability options.

    Returns ``(stop, top_logprobs)`` where ``stop`` is a tuple of strings and
    ``top_logprobs`` is ``None`` when log probabilities were not requested.
    """
    stop = normalize_stop(request.get("stop"))
    enabled = request.get("logprobs", False)
    if enabled is None:
        enabled = False
    if type(enabled) is not bool:
        raise ValueError("logprobs must be a boolean")
    top = request.get("top_logprobs")
    if top is not None and not enabled:
        raise ValueError("top_logprobs requires logprobs=true")
    if not enabled:
        return stop, None
    return stop, normalize_logprobs(0 if top is None else top)


def _usage(stats):
    """Translate decoder token counts into the OpenAI usage structure."""
    prompt = int(stats.get("prompt_tokens", 0))
    completion = int(stats.get("generated_tokens", 0))
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if "cached_prompt_tokens" in stats:
        usage["prompt_tokens_details"] = {
            "cached_tokens": int(stats["cached_prompt_tokens"])
        }
    return usage


def _generation_options(sampling=None, stop=None, logprobs=None):
    """Keyword arguments for ``generate_messages``, omitting unused options.

    Decoders that predate an option are still driven with their historical
    signature as long as a request does not use that option.
    """
    options = {}
    if sampling is not None:
        options["sampling"] = sampling
    if stop:
        options["stop"] = list(stop)
    if logprobs is not None:
        options["logprobs"] = logprobs
    return options


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
    graceful_shutdown_timeout: float = 130.0


class InferenceTracker:
    """Reject new inference during shutdown and wait for admitted work."""

    def __init__(self):
        """Initialize atomic admission and drain coordination."""
        self._active = 0
        self._draining = False
        self._condition = threading.Condition()

    @property
    def draining(self):
        """Return whether new inference admission has been disabled."""
        with self._condition:
            return self._draining

    @property
    def active(self):
        """Return the number of admitted requests, including queued work."""
        with self._condition:
            return self._active

    def try_enter(self):
        """Atomically admit a request unless shutdown has begun."""
        with self._condition:
            if self._draining:
                return False
            self._active += 1
            return True

    def leave(self):
        """Release one admission and notify drain waiters when all work ends."""
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("inference tracker leave without enter")
            self._active -= 1
            if self._active == 0:
                self._condition.notify_all()

    def begin_shutdown(self):
        """Permanently disable new admission and notify drain waiters."""
        with self._condition:
            self._draining = True
            self._condition.notify_all()

    def wait(self, timeout):
        """Wait up to timeout seconds for all admitted inference to finish."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


class RateLimiter:
    def __init__(self, requests_per_minute, clock=time.monotonic):
        """Initialize client histories and an injectable monotonic clock."""
        self.limit = requests_per_minute
        self._clock = clock
        self._requests = defaultdict(deque)
        self._lock = threading.Lock()
        self._next_cleanup = 0.0

    def allow(self, client):
        """Apply a rolling one-minute client limit and periodically prune idle clients."""
        if self.limit <= 0:
            return True
        now = self._clock()
        with self._lock:
            if now >= self._next_cleanup:
                stale = [
                    key
                    for key, entries in self._requests.items()
                    if not entries or now - entries[-1] >= 60
                ]
                for key in stale:
                    del self._requests[key]
                self._next_cleanup = now + 60
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
        """Normalize installed model records and initialize serialized decoder ownership."""
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
        """Return OpenAI-compatible metadata for every installed model."""
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

    def model_info(self, model_id):
        """Return metadata for one installed model, or None if absent."""
        if model_id not in self.models:
            return None
        record = self.models[model_id]
        return {
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "name": record.get("display_name", model_id),
        }

    def has_model(self, model_id):
        """Check whether the requested model is present in the catalog."""
        return model_id in self.models

    def context_length(self, model_id):
        """Return the configured model context length, defaulting to 64."""
        return int(self.models[model_id].get("context_length", 64))

    def _close_active(self):
        """Deterministically release the currently loaded decoder, if any.

        The active decoder owns the physical NPU's XRT hardware contexts. They
        must be released before another model loads (or when the worker exits),
        otherwise the driver's system-wide context pool leaks and eventually
        fails ``CREATE_HWCTX``. Never rely on garbage collection alone here.
        """
        decoder = self._active_decoder
        self._active_decoder = None
        self._active_model_id = None
        if decoder is not None:
            close = getattr(decoder, "close", None)
            if close is not None:
                close()
        gc.collect()

    def _load(self, model_id):
        """Reuse the active decoder or close it before loading and warming another."""
        record = self.models[model_id]
        if "decoder" in record:
            return record["decoder"]
        if self._active_model_id == model_id:
            return self._active_decoder
        if self.decoder_factory is None:
            raise RuntimeError("no decoder factory configured")
        # Release the previous model's NPU/XRT context before loading the next.
        self._close_active()
        print(f"Loading {model_id} from {record['path']} on XDNA1...", flush=True)
        self._active_decoder = self.decoder_factory(record["path"])
        warmup = getattr(self._active_decoder, "warmup", None)
        if warmup is not None:
            print(f"Warming {model_id}...", flush=True)
            warmup()
        self._active_model_id = model_id
        return self._active_decoder

    def close(self):
        """Release any loaded decoder and its NPU/XRT resources."""
        self._close_active()

    def prewarm_default(self):
        """Load and warm the default model before admitting normal traffic."""
        with self.lock:
            self._load(self.default_model_id)

    @property
    def ready(self):
        """Report readiness for the in-process test engine."""
        return True

    def generate(
        self,
        model_id,
        messages,
        max_tokens,
        timeout=None,
        sampling=None,
        stop=None,
        logprobs=None,
    ):
        """Serialize model inference and yield text chunks with final statistics."""
        with self.lock:
            decoder = self._load(model_id)
            yield from decoder.generate_messages(
                messages,
                max_tokens,
                **_generation_options(sampling, stop, logprobs),
            )


class InferenceTimeout(TimeoutError):
    pass


def _worker_loop(models, decoder_factory, connection):
    """Serve parent commands and release decoder resources on worker exit."""
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
                # Options are revalidated here rather than trusted from the
                # parent process, like the sampling parameters.
                stop = normalize_stop(command.get("stop"))
                logprobs = normalize_logprobs(command.get("logprobs"))
                for chunk in engine.generate(
                    command["model"],
                    command["messages"],
                    command["max_tokens"],
                    sampling=command.get("sampling"),
                    stop=stop,
                    logprobs=logprobs,
                ):
                    connection.send(("chunk", tuple(chunk)))
                connection.send(("done", None))
            except Exception as exc:
                logging.exception("inference worker failed")
                connection.send(("error", type(exc).__name__))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            engine.close()
        except Exception:
            logging.exception("failed to release NPU resources on worker exit")
        connection.close()


class ProcessCompletionEngine:
    """Run the NPU/XRT context in a killable, restartable worker process."""

    def __init__(self, models, decoder_factory, timeout=120):
        """Initialize the spawn context and independent generation/lifecycle locks."""
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
        # Worker ownership must be independent of the lock held across yields.
        self._lifecycle_lock = threading.RLock()
        self._closed = False

    def model_list(self):
        """Return OpenAI-compatible metadata for every installed model."""
        return CompletionEngine(self.models).model_list()

    def model_info(self, model_id):
        """Return metadata for one installed model, or None if absent."""
        return CompletionEngine(self.models).model_info(model_id)

    def has_model(self, model_id):
        """Check whether the requested model is present in the catalog."""
        return model_id in self.models

    def context_length(self, model_id):
        """Return the configured model context length, defaulting to 64."""
        return int(self.models[model_id].get("context_length", 64))

    @property
    def ready(self):
        """Return readiness without racing worker teardown."""
        with self._lifecycle_lock:
            return (
                not self._closed and self._ready
                and self._process is not None and self._process.is_alive()
            )

    def _start(self):
        """Start a worker unless shutdown has permanently closed this engine."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("inference engine is closed")
            self._start_locked()

    def _start_locked(self):
        """Create worker and pipe while holding lifecycle ownership."""
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
        """Serialize worker disposal against cancellation and startup."""
        with self._lifecycle_lock:
            self._terminate_locked(count_restart and not self._closed)

    def _terminate_locked(self, count_restart):
        """Dispose the worker with bounded terminate/kill waits."""
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
        """Wait on a stable pipe reference and normalize cancellation errors."""
        remaining = deadline - time.monotonic()
        connection = self._connection
        if connection is None:
            raise RuntimeError("inference engine is closed")
        try:
            if remaining <= 0 or not connection.poll(remaining):
                self._terminate()
                raise InferenceTimeout("inference worker exceeded its deadline")
            return connection.recv()
        except InferenceTimeout:
            raise
        except (EOFError, OSError, ValueError) as exc:
            self._terminate()
            raise RuntimeError("inference worker exited unexpectedly") from exc

    def prewarm_default(self):
        """Load and warm the default model before admitting normal traffic."""
        with self._lock:
            self._send_command({"op": "prewarm"})
            kind, _ = self._receive(time.monotonic() + self.timeout)
            if kind != "done":
                self._terminate()
                raise RuntimeError("inference worker prewarm failed")
            self._ready = True

    def generate(
        self,
        model_id,
        messages,
        max_tokens,
        timeout=None,
        sampling=None,
        stop=None,
        logprobs=None,
    ):
        """Serialize model inference and yield text chunks with final statistics."""
        deadline = time.monotonic() + (timeout or self.timeout)
        with self._lock:
            self._send_command(
                {
                    "op": "generate",
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    # Sent as a plain dict, never a live object, and revalidated
                    # inside the worker before it reaches the decoder.
                    "sampling": sampling,
                    "stop": list(stop) if stop else None,
                    "logprobs": logprobs,
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
                        self._terminate()
                        raise RuntimeError("inference worker request failed")
            except GeneratorExit:
                self._terminate()
                raise

    def _send_command(self, command):
        """Start and dispatch atomically against permanent engine cancellation."""
        with self._lifecycle_lock:
            self._start()
            self._connection.send(command)

    def close(self):
        """Close an idle engine; use abort when the drain deadline expires."""
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

    def abort(self):
        """Cancel inference without waiting for its generation lock.

        Closing is permanent so already queued calls cannot recreate a worker.
        Lifecycle locking keeps pipe/process disposal single-owner; cleanup
        uses at most the existing two five-second process waits.
        """
        with self._lifecycle_lock:
            self._closed = True
            self._terminate_locked(count_restart=False)


@dataclass(frozen=True)
class NPUDecoderFactory:
    npu_layers: int | None = None
    npu_percent: float | None = None

    def __call__(self, path):
        """Construct an NPU decoder with the selected layer-offload configuration."""
        from runtime.generate import NPUDecoder

        metadata = json.loads((Path(path) / "metadata.json").read_text())
        layers = int(metadata["layers"])
        selected = self.npu_layers
        if self.npu_percent is not None:
            selected = round(layers * self.npu_percent / 100.0)
        return NPUDecoder(path, npu_layers=selected)


def make_handler(engine, config=None):
    """Build handlers sharing authentication, admission, and drain state."""
    config = config or ServerConfig(api_key="test-only")
    slots = threading.BoundedSemaphore(config.queue_capacity + 1)
    limiter = RateLimiter(config.rate_limit_per_minute)
    inference_tracker = InferenceTracker()

    class Handler(BaseHTTPRequestHandler):
        server_version = "HawkPointNPU/1.0"
        tracker = inference_tracker
        server_config = config

        def setup(self):
            """Initialize request identity and apply the socket idle timeout."""
            super().setup()
            self.request_id = f"req-{uuid.uuid4().hex}"
            # Bound header/body reads too, before inference admission.
            self.connection.settimeout(config.request_timeout)

        def log_message(self, fmt, *args):
            """Write an access-log entry correlated with the response request ID."""
            sys.stderr.write(
                f"[{self.log_date_time_string()}] {self.request_id} "
                f"{self.address_string()} "
                f"{fmt % args}\n"
            )

        def _cors(self):
            """Allow browser access only for an explicitly configured origin."""
            origin = self.headers.get("Origin")
            if origin and origin in config.cors_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _common_headers(self):
            """Attach diagnostic, cache-control, and browser-visible headers."""
            self.send_header("X-Request-ID", self.request_id)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Access-Control-Expose-Headers",
                "X-Request-ID, Retry-After",
            )

        def _headers(self, status=200, content_type="application/json"):
            """Start an empty or streaming response, disabling SSE proxy buffering."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self._common_headers()
            if content_type == "text/event-stream":
                self.send_header("X-Accel-Buffering", "no")
            self._cors()
            self.send_header(
                "Access-Control-Allow-Headers",
                "Authorization, Content-Type",
            )
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def _json(self, payload, status=200, extra_headers=None):
            """Serialize one JSON response with its length and optional extra headers."""
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._common_headers()
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _error(
            self,
            message,
            status=400,
            error_type="invalid_request_error",
            extra_headers=None,
        ):
            """Return a structured API error with a chosen HTTP status."""
            self._json(
                {
                    "error": {
                        "message": str(message),
                        "type": error_type,
                    }
                },
                status,
                extra_headers,
            )

        def _authorized(self):
            """Check the bearer token in constant time or send an authentication error."""
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {config.api_key}"
            if hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
                return True
            self._error("missing or invalid bearer token", 401, "authentication_error")
            return False

        def do_OPTIONS(self):
            """Return the supported CORS preflight methods and headers."""
            self._headers(204)

        def do_GET(self):
            """Serve liveness, readiness, and authenticated model metadata routes."""
            path = urlsplit(self.path).path.rstrip("/")
            if path == "/health":
                self._json(
                    {
                        "status": "ok",
                        "service": "hawkpoint-npu-api",
                    }
                )
                return
            if path == "/ready":
                ready = engine.ready and not inference_tracker.draining
                self._json(
                    {
                        "status": "ready" if ready else "not_ready",
                        "draining": inference_tracker.draining,
                        "active_requests": inference_tracker.active,
                        "models": list(engine.models),
                        "device": "npu1",
                        "worker_restarts": getattr(engine, "worker_restarts", 0),
                    },
                    200 if ready else 503,
                )
                return
            if path == "/v1/models":
                if not self._authorized():
                    return
                self._json(
                    {
                        "object": "list",
                        "data": engine.model_list(),
                    }
                )
                return
            if path.startswith("/v1/models/"):
                if not self._authorized():
                    return
                model_id = path.removeprefix("/v1/models/")
                model = engine.model_info(model_id)
                if model is None:
                    self._error(
                        f"model {model_id!r} is not installed",
                        404,
                        "model_not_found",
                    )
                    return
                self._json(model)
                return
            self._error("not found", 404)

        def do_POST(self):
            """Validate and admit a chat request before driving completion generation."""
            if urlsplit(self.path).path.rstrip("/") != "/v1/chat/completions":
                self._error("not found", 404)
                return
            if not self._authorized():
                return
            if inference_tracker.draining:
                self._error(
                    "server is shutting down",
                    503,
                    "server_unavailable",
                    {"Retry-After": "1"},
                )
                return
            client = self.client_address[0]
            if not limiter.allow(client):
                self._error(
                    "rate limit exceeded",
                    429,
                    "rate_limit_error",
                    {"Retry-After": "60"},
                )
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
                requested, include_usage = _validate_completion_options(request)
                stop, logprobs = _validate_output_options(request)
                messages = _validate_messages(request.get("messages"))
                model_id = request.get("model") or engine.default_model_id
                if not engine.has_model(model_id):
                    self._error(
                        f"model {model_id!r} is not installed",
                        404,
                        "model_not_found",
                    )
                    return
                max_tokens = min(
                    requested,
                    engine.context_length(model_id) - 1,
                )
                params = SamplingParams.from_request(request)
                # A request that asks for nothing but the defaults keeps the
                # original greedy call path, so decoders that predate sampling
                # are still driven with their historical signature.
                sampling = None if params == GREEDY else params.to_dict()
            except (TimeoutError, socket.timeout):
                self._error("request body timed out", 408, "timeout_error")
                return
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._error(exc)
                return

            if not slots.acquire(blocking=False):
                self._error(
                    "NPU request queue is full",
                    429,
                    "server_overloaded",
                    {"Retry-After": "1"},
                )
                return
            if not inference_tracker.try_enter():
                slots.release()
                self._error(
                    "server is shutting down",
                    503,
                    "server_unavailable",
                    {"Retry-After": "1"},
                )
                return
            self.connection.settimeout(config.request_timeout)
            self._deadline = time.monotonic() + config.request_timeout
            output = {"stop": stop, "logprobs": logprobs}
            try:
                if request.get("stream", False):
                    self._stream(
                        model_id,
                        messages,
                        max_tokens,
                        sampling,
                        include_usage,
                        **output,
                    )
                else:
                    self._complete(
                        model_id, messages, max_tokens, sampling, **output
                    )
            finally:
                inference_tracker.leave()
                slots.release()

        def _generation(self, model_id, messages, max_tokens, sampling, stop, logprobs):
            """Start engine generation, forwarding only the options in use."""
            return engine.generate(
                model_id,
                messages,
                max_tokens,
                timeout=config.request_timeout,
                **_generation_options(sampling, stop, logprobs),
            )

        def _complete(
            self,
            model_id,
            messages,
            max_tokens,
            sampling=None,
            stop=(),
            logprobs=None,
        ):
            """Collect one completion and return token usage or a sanitized error."""
            pieces = []
            entries = []
            stats = None
            try:
                with closing(self._generation(
                    model_id, messages, max_tokens, sampling, stop, logprobs
                )) as generation:
                    for text, final_stats, *extra in generation:
                        if time.monotonic() > self._deadline:
                            raise TimeoutError
                        pieces.append(text)
                        if extra:
                            entries.extend(extra[0])
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
                            "logprobs": (
                                None
                                if logprobs is None
                                else {"content": entries}
                            ),
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": _usage(stats),
                    "x_hawkpoint_stats": stats,
                }
            )

        def _stream(
            self,
            model_id,
            messages,
            max_tokens,
            sampling=None,
            include_usage=False,
            stop=(),
            logprobs=None,
        ):
            """Emit SSE chunks, optional usage, and a success or error terminator."""
            completion_id = _completion_id()
            created = int(time.time())
            self._headers(200, "text/event-stream")

            def send(payload):
                """Write and flush a JSON SSE event with optional usage metadata."""
                if include_usage and "choices" in payload:
                    payload.setdefault("usage", None)
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
                with closing(self._generation(
                    model_id, messages, max_tokens, sampling, stop, logprobs
                )) as generation:
                    for text, final_stats, *extra in generation:
                        if time.monotonic() > self._deadline:
                            raise TimeoutError
                        entries = extra[0] if extra else []
                        if text or entries:
                            choice = {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                            if logprobs is not None:
                                choice["logprobs"] = {"content": entries}
                            send(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_id,
                                    "choices": [choice],
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
                if include_usage:
                    send({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [],
                        "usage": _usage(stats or {}),
                    })
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                logging.exception("streaming inference request failed")
                timed_out = isinstance(exc, TimeoutError)
                try:
                    send({"error": {
                        "message": "inference request timed out" if timed_out else "internal inference error",
                        "type": "timeout_error" if timed_out else "server_error",
                    }})
                except (BrokenPipeError, ConnectionResetError, socket.timeout):
                    pass

    return Handler


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16


def serve(server, engine):
    """Serve until interrupted and release HTTP plus NPU resources cleanly."""
    previous_handlers = {}
    handler = server.RequestHandlerClass
    tracker = getattr(handler, "tracker", None)
    shutdown_timeout = getattr(
        getattr(handler, "server_config", None),
        "graceful_shutdown_timeout",
        130.0,
    )

    def request_shutdown(signum, frame):
        """Begin admission drain and stop the server from a separate thread."""
        if tracker is not None:
            tracker.begin_shutdown()
        # shutdown() must run outside serve_forever()'s thread.
        threading.Thread(
            target=server.shutdown,
            daemon=True,
            name="hawkpoint-api-shutdown",
        ).start()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_shutdown)
    try:
        server.serve_forever()
    finally:
        if tracker is not None:
            tracker.begin_shutdown()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        server.server_close()
        if tracker is not None and not tracker.wait(shutdown_timeout):
            logging.warning(
                "graceful shutdown deadline expired with active inference"
            )
            abort = getattr(engine, "abort", None)
            if abort is not None:
                abort()
            else:
                engine.close()
        else:
            engine.close()


def main():
    """Validate CLI settings, create the worker and HTTP server, and serve traffic."""
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
    parser.add_argument(
        "--graceful-shutdown-timeout",
        type=float,
        default=130,
        help="seconds to drain admitted inference before worker termination",
    )
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
    if args.rate_limit_per_minute < 0:
        parser.error("--rate-limit-per-minute cannot be negative")
    if args.graceful_shutdown_timeout <= 0:
        parser.error("--graceful-shutdown-timeout must be positive")
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
        graceful_shutdown_timeout=args.graceful_shutdown_timeout,
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
    serve(server, engine)


if __name__ == "__main__":
    main()
