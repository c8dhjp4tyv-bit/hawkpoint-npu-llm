#!/usr/bin/env python3
"""Security, protocol, overload, streaming, and shutdown tests."""

from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu_llm.api_server import (  # noqa: E402
    BoundedHTTPServer,
    CompletionEngine,
    InferenceTimeout,
    ProcessCompletionEngine,
    RateLimiter,
    ServerConfig,
    make_handler,
    serve,
)


API_KEY = "integration-test-secret"


class FakeDecoder:
    context_length = 64

    def generate_messages(self, messages, max_new_tokens):
        assert messages[-1]["content"] == "Hello"
        assert max_new_tokens == 5
        yield "Hello", None
        yield "!", None
        yield "", {
            "prompt_tokens": 4,
            "generated_tokens": 2,
            "finish_reason": "length",
            "decode_tokens_per_second": 2.5,
        }


class RecordingDecoder:
    """Echoes back the sampling configuration the server forwarded."""

    context_length = 64

    def __init__(self):
        self.calls = []

    def generate_messages(self, messages, max_new_tokens, sampling=None):
        self.calls.append(sampling)
        yield "ok", None
        yield "", {
            "prompt_tokens": 1,
            "generated_tokens": 1,
            "finish_reason": "stop",
            "sampling": sampling,
        }


class FailingDecoder:
    context_length = 64

    def generate_messages(self, messages, max_new_tokens):
        raise RuntimeError("secret internal detail")
        yield


class SlowDecoder:
    context_length = 64

    def generate_messages(self, messages, max_new_tokens):
        time.sleep(0.35)
        yield "done", {"generated_tokens": 1}


class HangingDecoder:
    context_length = 64

    def generate_messages(self, messages, max_new_tokens):
        time.sleep(30)
        yield "unreachable", None


class ToggleDecoder:
    """Fails while a marker file exists, otherwise completes normally.

    Used to prove a worker killed by an inference error is recreated on the
    next request. The marker is checked at generate time so the same pickled
    instance behaves differently across worker restarts.
    """

    context_length = 64

    def __init__(self, marker):
        self.marker = marker

    def generate_messages(self, messages, max_new_tokens):
        if Path(self.marker).exists():
            raise RuntimeError("secret internal detail")
        yield "ok", None
        yield "", {
            "prompt_tokens": 1,
            "generated_tokens": 1,
            "finish_reason": "stop",
            "decode_tokens_per_second": 1.0,
        }


class CloseTrackingDecoder:
    """Records close() calls so model-switch cleanup can be asserted offline."""

    context_length = 64

    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.closed = False

    def generate_messages(self, messages, max_new_tokens):
        yield self.name, None
        yield "", {
            "prompt_tokens": 1,
            "generated_tokens": 1,
            "finish_reason": "stop",
        }

    def close(self):
        self.closed = True
        self.events.append(("close", self.name))


class TrackingFactory:
    def __init__(self, events):
        self.events = events
        self.created = []

    def __call__(self, path):
        decoder = CloseTrackingDecoder(str(path), self.events)
        self.created.append(decoder)
        self.events.append(("create", str(path)))
        return decoder


def fetch(url, data=None, *, api_key=API_KEY, origin=None, raw=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if origin:
        headers["Origin"] = origin
    request = Request(
        url,
        data=raw if raw is not None else (
            json.dumps(data).encode() if data is not None else None
        ),
        headers=headers,
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, response.read().decode(), dict(response.headers)
    except HTTPError as exc:
        return exc.code, exc.read().decode(), dict(exc.headers)


def payload(model="smollm2-135m-xdna1"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
    }


def start_server(models, **config):
    server = BoundedHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            CompletionEngine(models),
            ServerConfig(api_key=API_KEY, **config),
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_protocol_and_security():
    server, thread, base = start_server(
        {
            "smollm2-135m-xdna1": FakeDecoder(),
            "smollm-135m-xdna1": FakeDecoder(),
        },
        max_body_bytes=512,
    )
    try:
        status, _, _ = fetch(f"{base}/v1/models", api_key=None)
        assert status == 401
        status, body, _ = fetch(f"{base}/ready", api_key=None)
        assert status == 200
        assert json.loads(body)["status"] == "ready"

        status, body, headers = fetch(
            f"{base}/v1/models",
            origin="http://localhost:3000",
        )
        assert status == 200
        assert len(json.loads(body)["data"]) == 2
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        status, _, headers = fetch(
            f"{base}/v1/models",
            origin="https://attacker.example",
        )
        assert status == 200
        assert "Access-Control-Allow-Origin" not in headers

        status, body, _ = fetch(
            f"{base}/v1/models/smollm2-135m-xdna1?source=sdk"
        )
        assert status == 200
        assert json.loads(body)["id"] == "smollm2-135m-xdna1"
        status, body, _ = fetch(f"{base}/v1/models/not-installed")
        assert status == 404
        assert json.loads(body)["error"]["type"] == "model_not_found"

        status, body, _ = fetch(f"{base}/v1/chat/completions", payload())
        response = json.loads(body)
        assert status == 200
        assert response["choices"][0]["message"]["content"] == "Hello!"
        assert response["usage"]["total_tokens"] == 6

        streaming = payload()
        streaming["stream"] = True
        status, body, _ = fetch(f"{base}/v1/chat/completions", streaming)
        assert status == 200
        assert '"content":"Hello"' in body
        assert "data: [DONE]" in body

        status, _, _ = fetch(
            f"{base}/v1/chat/completions",
            payload("not-installed"),
        )
        assert status == 404
        status, _, _ = fetch(
            f"{base}/v1/chat/completions",
            raw=b"{broken",
        )
        assert status == 400
        status, _, _ = fetch(
            f"{base}/v1/chat/completions",
            raw=b" " * 513,
        )
        assert status == 413
    finally:
        stop_server(server, thread)


def test_backpressure_and_safe_errors():
    server, thread, base = start_server(
        {"smollm2-135m-xdna1": SlowDecoder()},
        queue_capacity=0,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(fetch, f"{base}/v1/chat/completions", payload())
            time.sleep(0.05)
            second = pool.submit(fetch, f"{base}/v1/chat/completions", payload())
            results = (first.result(), second.result())
            statuses = {result[0] for result in results}
        assert statuses == {200, 429}
        overloaded = next(result for result in results if result[0] == 429)
        assert overloaded[2]["Retry-After"] == "1"
    finally:
        stop_server(server, thread)

    server, thread, base = start_server(
        {"smollm2-135m-xdna1": FailingDecoder()}
    )
    try:
        status, body, _ = fetch(f"{base}/v1/chat/completions", payload())
        assert status == 500
        assert "internal inference error" in body
        assert "secret internal detail" not in body

        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.connect()
        connection.close()
    finally:
        stop_server(server, thread)


def test_hard_process_timeout():
    engine = ProcessCompletionEngine(
        {"smollm2-135m-xdna1": HangingDecoder()},
        decoder_factory=None,
        timeout=0.2,
    )
    started = time.monotonic()
    try:
        try:
            list(
                engine.generate(
                    "smollm2-135m-xdna1",
                    [{"role": "user", "content": "Hello"}],
                    5,
                    timeout=0.2,
                )
            )
            raise AssertionError("hanging worker unexpectedly completed")
        except InferenceTimeout:
            pass
        assert time.monotonic() - started < 3
        assert engine.worker_restarts == 1
        assert not engine.ready
    finally:
        engine.close()


def test_worker_error_forces_restart():
    engine = ProcessCompletionEngine(
        {"smollm2-135m-xdna1": FailingDecoder()},
        decoder_factory=None,
        timeout=2,
    )
    try:
        try:
            list(
                engine.generate(
                    "smollm2-135m-xdna1",
                    [{"role": "user", "content": "Hello"}],
                    5,
                )
            )
            raise AssertionError("failing worker unexpectedly completed")
        except RuntimeError as exc:
            assert str(exc) == "inference worker request failed"
        assert engine.worker_restarts == 1
        assert engine._process is None
        assert not engine.ready
    finally:
        engine.close()


def test_inference_error_recreates_worker():
    """An inference error must kill the worker; the next request recreates it."""
    messages = [{"role": "user", "content": "Hello"}]
    with tempfile.TemporaryDirectory() as directory:
        marker = Path(directory) / "fail"
        engine = ProcessCompletionEngine(
            {"smollm2-135m-xdna1": ToggleDecoder(str(marker))},
            decoder_factory=None,
            timeout=5,
        )
        try:
            # Healthy worker first, so we can prove the PID actually changes.
            first = list(engine.generate("smollm2-135m-xdna1", messages, 1))
            assert [text for text, _ in first if text] == ["ok"]
            healthy_pid = engine._process.pid
            assert engine.worker_restarts == 0

            # Force the next inference to fail; the worker must be torn down.
            marker.touch()
            try:
                list(engine.generate("smollm2-135m-xdna1", messages, 1))
                raise AssertionError("failing worker unexpectedly completed")
            except RuntimeError as exc:
                assert str(exc) == "inference worker request failed"
            assert engine.worker_restarts == 1
            assert engine._process is None
            assert not engine.ready

            # Recovery: the next request must spawn a brand-new worker process.
            marker.unlink()
            recovered = list(engine.generate("smollm2-135m-xdna1", messages, 1))
            assert [text for text, _ in recovered if text] == ["ok"]
            assert engine._process is not None and engine._process.is_alive()
            assert engine._process.pid != healthy_pid
            # A successful request must not itself count as a restart.
            assert engine.worker_restarts == 1
        finally:
            engine.close()


def test_model_switch_releases_previous_decoder():
    """Switching models must deterministically close the previous decoder."""
    messages = [{"role": "user", "content": "Hello"}]
    events = []
    factory = TrackingFactory(events)
    engine = CompletionEngine(
        {
            "model-a": {"path": "model-a", "context_length": 64},
            "model-b": {"path": "model-b", "context_length": 64},
        },
        decoder_factory=factory,
    )
    try:
        list(engine.generate("model-a", messages, 1))
        list(engine.generate("model-b", messages, 1))
        # model-a must be closed before model-b is created.
        assert events == [
            ("create", "model-a"),
            ("close", "model-a"),
            ("create", "model-b"),
        ]
        assert factory.created[0].closed is True
        assert factory.created[1].closed is False
    finally:
        engine.close()
    # Closing the engine releases the still-active decoder.
    assert factory.created[1].closed is True


def test_sampling_parameters_reach_the_decoder():
    decoder = RecordingDecoder()
    server, thread, base = start_server({"smollm2-135m-xdna1": decoder})
    try:
        # A request with no sampling fields keeps the historical greedy path:
        # the decoder is called without a sampling argument at all.
        status, body, _ = fetch(f"{base}/v1/chat/completions", payload())
        assert status == 200
        assert decoder.calls == [None]
        assert json.loads(body)["x_hawkpoint_stats"]["sampling"] is None

        sampled = payload()
        sampled.update({"temperature": 0.8, "top_p": 0.9, "top_k": 40, "seed": 7})
        status, body, _ = fetch(f"{base}/v1/chat/completions", sampled)
        assert status == 200
        forwarded = decoder.calls[-1]
        assert forwarded["temperature"] == 0.8
        assert forwarded["top_p"] == 0.9
        assert forwarded["top_k"] == 40
        assert forwarded["seed"] == 7
        assert forwarded["repetition_penalty"] == 1.0
        assert json.loads(body)["x_hawkpoint_stats"]["sampling"] == forwarded

        # Streaming takes the same path.
        streaming = payload()
        streaming.update({"stream": True, "temperature": 1.0})
        status, body, _ = fetch(f"{base}/v1/chat/completions", streaming)
        assert status == 200
        assert decoder.calls[-1]["temperature"] == 1.0

        # Explicit defaults are still recognized as greedy.
        neutral = payload()
        neutral.update({"temperature": 0, "top_p": 1.0})
        status, _, _ = fetch(f"{base}/v1/chat/completions", neutral)
        assert status == 200
        assert decoder.calls[-1] is None

        before = len(decoder.calls)
        for invalid in (
            {"temperature": 5},
            {"temperature": "hot"},
            {"top_p": 0},
            {"top_k": -3},
            {"repetition_penalty": 0},
            {"presence_penalty": 99},
            {"seed": -1},
            {"n": 2},
        ):
            request = payload()
            request.update(invalid)
            status, body, _ = fetch(f"{base}/v1/chat/completions", request)
            assert status == 400, (invalid, status)
            assert json.loads(body)["error"]["type"] == "invalid_request_error"
        # A rejected request never reaches the NPU worker.
        assert len(decoder.calls) == before
    finally:
        stop_server(server, thread)



def test_strict_request_types():
    decoder = RecordingDecoder()
    server, thread, base = start_server(
        {"smollm2-135m-xdna1": decoder}, rate_limit_per_minute=0
    )
    try:
        invalid = [
            {"model": value} for value in ([], {}, 123, False, "")
        ] + [
            {"max_tokens": value} for value in (0, -1, True, 1.5, "5", None)
        ] + [
            {"max_completion_tokens": value}
            for value in (0, -1, True, 1.5, "5", None)
        ] + [
            {"stream": value} for value in ("false", 1, [], None)
        ] + [
            {"n": True}, {"n": 1.0},
            {"max_tokens": 5, "max_completion_tokens": 5},
            {"messages": [{"role": [], "content": "Hello"}]},
            {"stream_options": {"include_usage": True}},
            {"stream": True, "stream_options": []},
            {"stream": True, "stream_options": {"include_usage": "yes"}},
            {"stream": True, "stream_options": {"unknown": True}},
        ]
        for fields in invalid:
            status, body, _ = fetch(
                f"{base}/v1/chat/completions", {**payload(), **fields}
            )
            assert status == 400, (fields, status, body)
            assert json.loads(body)["error"]["type"] == "invalid_request_error"
        assert decoder.calls == []
        status, _, _ = fetch(f"{base}/v1/models", api_key="invalid-é")
        assert status == 401
        # Invalid clients must not leave the server unable to serve valid work.
        assert fetch(f"{base}/v1/chat/completions", payload())[0] == 200
    finally:
        stop_server(server, thread)


def test_modern_token_alias_routes_and_headers():
    decoder = RecordingDecoder()
    server, thread, base = start_server({"smollm2-135m-xdna1": decoder})
    try:
        request = payload()
        del request["max_tokens"]
        request["max_completion_tokens"] = 5
        status, _, headers = fetch(
            f"{base}/v1/chat/completions?client=test", request
        )
        assert status == 200
        assert headers["X-Request-ID"].startswith("req-")
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Cache-Control"] == "no-store"

        status, _, model_headers = fetch(f"{base}/v1/models?limit=1")
        assert status == 200
        assert model_headers["X-Request-ID"].startswith("req-")
        assert model_headers["X-Request-ID"] != headers["X-Request-ID"]

        request["stream"] = True
        status, _, stream_headers = fetch(
            f"{base}/v1/chat/completions?client=test", request
        )
        assert status == 200
        assert stream_headers["X-Accel-Buffering"] == "no"
    finally:
        stop_server(server, thread)


def test_rate_limiter_evicts_idle_clients():
    now = [0.0]
    limiter = RateLimiter(1, clock=lambda: now[0])
    assert limiter.allow("first")
    assert not limiter.allow("first")
    now[0] = 61.0
    assert limiter.allow("second")
    assert "first" not in limiter._requests


class ServeLifecycleEngine:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_serve_closes_resources():
    engine = ServeLifecycleEngine()
    server = BoundedHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            CompletionEngine({"smollm2-135m-xdna1": FakeDecoder()}),
            ServerConfig(api_key=API_KEY),
        ),
    )
    thread = threading.Thread(target=serve, args=(server, engine), daemon=True)
    thread.start()
    try:
        status, _, _ = fetch(
            f"http://127.0.0.1:{server.server_port}/health", api_key=None
        )
        assert status == 200
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert engine.closed


def test_stream_usage_and_errors():
    server, thread, base = start_server({"smollm2-135m-xdna1": FakeDecoder()})
    try:
        for options in (None, {"include_usage": False}, {"include_usage": True}):
            request = {**payload(), "stream": True}
            if options is not None:
                request["stream_options"] = options
            status, body, _ = fetch(f"{base}/v1/chat/completions", request)
            assert status == 200
            frames = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
            assert frames[-1] == "[DONE]"
            chunks = [json.loads(frame) for frame in frames[:-1]]
            assert len({chunk["id"] for chunk in chunks}) == 1
            if options and options["include_usage"]:
                assert chunks[-1]["choices"] == []
                assert chunks[-1]["usage"] == {
                    "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6
                }
                assert all(chunk["usage"] is None for chunk in chunks[:-1])
                assert chunks[-2]["choices"][0]["finish_reason"] == "length"
            else:
                assert all("usage" not in chunk for chunk in chunks)
    finally:
        stop_server(server, thread)

    server, thread, base = start_server({"smollm2-135m-xdna1": FailingDecoder()})
    try:
        status, body, _ = fetch(
            f"{base}/v1/chat/completions", {**payload(), "stream": True}
        )
        assert status == 200  # Headers precede inference; the error is an SSE event.
        assert '"type":"server_error"' in body
        assert "secret internal detail" not in body
        assert "[DONE]" not in body
    finally:
        stop_server(server, thread)


class CleanupDecoder:
    context_length = 64

    def __init__(self):
        self.closed = threading.Event()

    def generate_messages(self, messages, max_new_tokens):
        try:
            time.sleep(0.08)
            yield "late", None
        finally:
            self.closed.set()


def test_timeout_closes_generation():
    for stream in (False, True):
        decoder = CleanupDecoder()
        server, thread, base = start_server(
            {"smollm2-135m-xdna1": decoder}, request_timeout=0.04
        )
        try:
            status, body, _ = fetch(
                f"{base}/v1/chat/completions", {**payload(), "stream": stream}
            )
            assert status == (200 if stream else 504)
            assert "timeout_error" in body
            assert "late" not in body
            assert decoder.closed.wait(1)
            assert "[DONE]" not in body
        finally:
            stop_server(server, thread)


def test_body_read_timeout():
    decoder = RecordingDecoder()
    server, thread, base = start_server(
        {"smollm2-135m-xdna1": decoder}, request_timeout=0.1
    )
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.putrequest("POST", "/v1/chat/completions")
        connection.putheader("Authorization", f"Bearer {API_KEY}")
        connection.putheader("Content-Length", "100")
        connection.endheaders()
        connection.send(b"{")
        response = connection.getresponse()
        assert response.status == 408
        assert json.loads(response.read())["error"]["type"] == "timeout_error"
        assert decoder.calls == []
    finally:
        connection.close()
        stop_server(server, thread)


def test_cancelled_process_generation_recovers():
    engine = ProcessCompletionEngine(
        {"smollm2-135m-xdna1": FakeDecoder()}, decoder_factory=None, timeout=5
    )
    try:
        generation = engine.generate("smollm2-135m-xdna1", payload()["messages"], 5)
        assert next(generation)[0] == "Hello"
        generation.close()
        assert engine._process is None
        assert not engine.ready
        assert engine.worker_restarts == 1
        recovered = list(engine.generate("smollm2-135m-xdna1", payload()["messages"], 5))
        assert "".join(text for text, _ in recovered) == "Hello!"
        assert engine.ready
    finally:
        engine.close()

def main():
    test_serve_closes_resources()
    test_modern_token_alias_routes_and_headers()
    test_rate_limiter_evicts_idle_clients()
    test_strict_request_types()
    test_stream_usage_and_errors()
    test_timeout_closes_generation()
    test_body_read_timeout()
    test_cancelled_process_generation_recovers()
    test_protocol_and_security()
    test_sampling_parameters_reach_the_decoder()
    test_backpressure_and_safe_errors()
    test_hard_process_timeout()
    test_worker_error_forces_restart()
    test_inference_error_recreates_worker()
    test_model_switch_releases_previous_decoder()
    print("PASS API security and protocol")


if __name__ == "__main__":
    main()
