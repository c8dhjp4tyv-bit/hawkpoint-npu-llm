#!/usr/bin/env python3
"""Security, protocol, overload, streaming, and shutdown tests."""

from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
from pathlib import Path
import sys
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
    ServerConfig,
    make_handler,
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
            statuses = {first.result()[0], second.result()[0]}
        assert statuses == {200, 429}
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


def main():
    test_protocol_and_security()
    test_backpressure_and_safe_errors()
    test_hard_process_timeout()
    test_worker_error_forces_restart()
    print("PASS API security and protocol")


if __name__ == "__main__":
    main()
