#!/usr/bin/env python3
"""Protocol smoke test for the OpenAI-compatible server."""

import json
from pathlib import Path
import sys
import threading
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu_llm.api_server import CompletionEngine, make_handler
from http.server import ThreadingHTTPServer


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


def fetch(url, data=None):
    request = Request(
        url,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        return response.status, response.read().decode()


def main():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(CompletionEngine(FakeDecoder())),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body = fetch(f"{base}/v1/models")
        assert status == 200
        assert json.loads(body)["data"][0]["id"] == "smollm2-135m-xdna1"

        payload = {
            "model": "smollm2-135m-xdna1",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
        }
        status, body = fetch(f"{base}/v1/chat/completions", payload)
        response = json.loads(body)
        assert status == 200
        assert response["choices"][0]["message"]["content"] == "Hello!"
        assert response["choices"][0]["finish_reason"] == "length"
        assert response["usage"]["total_tokens"] == 6

        payload["stream"] = True
        status, body = fetch(f"{base}/v1/chat/completions", payload)
        assert status == 200
        assert '"content":"Hello"' in body
        assert '"finish_reason":"length"' in body
        assert "data: [DONE]" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print("PASS API server")


if __name__ == "__main__":
    main()
