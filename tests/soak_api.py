#!/usr/bin/env python3
"""Long-running model-switching smoke test for a separately started API."""

import json
import os
from urllib.request import Request, urlopen


BASE_URL = os.environ.get("HAWKPOINT_API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ["HAWKPOINT_API_KEY"]
COMPLETIONS = int(os.environ.get("HAWKPOINT_SOAK_COMPLETIONS", "1000"))


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(
        BASE_URL + path,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(req, timeout=180) as response:
        return json.load(response)


models = [item["id"] for item in request("/v1/models")["data"]]
if not models:
    raise SystemExit("no installed models")
for index in range(COMPLETIONS):
    model = models[index % len(models)]
    result = request(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"Reply OK. Run {index}."}],
            "max_tokens": 4,
        },
    )
    if not result["choices"]:
        raise RuntimeError(f"completion {index} returned no choices")
    if index and index % 100 == 0:
        print(f"completed {index}/{COMPLETIONS}", flush=True)
print(f"PASS {COMPLETIONS} completions across {len(models)} model(s)")
