#!/usr/bin/env python3
"""Interactive or single-prompt chat with SmolLM2 / Qwen on XDNA1 NPU or via OpenAI API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from runtime.sampling import GREEDY, SamplingParams


def stream_chat_api(
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    sampling: SamplingParams | None = None,
):
    """Stream completions from a running HawkPoint OpenAI-compatible API server."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if sampling is not None and sampling != GREEDY:
        params_dict = sampling.to_dict()
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        ):
            if key in params_dict:
                payload[key] = params_dict[key]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    start_time = time.monotonic()
    first_token_time = None
    generated_tokens = 0
    prompt_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line.removeprefix("data: ")
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if "choices" in data and data["choices"]:
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        generated_tokens += 1
                        yield content, None
                if "usage" in data and data["usage"]:
                    u = data["usage"]
                    if "completion_tokens" in u:
                        generated_tokens = u["completion_tokens"]
                    if "prompt_tokens" in u:
                        prompt_tokens = u["prompt_tokens"]
    except urllib.error.HTTPError as exc:
        err_msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {err_msg}") from exc
    except Exception as exc:
        raise RuntimeError(f"Connection to API failed: {exc}") from exc

    end_time = time.monotonic()
    ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
    decode_duration = (end_time - first_token_time) if first_token_time else (end_time - start_time)
    tps = (generated_tokens / decode_duration) if decode_duration > 0 else 0.0

    stats = {
        "generated_tokens": generated_tokens,
        "prompt_tokens": prompt_tokens,
        "decode_tokens_per_second": tps,
        "ttft_seconds": ttft,
        "total_latency_seconds": end_time - start_time,
    }
    yield "", stats


def list_api_models(base_url: str, api_key: str | None) -> list[str]:
    """Fetch installed model identifiers from the API server."""
    url = f"{base_url.rstrip('/')}/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


def is_local_api_available(url: str = "http://127.0.0.1:8000") -> bool:
    """Check if the local API server responds to health checks."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=0.8):
            return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chat with SmolLM2 or Qwen on XDNA1 NPU or via API")
    p.add_argument(
        "--model",
        default=None,
        help="Model directory path (local NPU) or model identifier string (API mode)",
    )
    p.add_argument(
        "--api-url",
        default=os.environ.get("HAWKPOINT_API_URL"),
        help="Connect to running API server (e.g. http://localhost:8000)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("HAWKPOINT_API_KEY"),
        help="API bearer token; reads from HAWKPOINT_API_KEY when omitted",
    )
    p.add_argument("--prompt", help="Single prompt completion; exits when complete")
    p.add_argument("--max-new-tokens", type=int, default=32)
    offload = p.add_mutually_exclusive_group()
    offload.add_argument("--npu-layers", type=int)
    offload.add_argument("--npu-percent", type=float)
    p.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant.",
    )
    sampling = p.add_argument_group("sampling (default: greedy)")
    sampling.add_argument("--temperature", type=float)
    sampling.add_argument("--top-p", type=float)
    sampling.add_argument("--top-k", type=int)
    sampling.add_argument("--repetition-penalty", type=float)
    sampling.add_argument("--presence-penalty", type=float)
    sampling.add_argument("--frequency-penalty", type=float)
    sampling.add_argument(
        "--seed",
        type=int,
        help="reproduce a sampled run; ignored when decoding greedily",
    )
    args = p.parse_args(argv)

    try:
        params = SamplingParams.build(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            seed=args.seed,
        )
    except ValueError as exc:
        p.error(str(exc))

    api_url = args.api_url
    use_api = bool(api_url)

    if not use_api:
        # Check if local NPU runtime is available
        try:
            from runtime.generate import NPUDecoder  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            if is_local_api_available("http://127.0.0.1:8000"):
                api_url = "http://127.0.0.1:8000"
                use_api = True
                print("Note: Native NPU runtime not imported; connecting to local API server on http://127.0.0.1:8000")
            else:
                sys.exit(
                    "Error: Native NPU hardware runtime is not available and local API server is not running on http://127.0.0.1:8000.\n"
                    "Start the server with: python launcher.py api"
                )

    messages = [{"role": "system", "content": args.system_prompt}]

    if use_api:
        # Determine model identifier
        model_id = args.model
        if not model_id:
            available = list_api_models(api_url, args.api_key)
            model_id = available[0] if available else "smollm2-135m-xdna1"

        def complete(prompt: str) -> dict:
            messages.append({"role": "user", "content": prompt})
            print(f"[{model_id}]: ", end="", flush=True)
            pieces = []
            stats = None
            for text, final_stats in stream_chat_api(
                api_url,
                args.api_key,
                model_id,
                messages,
                args.max_new_tokens,
                sampling=params,
            ):
                pieces.append(text)
                print(text, end="", flush=True)
                if final_stats is not None:
                    stats = final_stats
            print()
            messages.append({"role": "assistant", "content": "".join(pieces)})
            return stats or {}

    else:
        from runtime.generate import NPUDecoder

        model_path = Path(args.model) if args.model else ROOT / "models/SmolLM2-135M-Instruct-xdna1-w8a16"
        npu_layers = args.npu_layers
        if args.npu_percent is not None:
            if not 0 <= args.npu_percent <= 100:
                p.error("--npu-percent must be between 0 and 100")
            layers = int(json.loads((model_path / "metadata.json").read_text())["layers"])
            npu_layers = round(layers * args.npu_percent / 100.0)

        decoder = NPUDecoder(model_path, npu_layers=npu_layers)

        def complete(prompt: str) -> dict:
            messages.append({"role": "user", "content": prompt})
            print("SmolLM: ", end="", flush=True)
            pieces = []
            stats = None
            for text, final_stats in decoder.generate_messages(
                messages, args.max_new_tokens, sampling=params
            ):
                pieces.append(text)
                print(text, end="", flush=True)
                if final_stats is not None:
                    stats = final_stats
            print()
            messages.append({"role": "assistant", "content": "".join(pieces)})
            return stats or {}

    if args.prompt:
        stats = complete(args.prompt)
        tps = stats.get("decode_tokens_per_second", 0.0)
        ttft = stats.get("ttft_seconds", 0.0)
        tokens = stats.get("generated_tokens", 0)
        ram = stats.get("peak_ram_mib")
        ram_str = f" peak_RAM={ram:.1f}MiB" if ram is not None else ""
        print(f"tokens={tokens} tok/s={tps:.2f} TTFT={ttft:.2f}s{ram_str}")
        return 0

    mode_info = f"via API ({api_url})" if use_api else "via local NPU"
    print(f"Interactive chat ({mode_info}). Commands: /reset, /stats, /models, /exit")
    last_stats = None
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break
        if prompt == "/reset":
            messages[:] = messages[:1]
            last_stats = None
            print("Conversation reset.")
            continue
        if prompt == "/stats":
            print(last_stats or "No generation statistics yet.")
            continue
        if prompt == "/models":
            if use_api:
                models = list_api_models(api_url, args.api_key)
                print("Available models:", ", ".join(models) if models else "none found")
            else:
                print(f"Current local model: {args.model}")
            continue
        last_stats = complete(prompt)

    return 0


if __name__ == "__main__":
    sys.exit(main())
