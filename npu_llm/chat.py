#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from npu_llm.runtime.generate import NPUDecoder
from npu_llm.runtime.prompts import seed_messages


def main():
    p = argparse.ArgumentParser(description="Chat with SmolLM2 on the XDNA1 NPU")
    p.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/SmolLM2-135M-Instruct-xdna1-w8a16",
    )
    p.add_argument("--prompt")
    p.add_argument("--max-new-tokens", type=int, default=16)
    offload = p.add_mutually_exclusive_group()
    offload.add_argument("--npu-layers", type=int)
    offload.add_argument("--npu-percent", type=float)
    p.add_argument(
        "--system-prompt",
        help=(
            "system message for the conversation; defaults to the prompt the "
            "loaded checkpoint's family was trained with"
        ),
    )
    args = p.parse_args()

    npu_layers = args.npu_layers
    if args.npu_percent is not None:
        if not 0 <= args.npu_percent <= 100:
            p.error("--npu-percent must be between 0 and 100")
        import json

        layers = int(
            json.loads((args.model / "metadata.json").read_text())["layers"]
        )
        npu_layers = round(layers * args.npu_percent / 100.0)
    # Context-managed so the NPU/XRT contexts are released deterministically
    # when the session ends, rather than at interpreter shutdown.
    with NPUDecoder(args.model, npu_layers=npu_layers) as decoder:
        # An explicit system prompt wins; otherwise the tokenizer supplies the one
        # matching the checkpoint family (SmolLM and Qwen expect different text).
        messages = seed_messages(args.system_prompt)
        history_start = len(messages)
        label = "Qwen" if decoder.model_family == "qwen2" else "SmolLM"

        def complete(prompt):
            messages.append({"role": "user", "content": prompt})
            print(f"{label}: ", end="", flush=True)
            pieces = []
            stats = None
            for text, final_stats in decoder.generate_messages(
                messages, args.max_new_tokens
            ):
                pieces.append(text)
                print(text, end="", flush=True)
                if final_stats is not None:
                    stats = final_stats
            print()
            messages.append({"role": "assistant", "content": "".join(pieces)})
            return stats

        if args.prompt:
            stats = complete(args.prompt)
            print(
                f"tokens={stats['generated_tokens']} "
                f"tok/s={stats['decode_tokens_per_second']:.2f} "
                f"TTFT={stats['ttft_seconds']:.2f}s "
                f"peak_RAM={stats['peak_ram_mib']:.1f}MiB"
            )
            return

        print("Interactive NPU chat. Commands: /reset, /stats, /exit")
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
                messages[:] = messages[:history_start]
                last_stats = None
                print("Conversation reset.")
                continue
            if prompt == "/stats":
                print(last_stats or "No generation statistics yet.")
                continue
            last_stats = complete(prompt)


if __name__ == "__main__":
    main()
