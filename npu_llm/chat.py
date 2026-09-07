#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from runtime.generate import NPUDecoder
from runtime.sampling import SamplingParams


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
        default="You are a helpful AI assistant named SmolLM.",
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
    args = p.parse_args()

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

    npu_layers = args.npu_layers
    if args.npu_percent is not None:
        if not 0 <= args.npu_percent <= 100:
            p.error("--npu-percent must be between 0 and 100")
        import json

        layers = int(
            json.loads((args.model / "metadata.json").read_text())["layers"]
        )
        npu_layers = round(layers * args.npu_percent / 100.0)
    decoder = NPUDecoder(args.model, npu_layers=npu_layers)
    messages = [{"role": "system", "content": args.system_prompt}]

    def complete(prompt):
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
            messages[:] = messages[:1]
            last_stats = None
            print("Conversation reset.")
            continue
        if prompt == "/stats":
            print(last_stats or "No generation statistics yet.")
            continue
        last_stats = complete(prompt)


if __name__ == "__main__":
    main()
