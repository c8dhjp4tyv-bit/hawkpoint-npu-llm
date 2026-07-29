#!/usr/bin/env python3
"""Compare an autoregressive Qwen token sequence on CPU BF16 and XDNA1."""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.generate import NPUDecoder  # noqa: E402


PROMPT = [{"role": "user", "content": "Reply with exactly OK."}]


def sequence(model_dir, npu_layers, count):
    decoder = NPUDecoder(model_dir, npu_layers=npu_layers)
    if npu_layers:
        decoder.warmup()
    prompt_ids = decoder.tokenizer.encode_chat(PROMPT)
    prompt_ids = prompt_ids[-(decoder.context_length - count) :]
    next_token = None
    timings = []
    position = 0
    for token_id in prompt_ids:
        next_token, elapsed = decoder.decode_token(token_id, position)
        timings.append(elapsed)
        position += 1
    generated = []
    for index in range(count):
        generated.append(next_token)
        if index + 1 < count:
            next_token, elapsed = decoder.decode_token(next_token, position)
            timings.append(elapsed)
            position += 1
    return generated, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 32 <= args.tokens <= 63:
        parser.error("--tokens must be between 32 and 63")

    cpu_tokens, cpu_times = sequence(args.model_dir, 0, args.tokens)
    gc.collect()
    npu_tokens, npu_times = sequence(args.model_dir, 24, args.tokens)
    if cpu_tokens != npu_tokens:
        differences = [
            index
            for index, pair in enumerate(zip(cpu_tokens, npu_tokens))
            if pair[0] != pair[1]
        ]
        raise AssertionError(f"Qwen token mismatch at positions {differences}")
    report = {
        "schema_version": 1,
        "prompt": PROMPT,
        "token_count": args.tokens,
        "expected_cpu_bf16_token_ids": cpu_tokens,
        "actual_npu_token_ids": npu_tokens,
        "agreement": args.tokens,
        "cpu_median_seconds": statistics.median(cpu_times),
        "npu_median_seconds": statistics.median(npu_times),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
