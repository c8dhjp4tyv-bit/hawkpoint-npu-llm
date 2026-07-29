#!/usr/bin/env python3
"""Compare the fused Qwen XDNA1 path with the NumPy BF16 reference."""

import argparse
import gc
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.generate import NPUDecoder


def run_prefix(model_dir, npu_layers, token_ids):
    decoder = NPUDecoder(model_dir, npu_layers=npu_layers)
    if npu_layers:
        decoder.warmup()
    output = []
    timings = []
    for position, token_id in enumerate(token_ids):
        next_token, elapsed = decoder.decode_token(token_id, position)
        output.append(next_token)
        timings.append(elapsed)
    return output, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--positions",
        type=int,
        default=12,
        help="number of ChatML prompt positions to compare",
    )
    args = parser.parse_args()

    tokenizer_decoder = NPUDecoder(args.model_dir, npu_layers=0)
    token_ids = tokenizer_decoder.tokenizer.encode_chat(
        [{"role": "user", "content": "Hello"}]
    )[: args.positions]
    del tokenizer_decoder
    gc.collect()

    cpu_tokens, cpu_times = run_prefix(args.model_dir, 0, token_ids)
    gc.collect()
    npu_tokens, npu_times = run_prefix(args.model_dir, 24, token_ids)

    if cpu_tokens != npu_tokens:
        differences = [
            index
            for index, pair in enumerate(zip(cpu_tokens, npu_tokens))
            if pair[0] != pair[1]
        ]
        raise AssertionError(f"token mismatch at positions {differences}")

    cpu_median = statistics.median(cpu_times[1:])
    npu_median = statistics.median(npu_times[1:])
    speedup = cpu_median / npu_median
    print(f"Token agreement: {len(token_ids)}/{len(token_ids)}")
    print(f"CPU warm median: {cpu_median:.6f} s/token")
    print(f"NPU warm median: {npu_median:.6f} s/token")
    print(f"Median speedup: {speedup:.3f}x")
    if speedup <= 1.0:
        raise AssertionError("fused NPU path did not beat the CPU baseline")


if __name__ == "__main__":
    main()
