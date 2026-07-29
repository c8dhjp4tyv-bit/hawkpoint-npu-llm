#!/usr/bin/env python3
"""Compare the fused Qwen XDNA1 path with the NumPy BF16 reference."""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.generate import NPUDecoder


NEAR_TIE_LOGIT_MARGIN = 0.05
BENCHMARK_PROMPT = [
    {
        "role": "user",
        "content": (
            "Explain why the sky appears blue during the day in one detailed "
            "paragraph with at least five sentences."
        ),
    }
]


def run_prefix(model_dir, npu_layers, token_ids):
    decoder = NPUDecoder(model_dir, npu_layers=npu_layers)
    if npu_layers:
        decoder.warmup()
    output = []
    timings = []
    diagnostics = []
    for position, token_id in enumerate(token_ids):
        next_token, elapsed, details = decoder.decode_token(
            token_id,
            position,
            diagnostics=True,
        )
        output.append(next_token)
        timings.append(elapsed)
        diagnostics.append(details)
    return output, timings, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--positions",
        type=int,
        default=12,
        help="number of ChatML prompt positions to compare",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="write machine-readable token, timing, and logit evidence",
    )
    args = parser.parse_args()

    tokenizer_decoder = NPUDecoder(args.model_dir, npu_layers=0)
    token_ids = tokenizer_decoder.tokenizer.encode_chat(BENCHMARK_PROMPT)[
        : args.positions
    ]
    if len(token_ids) != args.positions:
        raise AssertionError(
            f"benchmark prompt has only {len(token_ids)} token positions"
        )
    del tokenizer_decoder
    gc.collect()

    cpu_tokens, cpu_times, cpu_diagnostics = run_prefix(
        args.model_dir,
        0,
        token_ids,
    )
    gc.collect()
    npu_tokens, npu_times, npu_diagnostics = run_prefix(
        args.model_dir,
        24,
        token_ids,
    )

    cpu_median = statistics.median(cpu_times[1:])
    npu_median = statistics.median(npu_times[1:])
    speedup = cpu_median / npu_median
    differences = [
        index
        for index, pair in enumerate(zip(cpu_tokens, npu_tokens))
        if pair[0] != pair[1]
    ]
    accepted_near_ties = [
        index
        for index in differences
        if (
            cpu_diagnostics[index]["top_logit_margin"]
            <= NEAR_TIE_LOGIT_MARGIN
            and npu_diagnostics[index]["top_logit_margin"]
            <= NEAR_TIE_LOGIT_MARGIN
            and set(cpu_diagnostics[index]["top_token_ids"][:2])
            == set(npu_diagnostics[index]["top_token_ids"][:2])
        )
    ]
    unaccepted_differences = [
        index for index in differences if index not in accepted_near_ties
    ]
    report = {
        "schema_version": 2,
        "prompt": BENCHMARK_PROMPT,
        "input_token_ids": token_ids,
        "positions": len(token_ids),
        "cpu_token_ids": cpu_tokens,
        "npu_token_ids": npu_tokens,
        "difference_positions": differences,
        "accepted_near_tie_positions": accepted_near_ties,
        "unaccepted_difference_positions": unaccepted_differences,
        "near_tie_logit_margin": NEAR_TIE_LOGIT_MARGIN,
        "cpu_diagnostics": cpu_diagnostics,
        "npu_diagnostics": npu_diagnostics,
        "cpu_median_seconds": cpu_median,
        "npu_median_seconds": npu_median,
        "median_speedup": speedup,
        "npu_faster_than_cpu": speedup > 1.0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if unaccepted_differences:
        raise AssertionError(
            "non-tie token mismatch at positions "
            f"{unaccepted_differences}"
        )

    print(
        "Exact prefill argmax agreement: "
        f"{len(token_ids) - len(differences)}/{len(token_ids)}"
    )
    print(f"Accepted BF16 near-ties: {accepted_near_ties}")
    print(f"CPU warm median: {cpu_median:.6f} s/token")
    print(f"NPU warm median: {npu_median:.6f} s/token")
    print(f"Median speedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()
