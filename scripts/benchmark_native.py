#!/usr/bin/env python3
"""Measure warm native decode throughput and optional phase timings.

This benchmark deliberately separates the one-time compile/warmup from the
steady-state generation loop.  It does not claim the target is met unless the
same command is run on a configured XDNA1 machine.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "npu_llm"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--npu-layers", type=int, default=None)
    parser.add_argument(
        "--prompt",
        default="Explain why the sky appears blue during the day.",
    )
    parser.add_argument("--min-tps", type=float, default=50.0)
    parser.add_argument("--stretch-tps", type=float, default=75.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help=(
            "let repeated runs reuse the cached prompt prefix; by default every "
            "run prefills the whole prompt so TTFT stays comparable"
        ),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="exit with status 2 when the median is below --min-tps",
    )
    args = parser.parse_args()
    if args.tokens < 1 or args.tokens >= 64:
        parser.error("--tokens must be between 1 and 63")
    if args.runs < 1 or args.warmup_runs < 0:
        parser.error("--runs must be positive and --warmup-runs cannot be negative")
    if args.profile:
        os.environ["HAWKPOINT_PROFILE"] = "1"

    try:
        from runtime.generate import NPUDecoder
    except ModuleNotFoundError as exc:
        if exc.name == "aie":
            raise SystemExit(
                "MLIR-AIE/IRON is not installed; run this benchmark in the "
                "validated ironenv on an XDNA1 host."
            ) from exc
        raise

    decoder = NPUDecoder(args.model_dir, npu_layers=args.npu_layers)
    try:
        started = time.perf_counter()
        decoder.warmup()
        compile_seconds = time.perf_counter() - started
        messages = [{"role": "user", "content": args.prompt}]
        for _ in range(args.warmup_runs):
            list(decoder.generate_messages(messages, args.tokens))

        results = []
        for run in range(args.runs):
            if not args.prefix_cache:
                decoder.reset_prefix_cache()
            started = time.perf_counter()
            final_stats = None
            for _, stats in decoder.generate_messages(messages, args.tokens):
                if stats is not None:
                    final_stats = stats
            if final_stats is None:
                raise RuntimeError("generation did not return final statistics")
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "run": run,
                    "decode_tokens_per_second": final_stats[
                        "decode_tokens_per_second"
                    ],
                    "wall_seconds": elapsed,
                    "stats": final_stats,
                }
            )
        rates = [item["decode_tokens_per_second"] for item in results]
        median = statistics.median(rates)
        report = {
            "schema_version": 1,
            "model_dir": str(args.model_dir),
            "configuration": {
                "decoder_w8": (
                    os.environ.get("HAWKPOINT_DECODER_W8") == "1"
                ),
                "cpu_final_norm": (
                    os.environ.get("HAWKPOINT_CPU_FINAL_NORM") == "1"
                ),
                "smollm_chunk": os.environ.get("HAWKPOINT_SMOLLM_CHUNK", "2"),
                "prefix_cache": args.prefix_cache,
                "decoder_engine": decoder._engine is not None,
            },
            "tokens": args.tokens,
            "runs": results,
            "compile_seconds": compile_seconds,
            "median_decode_tokens_per_second": median,
            "min_target_tokens_per_second": args.min_tps,
            "stretch_target_tokens_per_second": args.stretch_tps,
            "target_met": median >= args.min_tps,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_suffix(args.report.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, args.report)
        if args.require_target and median < args.min_tps:
            return 2
        return 0
    finally:
        decoder.close()


if __name__ == "__main__":
    raise SystemExit(main())
