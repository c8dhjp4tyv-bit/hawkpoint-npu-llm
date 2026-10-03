#!/usr/bin/env python3
"""Offline benchmark harness for cross-placement logit agreement verification.

Provides hardware-free simulation, replaying, and evaluation of multi-backend
(CPU, GPU, NPU) inference agreement gates without requiring physical hardware
or llama-server binaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from placement_agreement import evaluate_agreement  # noqa: E402


def build_synthetic_placement_data(
    positions: int = 16,
    top_k: int = 5,
    margin_threshold: float = 1.0,
    diverge_on_confident: bool = False,
    diverge_on_ambiguous: bool = True,
    seed: int = 42,
) -> dict[str, list[list[dict]]]:
    """Synthesize multi-placement top-k logit records for offline evaluation.

    Generates realistic log-probability distributions with known margins.
    """
    if positions <= 0:
        raise ValueError("positions must be positive")
    if top_k < 2:
        raise ValueError("top_k must be at least 2")

    rng = random.Random(seed)
    vocab_base = 100

    cpu_rows = []
    gpu_rows = []
    xdna_rows = []

    for pos in range(positions):
        # Alternate between high-confidence (large margin) and ambiguous (small margin) positions
        is_confident = (pos % 2 == 0)
        if is_confident:
            top1_logprob = -0.1
            top2_logprob = top1_logprob - (margin_threshold + rng.uniform(0.5, 2.0))
        else:
            top1_logprob = -0.6
            top2_logprob = top1_logprob - rng.uniform(0.0, margin_threshold * 0.4)

        # Rest of candidates
        other_logprobs = [top2_logprob - rng.uniform(0.5, 2.0) * (i + 1) for i in range(top_k - 2)]

        top1_id = vocab_base + pos * 10
        top2_id = vocab_base + pos * 10 + 1
        other_ids = [vocab_base + pos * 10 + i for i in range(2, top_k)]

        ref_candidates = [
            {"id": top1_id, "logprob": round(top1_logprob, 4)},
            {"id": top2_id, "logprob": round(top2_logprob, 4)},
        ] + [{"id": tid, "logprob": round(lp, 4)} for tid, lp in zip(other_ids, other_logprobs, strict=True)]
        cpu_rows.append(ref_candidates)

        # GPU placement: generally matches CPU, may flip near-ties
        gpu_cands = [dict(c) for c in ref_candidates]
        if not is_confident and diverge_on_ambiguous:
            # Swap top-1 and top-2 for ambiguous position (tolerated) with consistent scores
            gpu_cands[0] = {"id": top2_id, "logprob": round(top1_logprob, 4)}
            gpu_cands[1] = {"id": top1_id, "logprob": round(top2_logprob, 4)}
            gpu_cands.sort(key=lambda c: c["logprob"], reverse=True)
        gpu_rows.append(gpu_cands)

        # XDNA placement: optionally diverges on confident tokens to test gate rejection
        xdna_cands = [dict(c) for c in ref_candidates]
        if is_confident and diverge_on_confident and pos == 0:
            # Force high-margin violation with consistent scores
            xdna_cands[0] = {"id": top2_id, "logprob": round(top1_logprob, 4)}
            xdna_cands[1] = {"id": top1_id, "logprob": round(top2_logprob, 4)}
            xdna_cands.sort(key=lambda c: c["logprob"], reverse=True)
        elif not is_confident and diverge_on_ambiguous:
            xdna_cands[0] = {"id": top2_id, "logprob": round(top1_logprob, 4)}
            xdna_cands[1] = {"id": top1_id, "logprob": round(top2_logprob, 4)}
            xdna_cands.sort(key=lambda c: c["logprob"], reverse=True)
        xdna_rows.append(xdna_cands)

    placements = {
        "cpu_only": cpu_rows,
        "gpu_only": gpu_rows,
        "xdna_only": xdna_rows,
    }
    return placements


def run_offline_harness(
    positions: int = 16,
    top_k: int = 5,
    margin_threshold: float = 1.0,
    scenario: str = "pass",
    seed: int = 42,
) -> tuple[dict, list[str]]:
    """Execute the offline agreement evaluation harness for a specific scenario."""
    if positions <= 0:
        raise ValueError("positions must be positive")
    if top_k < 2:
        raise ValueError("top_k must be at least 2")
    if scenario == "pass":
        # Tolerates ambiguous flips, no high-margin divergence
        data = build_synthetic_placement_data(
            positions=positions,
            top_k=top_k,
            margin_threshold=margin_threshold,
            diverge_on_confident=False,
            diverge_on_ambiguous=True,
            seed=seed,
        )
    elif scenario == "fail_confident":
        # Introduces a confident mismatch on xdna_only
        data = build_synthetic_placement_data(
            positions=positions,
            top_k=top_k,
            margin_threshold=margin_threshold,
            diverge_on_confident=True,
            diverge_on_ambiguous=True,
            seed=seed,
        )
    elif scenario == "exact":
        # Exact agreement across all placements
        data = build_synthetic_placement_data(
            positions=positions,
            top_k=top_k,
            margin_threshold=margin_threshold,
            diverge_on_confident=False,
            diverge_on_ambiguous=False,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown scenario: {scenario!r}")

    report, failures = evaluate_agreement(
        data,
        reference_name="cpu_only",
        margin_threshold=margin_threshold,
    )
    report["scenario"] = scenario
    report["margin_threshold"] = margin_threshold
    report["gate_failures"] = failures
    report["passed"] = len(failures) == 0
    return report, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run offline cross-placement logit agreement verification harness."
    )
    parser.add_argument(
        "--scenario",
        choices=["pass", "fail_confident", "exact", "all"],
        default="all",
        help="Scenario to evaluate (default: all)",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=16,
        help="Number of teacher-forced positions (default: 16)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k candidates per position (default: 5)",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=1.0,
        help="Threshold margin for failing gate (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write JSON report to specified path",
    )
    args = parser.parse_args(argv)

    if args.positions <= 0:
        parser.error("positions must be positive")
    if args.top_k < 2:
        parser.error("top-k must be at least 2")

    scenarios = ["pass", "fail_confident", "exact"] if args.scenario == "all" else [args.scenario]
    results = {}
    any_unexpected_failure = False

    for sc in scenarios:
        report, failures = run_offline_harness(
            positions=args.positions,
            top_k=args.top_k,
            margin_threshold=args.margin_threshold,
            scenario=sc,
        )
        results[sc] = report
        if sc in ("pass", "exact") and not report["passed"]:
            any_unexpected_failure = True
        elif sc == "fail_confident" and report["passed"]:
            any_unexpected_failure = True

        status = "PASSED" if report["passed"] else "FAILED (expected)" if sc == "fail_confident" else "FAILED"
        print(f"Scenario [{sc}]: {status} - failures: {failures}")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    return 1 if any_unexpected_failure else 0


if __name__ == "__main__":
    sys.exit(main())
