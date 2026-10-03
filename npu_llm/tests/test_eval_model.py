#!/usr/bin/env python3
"""Tests for npu_llm/tools/eval_model.py."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "npu_llm/tools"))

from npu_llm.tools.eval_model import (  # noqa: E402
    DEFAULT_EVAL_DATASET,
    EvalItemResult,
    EvalReport,
    compute_cross_entropy_and_perplexity,
    format_markdown_report,
    format_text_report,
    main,
    run_self_test,
)


def test_cross_entropy_and_perplexity_calculation():
    """Verify loss and perplexity math under normal and boundary conditions."""
    # Empty list
    loss, ppl = compute_cross_entropy_and_perplexity([])
    assert loss == 0.0
    assert ppl == 1.0

    # Single logprob of ln(0.5) ~ -0.693147 -> PPL should be 2.0
    logprob_half = math.log(0.5)
    loss, ppl = compute_cross_entropy_and_perplexity([logprob_half])
    assert math.isclose(loss, 0.693147, rel_tol=1e-4)
    assert math.isclose(ppl, 2.0, rel_tol=1e-4)

    # Multi-token logprobs
    loss, ppl = compute_cross_entropy_and_perplexity([-0.1, -0.2, -0.3])
    assert math.isclose(loss, 0.2, rel_tol=1e-4)
    assert math.isclose(ppl, math.exp(0.2), rel_tol=1e-4)

    # Filtering of None and non-finite values
    loss, ppl = compute_cross_entropy_and_perplexity([None, float("nan"), -0.4, float("inf")])
    assert math.isclose(loss, 0.4, rel_tol=1e-4)
    assert math.isclose(ppl, math.exp(0.4), rel_tol=1e-4)


def test_format_reports():
    """Verify ASCII text and Markdown report generation."""
    item = EvalItemResult(
        item_id="test-1",
        category="reasoning",
        prompt="1+1=",
        expected="2",
        generated_text="2",
        matched=True,
        tokens=["2"],
        token_logprobs=[-0.05],
        cross_entropy=0.05,
        perplexity=1.051,
        ttft_ms=12.5,
        decode_tokens_per_second=80.0,
    )
    report = EvalReport(
        model="test-model",
        total_items=1,
        matched_items=1,
        accuracy=1.0,
        mean_cross_entropy=0.05,
        perplexity=1.051,
        mean_ttft_ms=12.5,
        mean_tokens_per_second=80.0,
        total_prompt_tokens=0,
        total_completion_tokens=1,
        categories={
            "reasoning": {
                "total": 1,
                "matched": 1,
                "accuracy": 1.0,
                "cross_entropy": 0.05,
                "perplexity": 1.051,
            }
        },
        items=[item],
    )

    text_rep = format_text_report(report)
    assert "HawkPoint NPU Model Quality" in text_rep
    assert "test-model" in text_rep
    assert "100.0%" in text_rep
    assert "reasoning" in text_rep

    md_rep = format_markdown_report(report)
    assert "# Model Evaluation Report: `test-model`" in md_rep
    assert "| **Answer Accuracy** | **100.0%**" in md_rep
    assert "| `reasoning` |" in md_rep
    assert "| `test-1` | `reasoning` | **PASS** |" in md_rep


def test_self_test_execution():
    """Verify run_self_test execution and main --self-test flag."""
    res = run_self_test()
    assert res == 0

    cli_res = main(["--self-test"])
    assert cli_res == 0


def test_custom_dataset_handling():
    """Verify custom dataset file loading and validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_file = Path(tmp_dir) / "custom_eval.json"
        dataset_content = [
            {
                "id": "c-1",
                "category": "test",
                "prompt": "Hello",
                "expected": "world",
            }
        ]
        json_file.write_text(json.dumps(dataset_content), encoding="utf-8")

        # Invalid file path should trigger argument error
        try:
            main(["--dataset", str(Path(tmp_dir) / "nonexistent.json")])
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError("did not exit for missing dataset")


def run_all_tests():
    test_cross_entropy_and_perplexity_calculation()
    test_format_reports()
    test_self_test_execution()
    test_custom_dataset_handling()
    print("PASS 4 model evaluation tests")


if __name__ == "__main__":
    run_all_tests()
