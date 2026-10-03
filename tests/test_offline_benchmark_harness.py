#!/usr/bin/env python3
"""Tests for the offline benchmark harness and cross-placement logit evaluation."""

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from offline_benchmark_harness import (  # noqa: E402
    build_synthetic_placement_data,
    generate_markdown_report,
    main,
    run_offline_harness,
)


def test_build_synthetic_placement_data_validation():
    """Verify input validation for synthetic data generation."""
    try:
        build_synthetic_placement_data(positions=0)
    except ValueError as exc:
        assert "positions must be positive" in str(exc)
    else:
        raise AssertionError("did not raise for positions=0")

    try:
        build_synthetic_placement_data(top_k=1)
    except ValueError as exc:
        assert "top_k must be at least 2" in str(exc)
    else:
        raise AssertionError("did not raise for top_k=1")


def test_offline_harness_scenarios():
    """Verify pass, fail_confident, and exact scenario evaluation semantics."""
    # Pass scenario
    report_pass, failures_pass = run_offline_harness(positions=8, top_k=4, scenario="pass")
    assert report_pass["passed"]
    assert len(failures_pass) == 0
    assert report_pass["placements"]["cpu_only"]["top1_matches"] == 8

    # Exact scenario
    report_exact, failures_exact = run_offline_harness(positions=8, top_k=4, scenario="exact")
    assert report_exact["passed"]
    assert len(failures_exact) == 0
    for p in ("cpu_only", "gpu_only", "xdna_only"):
        assert report_exact["placements"][p]["top1_matches"] == 8
        assert len(report_exact["placements"][p]["tolerated_mismatches"]) == 0

    # Fail confident scenario
    report_fail, failures_fail = run_offline_harness(positions=8, top_k=4, scenario="fail_confident")
    assert not report_fail["passed"]
    assert len(failures_fail) > 0
    assert any("xdna_only" in f for f in failures_fail)

    # Unknown scenario
    try:
        run_offline_harness(scenario="invalid_scenario")
    except ValueError as exc:
        assert "unknown scenario" in str(exc)
    else:
        raise AssertionError("did not raise for unknown scenario")


def test_generate_markdown_report():
    """Verify Markdown report formatting and table structure."""
    report_pass, _ = run_offline_harness(positions=4, top_k=3, scenario="pass")
    md = generate_markdown_report({"pass": report_pass})
    assert "# Cross-Placement Logit Agreement Evaluation Report" in md
    assert "| Scenario | Placement | Top-1 Agreement |" in md
    assert "`pass`" in md
    assert "`cpu_only`" in md
    assert "**PASS**" in md


def test_cli_execution():
    """Verify CLI flags, output formats, and error handling."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "report.json"
        md_path = Path(tmp_dir) / "report.md"

        # JSON format output
        code = main(["--scenario", "pass", "--format", "json", "--output", str(json_path)])
        assert code == 0
        assert json_path.is_file()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "pass" in data
        assert data["pass"]["passed"]

        # Markdown format output
        code = main(["--scenario", "pass", "--format", "markdown", "--output", str(md_path)])
        assert code == 0
        assert md_path.is_file()
        content = md_path.read_text(encoding="utf-8")
        assert "| `pass` |" in content

        # CLI validation error
        try:
            main(["--positions", "0"])
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError("did not exit for invalid positions")


def run_all_tests():
    test_build_synthetic_placement_data_validation()
    test_offline_harness_scenarios()
    test_generate_markdown_report()
    test_cli_execution()
    print("PASS 4 offline benchmark harness tests")


if __name__ == "__main__":
    run_all_tests()
