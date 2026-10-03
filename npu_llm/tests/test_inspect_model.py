#!/usr/bin/env python3
"""Tests for npu_llm/tools/inspect_model.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "npu_llm/tools"))

from npu_llm.tools.inspect_model import (  # noqa: E402
    compute_parameter_breakdown,
    format_markdown,
    format_text,
    inspect_model,
    main,
    verify_model_files,
)


def _create_mock_model_dir(dir_path: Path, corrupt_size: bool = False, corrupt_hash: bool = False) -> dict:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    weight_data = b"0123456789abcdef"
    weight_file = dir_path / "layer_0.bin"
    weight_file.write_bytes(weight_data)

    weight_hash = hashlib.sha256(weight_data).hexdigest()
    recorded_size = len(weight_data) + (10 if corrupt_size else 0)
    recorded_hash = "wronghash" if corrupt_hash else weight_hash

    manifest = {
        "format": "xdna1-w8a16",
        "model_id": "mock-smollm",
        "model_family": "llama",
        "layers": 30,
        "hidden_size": 576,
        "intermediate_size": 1536,
        "attention_heads": 9,
        "kv_heads": 3,
        "head_dim": 64,
        "vocab_size": 49152,
        "context_length": 64,
        "files": {
            "layer_0.bin": {
                "size": recorded_size,
                "sha256": recorded_hash,
            }
        },
    }
    (dir_path / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_parameter_breakdown():
    metadata = {
        "layers": 30,
        "hidden_size": 576,
        "intermediate_size": 1536,
        "attention_heads": 9,
        "kv_heads": 3,
        "head_dim": 64,
        "vocab_size": 49152,
        "context_length": 64,
    }
    res = compute_parameter_breakdown(metadata)
    assert res["total_parameters_millions"] > 100
    assert res["all_layers_parameters"] > 0
    assert res["estimated_weights_mib"] > 0


def test_verify_model_files_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        manifest = _create_mock_model_dir(model_dir)
        check = verify_model_files(model_dir, manifest, verify_checksums=True)
        assert check["valid"] is True
        assert check["verified_files_count"] == 1
        assert check["tokenizer_present"] is True


def test_verify_model_files_failure_modes():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        manifest = _create_mock_model_dir(model_dir, corrupt_size=True)
        check = verify_model_files(model_dir, manifest)
        assert check["valid"] is False
        assert len(check["size_mismatches"]) == 1

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        manifest = _create_mock_model_dir(model_dir, corrupt_hash=True)
        check = verify_model_files(model_dir, manifest, verify_checksums=True)
        assert check["valid"] is False
        assert len(check["checksum_failures"]) == 1


def test_inspect_model_and_formats():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        _create_mock_model_dir(model_dir)
        report = inspect_model(model_dir, verify_checksums=True)
        assert report["model_id"] == "mock-smollm"
        assert report["hardware_demands"]["xdna1_supported"] is True

        text_out = format_text(report)
        assert "Model Inspection: mock-smollm" in text_out
        assert "VALID [OK]" in text_out

        md_out = format_markdown(report)
        assert "## Model Specification: `mock-smollm`" in md_out
        assert "✅ Passed" in md_out


def test_main_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "model"
        out_file = Path(tmpdir) / "spec.md"
        _create_mock_model_dir(model_dir)

        code = main([str(model_dir), "--format", "markdown", "-o", str(out_file), "--verify-checksums"])
        assert code == 0
        assert out_file.exists()
        assert "## Model Specification" in out_file.read_text(encoding="utf-8")


if __name__ == "__main__":
    test_parameter_breakdown()
    test_verify_model_files_success()
    test_verify_model_files_failure_modes()
    test_inspect_model_and_formats()
    test_main_cli()
    print("PASS 5 inspect_model tests")
