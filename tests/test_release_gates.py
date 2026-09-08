#!/usr/bin/env python3
"""Hosted tests for immutable release pins and placement agreement."""

import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_ollama_matrix import compare_placement_responses  # noqa: E402
from placement_agreement import PLACEMENTS, evaluate_agreement  # noqa: E402
from verify_hardware_versions import evaluate_gate, is_xdna1_name  # noqa: E402
from verify_ollama_manifest import verify_manifest  # noqa: E402


def _rows(*token_ids_with_logprobs):
    """Build one placement's per-position top-k rows.

    Each argument is a list of ``(token_id, logprob)`` for one position.
    """
    return [
        [{"id": tid, "logprob": lp} for tid, lp in position]
        for position in token_ids_with_logprobs
    ]


def test_logit_agreement_tolerance():
    # Reference: position 0 is confident (margin 2.0); position 1 is an ambiguous
    # near-tie (margin 0.1).
    reference = _rows(
        [(10, -0.1), (11, -2.1), (12, -3.0)],
        [(20, -0.5), (21, -0.6), (22, -3.0)],
    )
    # Agrees on the confident token; differs only on the ambiguous one.
    tolerant = _rows(
        [(10, -0.1), (11, -2.0), (12, -3.1)],
        [(21, -0.5), (20, -0.6), (22, -3.0)],
    )
    # Diverges on the CONFIDENT token -> must fail.
    divergent = _rows(
        [(11, -0.2), (10, -1.9), (12, -3.0)],
        [(20, -0.5), (21, -0.6), (22, -3.0)],
    )

    report, failures = evaluate_agreement(
        {"cpu_only": reference, "gpu_only": tolerant},
        "cpu_only",
        margin_threshold=1.0,
    )
    assert failures == []
    tol = report["placements"]["gpu_only"]
    assert tol["top1_matches"] == 1
    assert len(tol["tolerated_mismatches"]) == 1
    assert tol["high_margin_violations"] == []

    _, failures = evaluate_agreement(
        {"cpu_only": reference, "gpu_only": divergent},
        "cpu_only",
        margin_threshold=1.0,
    )
    assert failures == ["gpu_only_top1_divergence"]


def test_release_pins():
    pins = json.loads((ROOT / "release-pins.json").read_text())
    ollama = pins["ollama"]
    assert re.fullmatch(r"v\d+\.\d+\.\d+", ollama["source_tag"])
    assert re.fullmatch(r"[0-9a-f]{40}", ollama["source_commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", ollama["model_manifest_sha256"])
    build_script = (
        ROOT / "ollama-xdna" / "scripts" / "build-and-install.sh"
    ).read_text()
    patch_file = (
        ROOT
        / "ollama-xdna"
        / "patches"
        / f"ollama-{ollama['source_tag']}-xdna.patch"
    )
    assert patch_file.is_file()
    assert not (
        ROOT / "ollama-xdna" / "patches" / "ollama-v0.32.5-xdna.patch"
    ).exists()
    assert ollama["source_tag"] in patch_file.name
    assert ollama["source_tag"] in build_script
    assert ollama["source_commit"] in build_script
    assert (
        ROOT / "ollama-xdna" / "backend" / "compile_quantized.py"
    ).is_file()
    for kernel in ("project_q4k_bf16.cc", "project_q6k_bf16.cc"):
        assert (ROOT / "npu_llm" / "kernels" / kernel).is_file()


def test_hardware_gate_separates_compatibility_from_release_certification():
    pins = {
        "kernel_release": "7.2.0-certified",
        "amdxdna_version": "7.2.0-certified",
    }
    observed = {
        "kernel_release": "7.3.0-compatible",
        "amdxdna_version": "7.3.0-compatible",
    }
    capabilities = {"xrt_can_submit_kernel": {"ok": True}}

    differences, compatibility_failures = evaluate_gate(
        observed, pins, capabilities, strict_release=False
    )
    assert set(differences) == set(pins)
    assert compatibility_failures == []

    _, strict_failures = evaluate_gate(
        observed, pins, capabilities, strict_release=True
    )
    assert strict_failures == [
        "version:amdxdna_version",
        "version:kernel_release",
    ]
    assert is_xdna1_name("RyzenAI-npu1")
    assert not is_xdna1_name("RyzenAI-npu2")


def test_hardware_workflows_use_the_intended_validation_mode():
    release_workflow = (ROOT / ".github/workflows/release.yml").read_text()
    compatibility_workflow = (
        ROOT / ".github/workflows/npu-hardware.yml"
    ).read_text()
    assert "--strict-release" in release_workflow
    assert "--strict-release" not in compatibility_workflow


def test_manifest_digest():
    manifest = {
        "schemaVersion": 2,
        "config": {"digest": "sha256:" + "1" * 64},
        "layers": [{"digest": "sha256:" + "2" * 64}],
    }
    payload = json.dumps(manifest, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        path = (
            Path(directory)
            / "manifests"
            / "registry.ollama.ai"
            / "library"
            / "qwen2.5"
            / "0.5b"
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        report = verify_manifest(directory, "qwen2.5:0.5b", digest)
        assert report["verified"]
        try:
            verify_manifest(directory, "qwen2.5:0.5b", "0" * 64)
            raise AssertionError("manifest mismatch unexpectedly passed")
        except RuntimeError:
            pass


def test_cross_placement_agreement():
    matching = [
        {"placement": name, "response_sha256": "a" * 64}
        for name in ("cpu_only", "gpu_only", "cpu_gpu", "cpu_gpu_npu")
    ]
    hashes, agreed = compare_placement_responses(matching, 4)
    assert agreed and len(hashes) == 4
    matching[-1]["response_sha256"] = "b" * 64
    _, agreed = compare_placement_responses(matching, 4)
    assert not agreed


def test_placement_matrix_exercises_xdna_without_cuda():
    assert PLACEMENTS["xdna_only"] == {"ngl": 0, "xdna": True}


def main():
    test_release_pins()
    test_manifest_digest()
    test_cross_placement_agreement()
    test_placement_matrix_exercises_xdna_without_cuda()
    test_logit_agreement_tolerance()
    test_hardware_gate_separates_compatibility_from_release_certification()
    test_hardware_workflows_use_the_intended_validation_mode()
    print("PASS immutable release gates")


if __name__ == "__main__":
    main()
