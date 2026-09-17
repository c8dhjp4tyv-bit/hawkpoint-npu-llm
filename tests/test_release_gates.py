#!/usr/bin/env python3
"""Hosted tests for immutable release pins and placement agreement."""

import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import subprocess
import os


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
    """Reject confident logit divergence while tolerating ambiguous near-ties."""
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
    """Validate immutable release references and required patch/kernel artifacts."""
    pins = json.loads((ROOT / "release-pins.json").read_text())
    ollama = pins["ollama"]
    colibri = pins["colibri"]
    assert colibri["source_repository"] == "https://github.com/JustVugg/colibri.git"
    assert re.fullmatch(r"[0-9a-f]{40}", colibri["source_commit"])
    colibri_patch = ROOT / "colibri-xdna" / "patches" / "colibri-a8f2ca6-xdna.patch"
    assert colibri_patch.is_file()
    assert colibri["source_commit"][:7] in colibri_patch.name
    assert (ROOT / "colibri-xdna" / "backend" / "backend_xdna.cpp").is_file()
    assert (ROOT / "colibri-xdna" / "backend" / "test_backend_xdna.cpp").is_file()
    assert (ROOT / "colibri-xdna" / "scripts" / "build.sh").is_file()
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


def test_bf16_decoder_uses_32_row_projection_blocks():
    """Keep BF16 object-FIFO dimensions aligned with the selected AIE kernel."""
    design = (ROOT / "npu_llm" / "designs" / "decoder_layer.py").read_text()
    assert '"layer_project32_k576_bf16"' in design
    assert '"layer_project64_k576_pair_bf16"' not in design


def test_hardware_gate_separates_compatibility_from_release_certification():
    """Distinguish version drift from functional hardware capability failures."""
    pins = {
        "kernel_release": "7.2.0-certified",
        "amdxdna_version": "7.2.0-certified",
    }
    observed = {
        "kernel_release": "7.3.0-compatible",
        "amdxdna_version": "7.3.0-compatible",
    }
    capabilities = {
        "xdna1_hardware": {"ok": False, "required": False},
        "xrt_can_submit_kernel": {"ok": True, "required": True},
    }

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
    capabilities["xrt_can_submit_kernel"]["ok"] = False
    _, probe_failures = evaluate_gate(
        observed, pins, capabilities, strict_release=False
    )
    assert probe_failures == ["capability:xrt_can_submit_kernel"]
    assert is_xdna1_name("RyzenAI-npu1")
    assert not is_xdna1_name("RyzenAI-npu2")


def test_hardware_workflows_use_the_intended_validation_mode():
    """Require strict certification only in the release workflow."""
    release_workflow = (ROOT / ".github/workflows/release.yml").read_text()
    compatibility_workflow = (
        ROOT / ".github/workflows/npu-hardware.yml"
    ).read_text()
    assert "--strict-release" in release_workflow
    assert "--strict-release" not in compatibility_workflow
    colibri_command = "./colibri-xdna/scripts/build.sh"
    assert colibri_command in release_workflow
    assert colibri_command in compatibility_workflow
    assert "hosted-colibri" in release_workflow
    runner_setup = (
        ROOT / "scripts" / "configure-hardware-runner.sh"
    ).read_text()
    assert 'minimum_actions_runner_version="2.327.1"' in runner_setup
    assert "Runner.Listener" in runner_setup


def _run_hardware_runner_version_gate(version):
    """Run the hardware setup with a controlled Runner.Listener version."""
    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        listener = temporary_root / "Runner.Listener"
        listener.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' '{version}'\n"
        )
        listener.chmod(0o755)
        environment = dict(
            os.environ,
            GITHUB_ENV=str(temporary_root / "github-env"),
            GITHUB_PATH=str(temporary_root / "github-path"),
            RUNNER_TEMP=str(temporary_root / "_work" / "_temp"),
            HAWKPOINT_RUNNER_LISTENER=str(listener),
            HAWKPOINT_MLIR_AIE_DIR=str(temporary_root / "missing-mlir-aie"),
        )
        return subprocess.run(
            [str(ROOT / "scripts" / "configure-hardware-runner.sh")],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )


def test_hardware_runner_enforces_node24_minimum():
    """Reject pre-Node-24 runners and accept the documented boundary."""
    outdated = _run_hardware_runner_version_gate("2.326.0")
    assert outdated.returncode != 0
    assert "is too old; version 2.327.1 or newer is required" in outdated.stderr

    minimum = _run_hardware_runner_version_gate("2.327.1")
    assert minimum.returncode != 0
    assert "is too old" not in minimum.stderr
    assert "MLIR-AIE checkout not found" in minimum.stderr


def test_python_dependency_files_are_in_sync():
    """Prevent dependency updates from bypassing hashed workflow installs."""
    requirements_in = (ROOT / "requirements.in").read_text()
    requirements_txt = (ROOT / "requirements.txt").read_text()
    requirements_lock = (ROOT / "requirements.lock").read_text()

    assert requirements_txt == requirements_in
    direct_pins = re.findall(
        r"^([A-Za-z0-9_.-]+)==([^\s#]+)$", requirements_in, re.MULTILINE
    )
    assert direct_pins
    for package, version in direct_pins:
        locked_pin = rf"^{re.escape(package)}=={re.escape(version)}(?:\s|$)"
        assert re.search(locked_pin, requirements_lock, re.MULTILINE), (
            f"requirements.lock is missing {package}=={version}; regenerate it"
        )

    install_commands = []
    workflow_directory = ROOT / ".github/workflows"
    workflows = sorted(
        path
        for path in workflow_directory.iterdir()
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    )
    for workflow in workflows:
        for line_number, line in enumerate(workflow.read_text().splitlines(), 1):
            if "pip install" in line:
                install_commands.append((workflow, line_number, line.strip()))

    assert install_commands
    required_arguments = "--require-hashes -r requirements.lock"
    for workflow, line_number, command in install_commands:
        assert required_arguments in command, (
            f"{workflow.relative_to(ROOT)}:{line_number} must install the "
            "hashed requirements.lock"
        )


def test_native_quick_budget_covers_all_models():
    """Fail if the real catalog outgrows the four-model quick-test budget."""
    # Load script defaults in a fresh process without an NPU or API server.
    environment = dict(os.environ, HAWKPOINT_API_KEY="offline-test")
    environment.pop("HAWKPOINT_SOAK_COMPLETIONS", None)
    environment.pop("HAWKPOINT_SOAK_RUN_LENGTH", None)
    code = """
from collections import Counter
import soak_api
import sys
sys.path.insert(0, '..')
from npu_llm.model_catalog import MODEL_PRESETS
models = list(MODEL_PRESETS)
assert len(models) == 4, 'Update the quick budget for the supported model set'
counts = Counter(soak_api.model_for(i, models)
                 for i in range(soak_api.COMPLETIONS))
assert counts == dict.fromkeys(models, 25), counts
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT / "tests",
        env=environment, check=True,
    )


def test_manifest_digest():
    """Verify exact manifest bytes and reject a mismatched digest."""
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
    """Compare placement response hashes without assuming numerical equivalence."""
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
    """Ensure the XDNA-only logit placement needs no CUDA offload."""
    assert PLACEMENTS["xdna_only"] == {"ngl": 0, "xdna": True}


def main():
    """Run all offline regression checks in this module."""
    test_native_quick_budget_covers_all_models()
    test_release_pins()
    test_manifest_digest()
    test_cross_placement_agreement()
    test_placement_matrix_exercises_xdna_without_cuda()
    test_logit_agreement_tolerance()
    test_hardware_gate_separates_compatibility_from_release_certification()
    test_hardware_workflows_use_the_intended_validation_mode()
    test_hardware_runner_enforces_node24_minimum()
    test_python_dependency_files_are_in_sync()
    print("PASS immutable release gates")


if __name__ == "__main__":
    main()
