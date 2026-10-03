#!/usr/bin/env python3
"""Tests for scripts/doctor.py."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.doctor import (  # noqa: E402
    CheckStatus,
    check_models_directory,
    check_network_and_services,
    check_npu_device_and_driver,
    check_os_and_cpu,
    check_python_dependencies,
    check_xrt_runtime,
    format_text_report,
    main,
    run_diagnostics,
)


def test_check_os_and_cpu():
    res = check_os_and_cpu()
    assert res["os"] != ""
    assert res["arch"] != ""
    assert "status" in res
    assert res["status"] in (CheckStatus.OK, CheckStatus.WARN, CheckStatus.FAIL)


def test_check_npu_device_and_driver():
    res = check_npu_device_and_driver()
    assert "driver_loaded" in res
    assert "accel_nodes" in res
    assert "user_accessible" in res
    assert res["status"] in (CheckStatus.OK, CheckStatus.WARN, CheckStatus.FAIL)


def test_check_xrt_runtime():
    res = check_xrt_runtime()
    assert "xrt_installed" in res
    assert res["status"] in (CheckStatus.OK, CheckStatus.WARN)


def test_check_python_dependencies():
    res = check_python_dependencies()
    assert "numpy" in res["installed"]
    assert "tokenizers" in res["installed"]
    assert res["status"] == CheckStatus.OK


def test_check_models_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty directory should trigger WARN
        empty_res = check_models_directory(Path(tmpdir))
        assert empty_res["status"] == CheckStatus.WARN
        assert empty_res["models_count"] == 0
        assert len(empty_res["remediation"]) > 0

        # With mock model
        model_dir = Path(tmpdir) / "SmolLM2-135M-Instruct-xdna1-w8a16"
        model_dir.mkdir()
        meta = {
            "model_id": "smollm2-135m-xdna1",
            "format": "xdna1-w8a16",
            "context_length": 64,
            "display_name": "Mock SmolLM2",
        }
        (model_dir / "metadata.json").write_text(json.dumps(meta))

        pop_res = check_models_directory(Path(tmpdir))
        assert pop_res["status"] == CheckStatus.OK
        assert pop_res["models_count"] == 1
        assert "smollm2-135m-xdna1" in pop_res["available_models"]


def test_check_network_and_services():
    res = check_network_and_services(port=8000)
    assert "port_8000_in_use" in res
    assert "docker_installed" in res


def test_run_diagnostics_and_format():
    diag = run_diagnostics()
    assert diag["overall_status"] in (CheckStatus.OK, CheckStatus.WARN, CheckStatus.FAIL)
    text = format_text_report(diag)
    assert "HawkPoint NPU Preflight Diagnostics (Doctor)" in text
    assert "Operating System" in text


def test_main_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main(["--models-dir", tmpdir, "--json"])
        # WARN or OK should return code <= 1
        assert code in (0, 1)


if __name__ == "__main__":
    test_check_os_and_cpu()
    test_check_npu_device_and_driver()
    test_check_xrt_runtime()
    test_check_python_dependencies()
    test_check_models_directory()
    test_check_network_and_services()
    test_run_diagnostics_and_format()
    test_main_cli()
    print("PASS 8 doctor tests")
