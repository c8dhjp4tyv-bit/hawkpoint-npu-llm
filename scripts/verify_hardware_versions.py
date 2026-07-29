#!/usr/bin/env python3
"""Fail closed unless the release runner matches the pinned hardware stack."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def command(*arguments):
    return subprocess.check_output(arguments, text=True, stderr=subprocess.STDOUT)


def field(text, label):
    match = re.search(rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"could not read {label!r} from xrt-smi")
    return match.group(1)


def nvidia_versions():
    gpu_line = command(
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ).splitlines()[0]
    gpu_name, driver = (value.strip() for value in gpu_line.rsplit(",", 1))
    overview = command("nvidia-smi")
    cuda_match = re.search(
        r"CUDA(?: UMD)? Version:\s*([0-9.]+)",
        overview,
    )
    if not cuda_match:
        raise RuntimeError("could not read CUDA UMD version from nvidia-smi")
    return gpu_name, driver, cuda_match.group(1)


def npu_name(text):
    match = re.search(r"^\|\[[^\]]+\]\s*\|([^|]+)\|", text, re.MULTILINE)
    if not match:
        raise RuntimeError("could not read NPU device name from xrt-smi")
    return match.group(1).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pins", type=Path, default=ROOT / "release-pins.json")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    pins = json.loads(args.pins.read_text())["hardware"]
    xrt_smi = shutil.which("xrt-smi") or "/opt/xilinx/xrt/bin/xrt-smi"
    xrt_output = command(xrt_smi, "examine")
    gpu_name, gpu_driver, cuda_version = nvidia_versions()
    observed = {
        "amdxdna_version": field(xrt_output, "amdxdna Version"),
        "cuda_umd_version": cuda_version,
        "firmware_version": field(xrt_output, "NPU Firmware Version"),
        "gpu_driver_version": gpu_driver,
        "gpu_name": gpu_name,
        "kernel_release": platform.release(),
        "npu_name": npu_name(xrt_output),
        "xrt_hash": field(xrt_output, "Hash"),
        "xrt_version": field(xrt_output, "Version"),
    }
    mismatches = {
        name: {"expected": expected, "observed": observed.get(name)}
        for name, expected in pins.items()
        if observed.get(name) != expected
    }
    report = {
        "schema_version": 1,
        "captured_epoch": int(time.time()),
        "expected": pins,
        "observed": observed,
        "mismatches": mismatches,
        "os_release": Path("/etc/os-release").read_text(),
        "xrt_examine": xrt_output,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(f"hardware version gate failed: {sorted(mismatches)}")
    print("PASS pinned hardware version gate")


if __name__ == "__main__":
    main()
