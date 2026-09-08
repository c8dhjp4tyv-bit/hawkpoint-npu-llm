#!/usr/bin/env python3
"""Validate the XDNA host stack and write a hardware compatibility report.

The normal mode is deliberately capability based.  Kernel and driver version
strings are useful evidence, but they are not a runtime compatibility contract:
the same XRT/amdxdna UAPI can be provided by several distro and mainline
kernels.  ``--strict-release`` adds the exact values from ``release-pins.json``
for reproducible release certification.
"""

import argparse
import ctypes.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XCLBIN = (
    ROOT
    / "ollama-xdna"
    / "backend"
    / "artifacts"
    / "experts-8x2048x2048"
    / "experts.xclbin"
)
DEFAULT_INSTS = (
    ROOT
    / "ollama-xdna"
    / "backend"
    / "artifacts"
    / "experts-8x2048x2048"
    / "insts.bin"
)
PROBE_SOURCE = ROOT / "ollama-xdna" / "backend" / "test_experts.cpp"
DEFAULT_XRT_ROOT = Path("/opt/xilinx/xrt")


def command(*arguments, timeout=None):
    """Run a command and return combined text output."""

    return subprocess.check_output(
        arguments,
        text=True,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def field(text, label):
    """Read a ``Label : value`` field from ``xrt-smi examine`` output."""

    match = re.search(
        rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", text, re.MULTILINE
    )
    if not match:
        raise RuntimeError(f"could not read {label!r} from xrt-smi")
    return match.group(1)


def nvidia_versions():
    """Return optional NVIDIA observations and a diagnostic on failure.

    NVIDIA is relevant to the mixed-placement release evidence, but it is not
    required to prove that an XDNA1-only runtime is compatible.  Consequently
    a missing ``nvidia-smi`` is recorded rather than rejected in compatibility
    mode; strict release mode rejects it through the exact-value comparison.
    """

    values = {
        "gpu_name": None,
        "gpu_driver_version": None,
        "cuda_umd_version": None,
    }
    try:
        gpu_lines = command(
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ).splitlines()
        if not gpu_lines:
            raise RuntimeError("nvidia-smi returned no GPUs")
        gpu_name, driver = (
            value.strip() for value in gpu_lines[0].rsplit(",", 1)
        )
        overview = command("nvidia-smi")
        cuda_match = re.search(r"CUDA(?: UMD)? Version:\s*([0-9.]+)", overview)
        if not cuda_match:
            raise RuntimeError("could not read CUDA UMD version from nvidia-smi")
        values.update(
            gpu_name=gpu_name,
            gpu_driver_version=driver,
            cuda_umd_version=cuda_match.group(1),
        )
        return values, None
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        IndexError,
        ValueError,
    ) as exc:
        return values, str(exc)


def npu_name(text):
    """Read the first device name from the ``Device(s) Present`` table."""

    match = re.search(r"^\s*\|\[[^\]]+\]\s*\|([^|]+)\|", text, re.MULTILINE)
    if not match:
        raise RuntimeError("could not read NPU device name from xrt-smi")
    return match.group(1).strip()


def _xrt_root():
    configured = os.environ.get("XILINX_XRT")
    return Path(configured) if configured else DEFAULT_XRT_ROOT


def _xrt_library_path():
    root = _xrt_root()
    for directory in (root / "lib64", root / "lib"):
        if any(directory.glob("libxrt_coreutil.so*")):
            return directory
    return None


def xrt_runtime_present():
    """Check for the XRT runtime without requiring a particular version."""

    library_path = _xrt_library_path()
    if library_path is not None:
        return True, str(library_path)
    if ctypes.util.find_library("xrt_coreutil"):
        return True, "dynamic linker"
    return False, "libxrt_coreutil.so was not found"


def amdxdna_loaded():
    if Path("/sys/module/amdxdna").is_dir():
        return True
    try:
        return any(
            line.split() and line.split()[0] == "amdxdna"
            for line in command("lsmod").splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        return False


def accelerator_accessible():
    device = Path("/dev/accel/accel0")
    return device.is_char_device() and os.access(device, os.R_OK | os.W_OK)


def is_xdna1_name(name):
    normalized = (name or "").lower()
    return bool(re.search(r"(?:npu1|xdna1)", normalized))


def _probe_detail(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "probe completed without output"


def functional_probe(xclbin_path, instructions_path, timeout):
    """Compile and run the known-good XRT/AIE expert probe.

    ``test_experts.cpp`` intentionally exercises the complete minimum path:
    XRT device open, xclbin registration, hardware-context creation, BO
    allocation and sync, kernel submission/wait, and a numerical result check.
    Building it here keeps the validator independent of a stale prebuilt test
    binary while using the exact XRT headers and libraries on the runner.
    """

    compiler = shutil.which("g++")
    xrt_root = _xrt_root()
    include_dir = xrt_root / "include"
    library_dir = _xrt_library_path()
    missing = [
        str(path)
        for path in (PROBE_SOURCE, Path(xclbin_path), Path(instructions_path))
        if not path.is_file()
    ]
    if compiler is None:
        missing.append("g++")
    if not include_dir.is_dir():
        missing.append(str(include_dir))
    if library_dir is None:
        missing.append("libxrt_coreutil.so")
    if missing:
        return {
            "ok": False,
            "detail": "probe prerequisites missing: " + ", ".join(missing),
        }

    with tempfile.TemporaryDirectory(prefix="hawkpoint-xdna-probe-") as directory:
        binary = Path(directory) / "xdna-capability-probe"
        compile_command = [
            compiler,
            "-O2",
            "-DNDEBUG",
            "-std=c++17",
            str(PROBE_SOURCE),
            "-isystem",
            str(include_dir),
            f"-L{library_dir}",
            "-lxrt_coreutil",
            f"-Wl,-rpath,{library_dir}",
            "-o",
            str(binary),
        ]
        try:
            compiled = subprocess.run(
                compile_command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "detail": f"probe compilation failed: {exc}"}
        if compiled.returncode != 0:
            return {
                "ok": False,
                "detail": "probe compilation failed: " + _probe_detail(compiled.stdout),
                "output": compiled.stdout[-4000:],
            }

        try:
            executed = subprocess.run(
                [str(binary), str(xclbin_path), str(instructions_path)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "detail": f"probe execution failed: {exc}"}
        result = {
            "ok": executed.returncode == 0,
            "detail": _probe_detail(executed.stdout),
            "returncode": executed.returncode,
            "output": executed.stdout[-4000:],
        }
        if executed.returncode != 0:
            result["detail"] = "probe execution failed: " + result["detail"]
        return result


def collect_versions(xrt_smi):
    """Collect observations without turning a missing optional field into a crash."""

    errors = []
    try:
        xrt_output = command(xrt_smi, "examine")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        xrt_output = ""
        errors.append(f"xrt-smi examine failed: {exc}")

    def xrt_field(label):
        try:
            return field(xrt_output, label)
        except RuntimeError as exc:
            errors.append(str(exc))
            return None

    observed = {
        "amdxdna_version": xrt_field("amdxdna Version"),
        "cuda_umd_version": None,
        "firmware_version": xrt_field("NPU Firmware Version"),
        "gpu_driver_version": None,
        "gpu_name": None,
        "kernel_release": platform.release(),
        "npu_name": None,
        "xrt_hash": xrt_field("Hash"),
        "xrt_version": xrt_field("Version"),
    }
    if xrt_output:
        try:
            observed["npu_name"] = npu_name(xrt_output)
        except RuntimeError as exc:
            errors.append(str(exc))

    gpu, gpu_error = nvidia_versions()
    observed.update(gpu)
    if gpu_error:
        errors.append(f"optional NVIDIA observation unavailable: {gpu_error}")
    return observed, xrt_output, errors


def version_differences(observed, pins):
    return {
        name: {"expected": expected, "observed": observed.get(name)}
        for name, expected in pins.items()
        if observed.get(name) != expected
    }


def evaluate_gate(observed, pins, capabilities, strict_release):
    """Return non-version and (optionally) strict version failures.

    Keeping this decision separate from host probing makes the two policies
    explicit and lets hosted tests exercise them without an XDNA device.
    """

    differences = version_differences(observed, pins)
    failures = [
        f"capability:{name}"
        for name, result in capabilities.items()
        if not result.get("ok")
    ]
    if strict_release:
        failures.extend(f"version:{name}" for name in sorted(differences))
    return differences, failures


def capability_report(observed, xrt_output, probe):
    """Return capability checks that are independent of version strings."""

    runtime_ok, runtime_detail = xrt_runtime_present()
    xrt_visible = bool(xrt_output and observed.get("npu_name"))
    xdna1 = is_xdna1_name(observed.get("npu_name"))
    probe_ok = bool(probe.get("ok"))
    amdxdna_ok = amdxdna_loaded()
    accelerator_ok = accelerator_accessible()

    # The probe reaches all of these operations in order.  Keeping them as
    # named report entries makes the compatibility contract auditable instead
    # of hiding it behind one opaque "version matched" result.
    return {
        "amdxdna_present": {
            "ok": amdxdna_ok,
            "detail": "amdxdna module is loaded"
            if amdxdna_ok
            else "amdxdna module is not loaded",
        },
        "xdna1_hardware": {
            "ok": xdna1,
            "detail": observed.get("npu_name") or "XDNA device name unavailable",
        },
        "npu_device_node": {
            "ok": accelerator_ok,
            "detail": "/dev/accel/accel0 is readable and writable"
            if accelerator_ok
            else "/dev/accel/accel0 is missing or inaccessible",
        },
        "xrt_runtime": {"ok": runtime_ok, "detail": runtime_detail},
        "xrt_device_visible": {
            "ok": xrt_visible,
            "detail": "xrt-smi examine reports an NPU"
            if xrt_visible
            else "xrt-smi did not report an NPU",
        },
        "xrt_can_open_device": {
            "ok": probe_ok,
            "detail": "covered by the known-good XRT probe"
            if probe_ok
            else probe.get("detail", "XRT probe failed"),
        },
        "xrt_can_create_hwctx": {
            "ok": probe_ok,
            "detail": "covered by the known-good XRT probe"
            if probe_ok
            else probe.get("detail", "XRT probe failed"),
        },
        "xrt_can_allocate_bo": {
            "ok": probe_ok,
            "detail": "covered by the known-good XRT probe"
            if probe_ok
            else probe.get("detail", "XRT probe failed"),
        },
        "xrt_can_sync_bo": {
            "ok": probe_ok,
            "detail": "covered by the known-good XRT probe"
            if probe_ok
            else probe.get("detail", "XRT probe failed"),
        },
        "xrt_can_load_xclbin": {
            "ok": probe_ok,
            "detail": "covered by the known-good XRT probe"
            if probe_ok
            else probe.get("detail", "XRT probe failed"),
        },
        "xrt_can_submit_kernel": {
            "ok": probe_ok,
            "detail": probe.get("detail", "XRT probe failed"),
        },
        "xrt_required_ioctls": {
            "ok": probe_ok,
            "detail": "device, hwctx, BO, sync, xclbin, submit, and wait path passed"
            if probe_ok
            else probe.get("detail", "XRT probe failed"),
        },
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(
        description="Check XDNA runtime capabilities and optionally certify exact release pins"
    )
    parser.add_argument("--pins", type=Path, default=ROOT / "release-pins.json")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--strict-release",
        action="store_true",
        help="enforce every hardware value in release-pins.json for release certification",
    )
    parser.add_argument(
        "--probe-xclbin",
        type=Path,
        default=DEFAULT_XCLBIN,
        help="known-good XCLBIN used by the functional XRT probe",
    )
    parser.add_argument(
        "--probe-instructions",
        type=Path,
        default=DEFAULT_INSTS,
        help="instruction buffer used by the functional XRT probe",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=120.0,
        help="per-stage timeout in seconds for compiling/running the functional probe",
    )
    args = parser.parse_args()

    pins_document = json.loads(args.pins.read_text())
    pins = pins_document["hardware"]
    xrt_smi = shutil.which("xrt-smi") or "/opt/xilinx/xrt/bin/xrt-smi"
    observed, xrt_output, collection_errors = collect_versions(xrt_smi)
    probe = functional_probe(
        args.probe_xclbin, args.probe_instructions, args.probe_timeout
    )
    capabilities = capability_report(observed, xrt_output, probe)
    differences, failures = evaluate_gate(
        observed, pins, capabilities, args.strict_release
    )

    mode = "strict-release" if args.strict_release else "compatibility"
    report = {
        "schema_version": 2,
        "mode": mode,
        "captured_epoch": int(time.time()),
        "compatible": not failures,
        "release_certified": bool(args.strict_release and not failures),
        "policy": {
            "exact_version_check": args.strict_release,
            "version_differences_are_fatal": args.strict_release,
            "required_capabilities": list(capabilities),
        },
        "expected": pins,
        "observed": observed,
        # ``mismatches`` stays empty in compatibility mode for consumers that
        # treated the old exact-pin field as a failure list.  All differences
        # remain available as non-fatal diagnostic evidence below.
        "mismatches": differences if args.strict_release else {},
        "version_differences": differences,
        "capabilities": capabilities,
        "capability_failures": [
            name for name, result in capabilities.items() if not result.get("ok")
        ],
        "failures": failures,
        "collection_errors": collection_errors,
        "functional_probe": probe,
        "os_release": Path("/etc/os-release").read_text()
        if Path("/etc/os-release").is_file()
        else "",
        "xrt_examine": xrt_output,
    }
    write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        if args.strict_release:
            raise SystemExit(
                "strict release hardware gate failed: " + ", ".join(failures)
            )
        raise SystemExit(
            "hardware compatibility gate failed: " + ", ".join(failures)
        )
    if args.strict_release:
        print("PASS strict release hardware certification")
    else:
        print("PASS compatible XDNA1 hardware capability gate")


if __name__ == "__main__":
    main()
