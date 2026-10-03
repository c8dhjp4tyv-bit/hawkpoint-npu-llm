#!/usr/bin/env python3
"""System diagnostics and preflight healthcheck for HawkPoint NPU deployment."""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu_llm.model_catalog import discover_models  # noqa: E402


class CheckStatus:
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


def check_os_and_cpu() -> dict:
    """Verify host operating system, architecture, and CPU vector instructions."""
    os_name = platform.system()
    arch = platform.machine()
    kernel_release = platform.release()

    is_linux = os_name.lower() == "linux"
    is_x86_64 = arch in ("x86_64", "AMD64")

    has_avx2 = False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    if "avx2" in line:
                        has_avx2 = True
                    break
    except Exception:
        pass

    status = CheckStatus.OK
    issues = []
    if not is_linux:
        status = CheckStatus.FAIL
        issues.append(f"HawkPoint NPU requires Linux; detected {os_name}")
    if not is_x86_64:
        status = CheckStatus.FAIL
        issues.append(f"Host architecture must be x86_64; detected {arch}")
    if not has_avx2:
        if status == CheckStatus.OK:
            status = CheckStatus.WARN
        issues.append("CPU missing AVX2 flags; CPU fallback operations will run slowly")

    return {
        "status": status,
        "os": os_name,
        "arch": arch,
        "kernel": kernel_release,
        "avx2": has_avx2,
        "issues": issues,
    }


def check_npu_device_and_driver() -> dict:
    """Verify /dev/accel/accel* presence, permissions, and amdxdna driver."""
    accel_nodes = list(Path("/dev/accel").glob("accel*")) if Path("/dev/accel").exists() else []
    driver_sysfs = Path("/sys/module/amdxdna")
    driver_loaded = driver_sysfs.exists()
    driver_version = None

    if driver_loaded:
        ver_file = driver_sysfs / "version"
        if ver_file.is_file():
            try:
                driver_version = ver_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

    user_accessible = False
    node_paths = []
    for node in accel_nodes:
        node_paths.append(str(node))
        if os.access(node, os.R_OK | os.W_OK):
            user_accessible = True

    status = CheckStatus.OK
    issues = []
    remediation = []

    if not accel_nodes:
        status = CheckStatus.WARN
        issues.append("No /dev/accel/accel* device nodes found")
        remediation.append("Ensure AMD Ryzen AI NPU is enabled in BIOS and amdxdna driver is installed")
    elif not user_accessible:
        status = CheckStatus.FAIL
        issues.append(f"Current user '{os.environ.get('USER')}' lacks read/write permissions to {node_paths}")
        remediation.append("Add user to render/video group: 'sudo usermod -aG render $USER' and re-login")

    if not driver_loaded:
        if status != CheckStatus.FAIL:
            status = CheckStatus.WARN
        issues.append("Kernel module 'amdxdna' is not currently loaded")
        remediation.append("Load module with 'sudo modprobe amdxdna' or install AMD XDNA driver")

    return {
        "status": status,
        "driver_loaded": driver_loaded,
        "driver_version": driver_version,
        "accel_nodes": node_paths,
        "user_accessible": user_accessible,
        "issues": issues,
        "remediation": remediation,
    }


def check_xrt_runtime() -> dict:
    """Verify Xilinx/AMD XRT runtime library and utility presence."""
    libxrt = ctypes.util.find_library("xrt_core")
    xrt_default_dir = Path("/opt/xilinx/xrt")
    has_xrt_dir = xrt_default_dir.is_dir()
    xrt_smi = shutil.which("xrt-smi") or (
        str(xrt_default_dir / "bin/xrt-smi")
        if (xrt_default_dir / "bin/xrt-smi").is_file()
        else None
    )

    status = CheckStatus.OK
    issues = []
    remediation = []

    if not libxrt and not has_xrt_dir:
        status = CheckStatus.WARN
        issues.append("XRT runtime (libxrt_core / /opt/xilinx/xrt) not found")
        remediation.append("Install xrt packages (xrt, xrt-dkms) for your distribution")

    return {
        "status": status,
        "xrt_installed": bool(libxrt or has_xrt_dir),
        "xrt_path": str(xrt_default_dir) if has_xrt_dir else libxrt,
        "xrt_smi": xrt_smi,
        "issues": issues,
        "remediation": remediation,
    }


def check_python_dependencies() -> dict:
    """Verify required Python packages are installed in the environment."""
    required_packages = ["numpy", "tokenizers", "safetensors", "huggingface_hub", "ml_dtypes"]
    installed = {}
    missing = []

    for pkg in required_packages:
        try:
            mod = __import__(pkg)
            installed[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            missing.append(pkg)

    status = CheckStatus.OK if not missing else CheckStatus.FAIL
    issues = [f"Missing required Python package: {pkg}" for pkg in missing]
    remediation = [f"pip install {' '.join(missing)}"] if missing else []

    return {
        "status": status,
        "installed": installed,
        "missing": missing,
        "issues": issues,
        "remediation": remediation,
    }


def check_models_directory(models_dir: Path) -> dict:
    """Verify converted model catalog in storage path."""
    models_dir = Path(models_dir)
    try:
        discovered = discover_models(models_dir) if models_dir.is_dir() else {}
    except RuntimeError as exc:
        return {
            "status": CheckStatus.FAIL,
            "models_dir": str(models_dir),
            "models_count": 0,
            "available_models": [],
            "issues": [str(exc)],
            "remediation": ["Remove duplicate model IDs or correct metadata.json in the model directories"],
        }

    status = CheckStatus.OK if discovered else CheckStatus.WARN
    issues = []
    remediation = []

    if not discovered:
        issues.append(f"No converted XDNA1 models found in {models_dir}")
        remediation.append("Run 'python scripts/prepare_model.py' to download and convert SmolLM2 135M")

    return {
        "status": status,
        "models_dir": str(models_dir.resolve()) if models_dir.exists() else str(models_dir),
        "models_count": len(discovered),
        "available_models": list(discovered.keys()),
        "issues": issues,
        "remediation": remediation,
    }


def check_network_and_services(port: int = 8000) -> dict:
    """Verify API port availability and existing server status."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    in_use = False
    try:
        sock.connect(("127.0.0.1", port))
        in_use = True
    except OSError:
        in_use = False
    finally:
        sock.close()

    api_server_running = False
    api_ready = False
    if in_use:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    api_server_running = True
        except Exception:
            pass

        if api_server_running:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/ready", method="GET")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    api_ready = resp.status == 200
            except Exception:
                pass

    docker_installed = shutil.which("docker") is not None
    conflict = in_use and not api_server_running

    return {
        "status": CheckStatus.FAIL if conflict else CheckStatus.OK,
        "issues": [f"Port {port} is occupied by an unrecognized service"] if conflict else [],
        "remediation": [f"Stop the service using port {port} or configure a different API port"] if conflict else [],
        "port_8000_in_use": in_use,
        "api_server_active": api_server_running,
        "api_server_ready": api_ready,
        "docker_installed": docker_installed,
    }


def run_diagnostics(models_dir: Path | None = None) -> dict:
    """Execute all preflight diagnostic checks and aggregate results."""
    m_dir = models_dir or (ROOT / "npu_llm/models")

    os_res = check_os_and_cpu()
    npu_res = check_npu_device_and_driver()
    xrt_res = check_xrt_runtime()
    py_res = check_python_dependencies()
    mod_res = check_models_directory(m_dir)
    net_res = check_network_and_services()

    all_statuses = [
        os_res["status"],
        npu_res["status"],
        xrt_res["status"],
        py_res["status"],
        mod_res["status"],
        net_res["status"],
    ]

    if CheckStatus.FAIL in all_statuses:
        overall = CheckStatus.FAIL
    elif CheckStatus.WARN in all_statuses:
        overall = CheckStatus.WARN
    else:
        overall = CheckStatus.OK

    return {
        "overall_status": overall,
        "checks": {
            "os_and_cpu": os_res,
            "npu_hardware": npu_res,
            "xrt_runtime": xrt_res,
            "python_environment": py_res,
            "model_catalog": mod_res,
            "network_services": net_res,
        },
    }


def _badge(status: str) -> str:
    if status == CheckStatus.OK:
        return "[✓]"
    if status == CheckStatus.WARN:
        return "[!]"
    return "[✗]"


def format_text_report(diag: dict) -> str:
    """Render human-readable formatted terminal diagnostic checklist."""
    c = diag["checks"]
    lines = [
        "==================================================",
        " HawkPoint NPU Preflight Diagnostics (Doctor)",
        "==================================================",
    ]

    # OS
    os_c = c["os_and_cpu"]
    lines.append(f"{_badge(os_c['status'])} Operating System: {os_c['os']} {os_c['arch']} (kernel {os_c['kernel']})")

    # NPU Hardware
    npu_c = c["npu_hardware"]
    if npu_c["accel_nodes"]:
        lines.append(f"{_badge(npu_c['status'])} NPU Device Nodes: {', '.join(npu_c['accel_nodes'])} (rw={npu_c['user_accessible']})")
    else:
        lines.append(f"{_badge(npu_c['status'])} NPU Device Nodes: None detected in /dev/accel")

    lines.append(f"{_badge(CheckStatus.OK if npu_c['driver_loaded'] else CheckStatus.WARN)} amdxdna Driver: {'loaded' if npu_c['driver_loaded'] else 'not loaded'}")

    # XRT
    xrt_c = c["xrt_runtime"]
    lines.append(f"{_badge(xrt_c['status'])} XRT Userspace: {'Installed' if xrt_c['xrt_installed'] else 'Missing'}")

    # Python
    py_c = c["python_environment"]
    lines.append(f"{_badge(py_c['status'])} Python Packages: {len(py_c['installed'])} present" + (f", {len(py_c['missing'])} missing" if py_c['missing'] else ""))

    # Models
    mod_c = c["model_catalog"]
    lines.append(f"{_badge(mod_c['status'])} Converted Models: {mod_c['models_count']} found in {mod_c['models_dir']}")

    # Network / Services
    net_c = c["network_services"]
    if net_c["api_server_active"]:
        state = "ready" if net_c["api_server_ready"] else "starting/draining"
        lines.append(f"[✓] Local API Server: Active on port 8000 ({state})")
    else:
        lines.append(f"[✓] Local API Server: Port 8000 {'in use' if net_c['port_8000_in_use'] else 'free'}")

    lines.append("--------------------------------------------------")

    # Collect remediation actions
    actions = []
    for item in c.values():
        for r in item.get("remediation", []):
            actions.append(r)
        for issue in item.get("issues", []):
            if not item.get("remediation"):
                actions.append(issue)

    if actions:
        lines.append("Actionable Recommendations:")
        for idx, act in enumerate(actions, 1):
            lines.append(f"  {idx}. {act}")
        lines.append("--------------------------------------------------")

    if diag["overall_status"] == CheckStatus.OK:
        lines.append("Status: All checks passed. System is ready for HawkPoint NPU!")
    elif diag["overall_status"] == CheckStatus.WARN:
        lines.append("Status: Warnings detected. Server will operate, but check notices above.")
    else:
        lines.append("Status: Critical issues detected. Please follow the recommendations above.")

    lines.append("==================================================")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HawkPoint NPU environment preflight diagnostics.")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "npu_llm/models", help="Model catalog directory")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    diag = run_diagnostics(models_dir=args.models_dir)

    if args.json:
        print(json.dumps(diag, indent=2))
    else:
        print(format_text_report(diag))

    if diag["overall_status"] == CheckStatus.FAIL:
        return 2
    if diag["overall_status"] == CheckStatus.WARN:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
