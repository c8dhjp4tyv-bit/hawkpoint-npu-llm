#!/usr/bin/env python3
"""Launch either the local API or API + Open WebUI."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
API = ROOT / "npu_llm/api_server.py"


def api_command(host, models_dir=None):
    command = [sys.executable, str(API), "--host", host, "--port", "8000"]
    if models_dir:
        command.extend(["--models-dir", str(models_dir)])
    return command


def wait_for_api(process, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("API server exited before becoming ready")
        try:
            with urlopen("http://127.0.0.1:8000/health", timeout=1):
                return
        except URLError:
            time.sleep(1)
    raise TimeoutError("API server did not become ready within 120 seconds")


def run_openwebui(models_dir=None):
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for the Open WebUI option")
    api = subprocess.Popen(api_command("0.0.0.0", models_dir), cwd=ROOT)
    try:
        wait_for_api(api)
        print("Open WebUI will be available at http://localhost:3000")
        subprocess.run(
            ["docker", "compose", "up", "--pull", "missing"],
            cwd=ROOT,
            check=True,
        )
    finally:
        api.terminate()
        try:
            api.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["api", "openwebui"])
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="directory containing converted model subdirectories",
    )
    args = parser.parse_args()
    mode = args.mode
    if mode is None:
        print("1) OpenAI-compatible API server (localhost:8000)")
        print("2) API server + Open WebUI (localhost:3000)")
        choice = input("Choose [1/2]: ").strip()
        mode = "openwebui" if choice == "2" else "api"

    if mode == "api":
        subprocess.run(
            api_command("127.0.0.1", args.models_dir),
            cwd=ROOT,
            check=True,
        )
    else:
        run_openwebui(args.models_dir)


if __name__ == "__main__":
    main()
