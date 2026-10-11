#!/usr/bin/env python3
"""Tests for repository examples."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_api_client_example():
    script = ROOT / "examples/api_client_example.py"
    cmd = [sys.executable, str(script), "--self-test"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"stdout: {res.stdout}\nstderr: {res.stderr}"
    assert "All self-test assertions PASSED successfully!" in res.stdout


if __name__ == "__main__":
    test_api_client_example()
    print("PASS 1 example tests")
