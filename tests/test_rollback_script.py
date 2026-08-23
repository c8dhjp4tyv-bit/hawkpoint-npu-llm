#!/usr/bin/env python3
"""Fault-injection tests for ollama-xdna/scripts/rollback.sh.

The script is driven against a synthetic filesystem with stubbed
``systemctl``/``curl`` (``OLLAMA_XDNA_TEST_ROOT``), so each destructive step
can be failed on purpose and the resulting state inspected. The invariant
under test: rollback either completes, or puts the XDNA install back; no good
copy is ever lost.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ollama-xdna" / "scripts" / "rollback.sh"
STAMP = "20260101-000000"


def build_tree(directory):
    """Create an XDNA install plus a matching pre-XDNA backup."""
    root = Path(directory)
    lib = root / "usr/local/lib"
    binaries = root / "usr/local/bin"
    drop_in_dir = root / "etc/systemd/system/ollama.service.d"
    lib.mkdir(parents=True)
    binaries.mkdir(parents=True)
    drop_in_dir.mkdir(parents=True)

    # Current (XDNA) install.
    (lib / "ollama").mkdir()
    (lib / "ollama/llama-server").write_text("xdna llama-server")
    (lib / "ollama/xdna").mkdir()
    (binaries / "ollama").write_text("xdna ollama")
    (binaries / "ollama").chmod(0o755)
    (drop_in_dir / "xdna.conf").write_text("[Service]\nxdna=1\n")

    # Previous install, kept by install-stage.sh.
    backup_runtime = lib / f"ollama.pre-xdna-{STAMP}"
    backup_runtime.mkdir()
    (backup_runtime / "llama-server").write_text("previous llama-server")
    (backup_runtime / "llama-server").chmod(0o755)
    backup_binary = binaries / f"ollama.pre-xdna-{STAMP}"
    backup_binary.write_text("previous ollama")
    backup_binary.chmod(0o755)
    return root


def write_stub(path, script):
    path.write_text("#!/usr/bin/env bash\n" + script)
    path.chmod(0o755)


def stubs(directory, fail_on=None, log=None):
    """Create systemctl/curl stubs; ``fail_on`` fails one systemctl verb."""
    stub_dir = Path(directory) / "stubs"
    stub_dir.mkdir(exist_ok=True)
    log = log or (stub_dir / "calls.log")
    systemctl = stub_dir / "systemctl"
    write_stub(
        systemctl,
        f'echo "systemctl $*" >> "{log}"\n'
        f'if [[ "$1" == "{fail_on or "__never__"}" ]]; then exit 1; fi\n'
        'exit 0\n',
    )
    curl = stub_dir / "curl"
    write_stub(curl, 'exit 0\n')
    return systemctl, curl, log


def run(root, systemctl, curl, extra_env=None):
    environment = {
        **os.environ,
        "OLLAMA_XDNA_TEST_ROOT": str(root),
        "OLLAMA_XDNA_SYSTEMCTL": str(systemctl),
        "OLLAMA_XDNA_CURL": str(curl),
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )


def state(root):
    lib = Path(root) / "usr/local/lib"
    binaries = Path(root) / "usr/local/bin"
    drop_in = Path(root) / "etc/systemd/system/ollama.service.d/xdna.conf"
    runtime = lib / "ollama/llama-server"
    return {
        "runtime": runtime.read_text() if runtime.exists() else None,
        "binary": (binaries / "ollama").read_text()
        if (binaries / "ollama").exists()
        else None,
        "drop_in": drop_in.read_text() if drop_in.exists() else None,
        "backup_runtime": sorted(p.name for p in lib.glob("ollama.pre-xdna-*")),
        "backup_binary": sorted(
            p.name for p in binaries.glob("ollama.pre-xdna-*")
        ),
        "failed_runtime": sorted(p.name for p in lib.glob("ollama.failed-xdna-*")),
        "failed_binary": sorted(
            p.name for p in binaries.glob("ollama.failed-xdna-*")
        ),
    }


def test_successful_rollback():
    with tempfile.TemporaryDirectory() as directory:
        root = build_tree(directory)
        systemctl, curl, _ = stubs(directory)
        result = run(root, systemctl, curl)
        assert result.returncode == 0, result.stderr
        after = state(root)
        assert after["runtime"] == "previous llama-server"
        assert after["binary"] == "previous ollama"
        # No drop-in backup existed, so the XDNA drop-in is removed.
        assert after["drop_in"] is None
        assert after["backup_runtime"] == []
        assert after["backup_binary"] == []
        # The displaced XDNA install is retained, never deleted.
        assert len(after["failed_runtime"]) == 1
        assert len(after["failed_binary"]) == 1


def test_missing_backup_binary_aborts_before_touching_anything():
    with tempfile.TemporaryDirectory() as directory:
        root = build_tree(directory)
        (root / f"usr/local/bin/ollama.pre-xdna-{STAMP}").unlink()
        systemctl, curl, log = stubs(directory)
        before = state(root)
        result = run(root, systemctl, curl)
        assert result.returncode != 0
        assert "binary backup is missing" in result.stderr
        assert state(root) == before
        # Validation happens before the service is even stopped.
        assert not log.exists() or "stop" not in log.read_text()


def test_unusable_backup_runtime_aborts():
    """A backup that cannot serve must not displace a working install."""
    with tempfile.TemporaryDirectory() as directory:
        root = build_tree(directory)
        (root / f"usr/local/lib/ollama.pre-xdna-{STAMP}/llama-server").unlink()
        systemctl, curl, _ = stubs(directory)
        before = state(root)
        result = run(root, systemctl, curl)
        assert result.returncode != 0
        assert "no llama-server" in result.stderr
        assert state(root) == before


def test_service_start_failure_restores_the_xdna_install():
    with tempfile.TemporaryDirectory() as directory:
        root = build_tree(directory)
        systemctl, curl, _ = stubs(directory, fail_on="start")
        result = run(root, systemctl, curl)
        assert result.returncode != 0
        assert "the XDNA install was put back" in result.stderr
        after = state(root)
        # Every component is back where it started, and the backup is intact.
        assert after["runtime"] == "xdna llama-server"
        assert after["binary"] == "xdna ollama"
        assert after["drop_in"] == "[Service]\nxdna=1\n"
        assert after["backup_runtime"] == [f"ollama.pre-xdna-{STAMP}"]
        assert after["backup_binary"] == [f"ollama.pre-xdna-{STAMP}"]
        assert after["failed_runtime"] == []
        assert after["failed_binary"] == []


def test_health_check_failure_restores_the_xdna_install():
    with tempfile.TemporaryDirectory() as directory:
        root = build_tree(directory)
        systemctl, curl, _ = stubs(directory)
        write_stub(curl, "exit 22\n")
        result = run(root, systemctl, curl, {"OLLAMA_XDNA_HEALTH_TRIES": "1"})
        assert result.returncode != 0
        after = state(root)
        assert after["runtime"] == "xdna llama-server"
        assert after["binary"] == "xdna ollama"
        assert after["backup_runtime"] == [f"ollama.pre-xdna-{STAMP}"]
        assert after["backup_binary"] == [f"ollama.pre-xdna-{STAMP}"]


def test_binary_restore_failure_restores_the_runtime_too():
    """Failure after the runtime move must not leave a mixed install."""
    with tempfile.TemporaryDirectory() as directory:
        root = build_tree(directory)
        systemctl, curl, _ = stubs(directory)
        # Fail exactly the binary-restore move, after the runtime has already
        # been swapped, by shadowing mv on PATH.
        stub_dir = Path(directory) / "stubs"
        write_stub(
            stub_dir / "mv",
            'if [[ "$1" == *pre-xdna* && "$2" == */usr/local/bin/ollama ]]; then\n'
            '    echo "injected mv failure" >&2\n'
            "    exit 1\n"
            "fi\n"
            'exec /bin/mv "$@"\n',
        )
        result = run(
            root,
            systemctl,
            curl,
            {"PATH": f"{stub_dir}:{os.environ['PATH']}"},
        )
        assert result.returncode != 0
        assert "the XDNA install was put back" in result.stderr
        after = state(root)
        assert after["runtime"] == "xdna llama-server"
        assert after["binary"] == "xdna ollama"
        assert after["backup_runtime"] == [f"ollama.pre-xdna-{STAMP}"]
        assert after["failed_runtime"] == []


def main():
    if shutil.which("bash") is None:
        print("SKIP rollback script tests: bash is unavailable")
        return
    test_successful_rollback()
    test_missing_backup_binary_aborts_before_touching_anything()
    test_unusable_backup_runtime_aborts()
    test_service_start_failure_restores_the_xdna_install()
    test_health_check_failure_restores_the_xdna_install()
    test_binary_restore_failure_restores_the_runtime_too()
    print("PASS rollback fault injection")


if __name__ == "__main__":
    sys.exit(main())
