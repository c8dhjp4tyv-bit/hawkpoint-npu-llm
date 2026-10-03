#!/usr/bin/env python3
"""Tests for launcher.py."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import launcher  # noqa: E402


def test_api_command_generation():
    """Verify api_command builds correct argument vectors."""
    base_cmd = launcher.api_command("127.0.0.1")
    assert base_cmd[0] == sys.executable
    assert str(launcher.API) in base_cmd
    assert "--host" in base_cmd
    assert "127.0.0.1" in base_cmd
    assert "--port" in base_cmd
    assert "8000" in base_cmd

    # With models_dir
    models_path = Path("/custom/models")
    cmd_with_models = launcher.api_command("0.0.0.0", models_dir=models_path)
    assert "--models-dir" in cmd_with_models
    assert str(models_path) in cmd_with_models

    # With npu_layers
    cmd_with_layers = launcher.api_command("127.0.0.1", npu_layers=16)
    assert "--npu-layers" in cmd_with_layers
    assert "16" in cmd_with_layers

    # With npu_percent
    cmd_with_pct = launcher.api_command("127.0.0.1", npu_percent=0.75)
    assert "--npu-percent" in cmd_with_pct
    assert "0.75" in cmd_with_pct


def test_launcher_subcommand_dispatch():
    """Verify dispatch for non-interactive subcommands."""
    # Doctor mode dispatch
    with patch("sys.argv", ["launcher.py", "doctor"]), \
         patch("subprocess.run") as mock_run, \
         patch("sys.exit") as mock_exit:
        mock_run.return_value = MagicMock(returncode=0)
        launcher.main()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "doctor.py" in str(cmd[1])
        mock_exit.assert_called_once_with(0)

    # Inspect mode dispatch
    with patch("sys.argv", ["launcher.py", "inspect"]), \
         patch("subprocess.run") as mock_run, \
         patch("sys.exit") as mock_exit:
        mock_run.return_value = MagicMock(returncode=0)
        launcher.main()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "inspect_model.py" in str(cmd[1])
        mock_exit.assert_called_once_with(0)

    # Benchmark mode dispatch
    with patch("sys.argv", ["launcher.py", "benchmark", "--api-key", "test-key"]), \
         patch("subprocess.run") as mock_run, \
         patch("sys.exit") as mock_exit:
        mock_run.return_value = MagicMock(returncode=0)
        launcher.main()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "benchmark_api.py" in str(cmd[1])
        assert "--api-key" in cmd
        assert "test-key" in cmd
        mock_exit.assert_called_once_with(0)

    # Eval mode dispatch
    with patch("sys.argv", ["launcher.py", "eval", "--api-key", "test-key"]), \
         patch("subprocess.run") as mock_run, \
         patch("sys.exit") as mock_exit:
        mock_run.return_value = MagicMock(returncode=0)
        launcher.main()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "eval_model.py" in str(cmd[1])
        assert "--api-key" in cmd
        assert "test-key" in cmd
        mock_exit.assert_called_once_with(0)


def test_launcher_interactive_menu_selection():
    """Verify numeric selection in interactive menu."""
    choices_to_script = {
        "4": "doctor.py",
        "5": "inspect_model.py",
        "6": "benchmark_api.py",
        "7": "eval_model.py",
    }
    for choice, expected_script in choices_to_script.items():
        with patch("sys.argv", ["launcher.py"]), \
             patch("builtins.input", return_value=choice), \
             patch("subprocess.run") as mock_run, \
             patch("sys.exit") as mock_exit:
            mock_run.return_value = MagicMock(returncode=0)
            launcher.main()
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert expected_script in str(cmd[1])
            mock_exit.assert_called_once_with(0)


def run_all_tests():
    test_api_command_generation()
    test_launcher_subcommand_dispatch()
    test_launcher_interactive_menu_selection()
    print("PASS 3 launcher tests")


if __name__ == "__main__":
    run_all_tests()
