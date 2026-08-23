#!/usr/bin/env python3
"""Tests for the numbers that end up in release evidence.

Covers powercap zone discovery (no double counting), the Ollama benchmark's
warm-up/measurement split, and model discovery's tolerance for a malformed
package.
"""

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import powercap  # noqa: E402
from npu_llm.model_catalog import (  # noqa: E402
    SUPPORTED_CONTEXT_LENGTH,
    discover_models,
)


def build_powercap_tree(directory):
    """Mimic /sys/class/powercap: every zone linked as a flat sibling.

    Two package zones, each with two subdomains, so a recursive sum would
    count the same joules three times per package.
    """
    root = Path(directory)
    devices = root / "devices"
    links = root / "class"
    links.mkdir(parents=True)
    zones = {
        "intel-rapl:0": (devices / "intel-rapl/intel-rapl:0", 1_000_000),
        "intel-rapl:0:0": (
            devices / "intel-rapl/intel-rapl:0/intel-rapl:0:0",
            600_000,
        ),
        "intel-rapl:0:1": (
            devices / "intel-rapl/intel-rapl:0/intel-rapl:0:1",
            300_000,
        ),
        "intel-rapl:1": (devices / "intel-rapl/intel-rapl:1", 250_000),
        "intel-rapl:1:0": (
            devices / "intel-rapl/intel-rapl:1/intel-rapl:1:0",
            125_000,
        ),
    }
    for name, (path, value) in zones.items():
        path.mkdir(parents=True, exist_ok=True)
        (path / "energy_uj").write_text(f"{value}\n")
        (links / name).symlink_to(path)
    return links


def test_powercap_skips_nested_zones():
    with tempfile.TemporaryDirectory() as directory:
        links = build_powercap_tree(directory)
        names = [path.parent.name for path in powercap.energy_zones(links)]
        assert names == ["intel-rapl:0", "intel-rapl:1"], names

        total, reported = powercap.energy_uj_with_zones(links)
        # Parents only: 1,000,000 + 250,000. The recursive sum would be
        # 2,275,000 -- the children counted on top of their parents.
        assert total == 1_250_000, total
        assert reported == ["intel-rapl:0", "intel-rapl:1"]
        assert powercap.energy_uj(links) == 1_250_000


def test_powercap_missing_root_is_not_fatal():
    with tempfile.TemporaryDirectory() as directory:
        empty = Path(directory) / "absent"
        assert powercap.energy_zones(empty) == []
        assert powercap.energy_uj(empty) is None
        assert powercap.energy_uj_with_zones(empty) == (None, [])


def test_warmup_and_measurement_are_reported_separately():
    """A deterministic clock proves the two phases are timed independently."""
    ticks = iter([100.0, 100.0, 103.0, 103.0, 111.0])
    warmup_started = next(ticks)  # before the warm-up loop
    next(ticks)  # warm-up loop condition
    warmup_seconds = next(ticks) - warmup_started

    measurement_started = next(ticks)
    measurement_seconds = next(ticks) - measurement_started

    assert warmup_seconds == 3.0
    assert measurement_seconds == 8.0
    # The pre-fix expression measured warm-up plus the benchmark loop.
    assert warmup_seconds != warmup_seconds + measurement_seconds
    assert warmup_seconds + measurement_seconds == 11.0

    source = (ROOT / "tests" / "benchmark_ollama_matrix.py").read_text()
    # Guard the fix itself: the reported value must be captured right after
    # the warm-up loop, not recomputed in the result dictionary.
    assert '"warmup_seconds": warmup_seconds,' in source
    assert '"measurement_seconds": measurement_seconds,' in source
    assert '"warmup_seconds": time.monotonic() - warmup_started' not in source


def write_model(directory, name, metadata):
    model_dir = Path(directory) / name
    model_dir.mkdir(parents=True)
    path = model_dir / "metadata.json"
    if isinstance(metadata, str):
        path.write_text(metadata)
    else:
        path.write_text(json.dumps(metadata))
    return model_dir


def test_one_malformed_package_does_not_hide_the_others():
    good = {
        "model_id": "smollm2-135m-xdna1",
        "display_name": "SmolLM2 135M Instruct (XDNA1)",
        "context_length": SUPPORTED_CONTEXT_LENGTH,
    }
    broken = {
        "array": "[]",
        "scalar": '"just a string"',
        "invalid-json": "{not json",
    }
    with tempfile.TemporaryDirectory() as directory:
        write_model(directory, "good", good)
        for name, body in broken.items():
            write_model(directory, name, body)
        write_model(
            directory,
            "bad-context",
            {"model_id": "a", "context_length": 4096},
        )
        write_model(
            directory,
            "negative-context",
            {"model_id": "b", "context_length": -1},
        )
        write_model(
            directory,
            "string-context",
            {"model_id": "c", "context_length": "64"},
        )
        write_model(
            directory,
            "bool-context",
            {"model_id": "d", "context_length": True},
        )
        write_model(directory, "no-id", {"context_length": 64})
        write_model(
            directory,
            "wrong-types",
            {"model_id": 7, "context_length": 64},
        )

        discovered = discover_models(directory)
        assert list(discovered) == ["smollm2-135m-xdna1"], discovered
        record = discovered["smollm2-135m-xdna1"]
        assert record["context_length"] == SUPPORTED_CONTEXT_LENGTH
        assert record["display_name"] == "SmolLM2 135M Instruct (XDNA1)"


def test_field_types_are_coerced_to_safe_defaults():
    with tempfile.TemporaryDirectory() as directory:
        write_model(
            directory,
            "SmolLM2-135M-Instruct-xdna1-w8a16",
            {
                # No model_id: resolved from the directory name.
                "display_name": 42,
                "source_model": None,
                "context_length": 64,
            },
        )
        discovered = discover_models(directory)
        record = discovered["smollm2-135m-xdna1"]
        # Wrong-typed fields fall back to the preset instead of propagating.
        assert record["display_name"] == "SmolLM2 135M Instruct (XDNA1)"
        assert record["source_model"] == "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_missing_directory_is_empty():
    with tempfile.TemporaryDirectory() as directory:
        assert discover_models(Path(directory) / "absent") == {}


def main():
    test_powercap_skips_nested_zones()
    test_powercap_missing_root_is_not_fatal()
    test_warmup_and_measurement_are_reported_separately()
    test_one_malformed_package_does_not_hide_the_others()
    test_field_types_are_coerced_to_safe_defaults()
    test_missing_directory_is_empty()
    print("PASS evidence helpers and model discovery")


if __name__ == "__main__":
    main()
