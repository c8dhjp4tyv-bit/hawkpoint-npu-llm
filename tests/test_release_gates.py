#!/usr/bin/env python3
"""Hosted tests for immutable release pins and placement agreement."""

import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from benchmark_ollama_matrix import compare_placement_responses  # noqa: E402
from verify_ollama_manifest import verify_manifest  # noqa: E402


def test_release_pins():
    pins = json.loads((ROOT / "release-pins.json").read_text())
    ollama = pins["ollama"]
    assert re.fullmatch(r"[0-9a-f]{40}", ollama["source_commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", ollama["model_manifest_sha256"])
    build_script = (
        ROOT / "ollama-xdna" / "scripts" / "build-and-install.sh"
    ).read_text()
    assert ollama["source_commit"] in build_script


def test_manifest_digest():
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
    matching = [
        {"placement": name, "response_sha256": "a" * 64}
        for name in ("cpu_only", "gpu_only", "cpu_gpu", "cpu_gpu_npu")
    ]
    hashes, agreed = compare_placement_responses(matching, 4)
    assert agreed and len(hashes) == 4
    matching[-1]["response_sha256"] = "b" * 64
    _, agreed = compare_placement_responses(matching, 4)
    assert not agreed


def main():
    test_release_pins()
    test_manifest_digest()
    test_cross_placement_agreement()
    print("PASS immutable release gates")


if __name__ == "__main__":
    main()
