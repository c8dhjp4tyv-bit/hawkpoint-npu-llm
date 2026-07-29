#!/usr/bin/env python3
"""Verify a locally pulled Ollama tag against a pinned registry manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def manifest_path(models_dir, model):
    name, separator, tag = model.partition(":")
    if not separator or not name or not tag or "@" in model:
        raise ValueError("model must use an explicit name:tag reference")
    parts = name.split("/")
    if len(parts) == 1:
        namespace, repository = "library", parts[0]
    elif len(parts) == 2:
        namespace, repository = parts
    else:
        raise ValueError("unsupported Ollama model name")
    return (
        Path(models_dir)
        / "manifests"
        / "registry.ollama.ai"
        / namespace
        / repository
        / tag
    )


def verify_manifest(models_dir, model, expected_sha256):
    path = manifest_path(models_dir, model)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    manifest = json.loads(payload)
    report = {
        "model": model,
        "manifest_path": str(path),
        "expected_manifest_sha256": expected_sha256,
        "actual_manifest_sha256": actual,
        "config_digest": manifest["config"]["digest"],
        "layer_digests": [layer["digest"] for layer in manifest["layers"]],
        "verified": actual == expected_sha256,
    }
    if actual != expected_sha256:
        raise RuntimeError(
            f"{model} manifest mismatch: expected {expected_sha256}, got {actual}"
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify_manifest(args.models_dir, args.model, args.sha256)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
