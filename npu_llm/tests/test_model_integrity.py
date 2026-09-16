#!/usr/bin/env python3
"""Verify runtime rejection of missing and corrupted model package files."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.model import XDNA1Model


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        weights = root / "weights.bin"
        data = b"verified model bytes"
        weights.write_bytes(data)
        metadata = {
            "tensors": {},
            "files": {
                weights.name: {
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            },
        }
        def write_metadata(payload):
            payload = dict(payload)
            payload.pop("metadata_sha256", None)
            payload["metadata_sha256"] = hashlib.sha256(
                json.dumps(
                    payload, separators=(",", ":"), sort_keys=True
                ).encode()
            ).hexdigest()
            (root / "metadata.json").write_text(json.dumps(payload))
            return payload

        write_metadata(metadata)
        XDNA1Model(root)

        canonical = write_metadata(metadata)
        XDNA1Model(root)
        canonical["tensors"] = {"tampered": {}}
        (root / "metadata.json").write_text(json.dumps(canonical))
        try:
            XDNA1Model(root)
            raise AssertionError("tampered metadata unexpectedly loaded")
        except RuntimeError as exc:
            assert "metadata checksum mismatch" in str(exc)

        weights.write_bytes(b"tampered model bytes")
        try:
            XDNA1Model(root)
            raise AssertionError("corrupted model package unexpectedly loaded")
        except RuntimeError as exc:
            assert "checksum mismatch" in str(exc)

        metadata.pop("files")
        weights.write_bytes(data)
        write_metadata(metadata)
        try:
            XDNA1Model(root)
            raise AssertionError("package without integrity manifest loaded")
        except RuntimeError as exc:
            assert "no integrity manifest" in str(exc)

        outside = root.parent / "hawkpoint-integrity-outside.bin"
        outside.write_bytes(b"outside")
        try:
            write_metadata(
                {
                    "tensors": {},
                    "files": {
                        "../hawkpoint-integrity-outside.bin": {
                            "size": outside.stat().st_size,
                            "sha256": hashlib.sha256(
                                outside.read_bytes()
                            ).hexdigest(),
                        }
                    },
                }
            )
            try:
                XDNA1Model(root)
                raise AssertionError("path traversal package unexpectedly loaded")
            except RuntimeError as exc:
                assert "escapes package root" in str(exc)
        finally:
            outside.unlink()
    print("PASS model package integrity")


if __name__ == "__main__":
    main()
