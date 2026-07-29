#!/usr/bin/env python3
"""Verify runtime rejection of missing and corrupted model package files."""

import hashlib
import json
from pathlib import Path
import tempfile

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
        (root / "metadata.json").write_text(json.dumps(metadata))
        XDNA1Model(root)

        weights.write_bytes(b"tampered model bytes")
        try:
            XDNA1Model(root)
            raise AssertionError("corrupted model package unexpectedly loaded")
        except RuntimeError as exc:
            assert "checksum mismatch" in str(exc)

        metadata.pop("files")
        weights.write_bytes(data)
        (root / "metadata.json").write_text(json.dumps(metadata))
        try:
            XDNA1Model(root)
            raise AssertionError("package without integrity manifest loaded")
        except RuntimeError as exc:
            assert "no integrity manifest" in str(exc)
    print("PASS model package integrity")


if __name__ == "__main__":
    main()
