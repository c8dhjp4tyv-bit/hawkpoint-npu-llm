import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


class QuantizedTensor:
    def __init__(self, root, spec):
        self.shape = tuple(spec["shape"])
        self.weight = np.memmap(
            root / spec["weight"], mode="r", dtype=np.int8, shape=self.shape
        )
        self.scale = np.memmap(
            root / spec["scale"],
            mode="r",
            dtype=np.float32,
            shape=(self.shape[0],),
        )


class XDNA1Model:
    def __init__(self, model_dir):
        self.root = Path(model_dir)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        self._verify_integrity()
        self.tensors = self.metadata["tensors"]

    def _verify_integrity(self):
        files = self.metadata.get("files")
        if not files:
            raise RuntimeError(
                "model package has no integrity manifest; reconvert the model"
            )
        for name, expected in files.items():
            path = self.root / name
            if not path.is_file() or path.stat().st_size != expected["size"]:
                raise RuntimeError(f"model package file is missing or truncated: {name}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected["sha256"]:
                raise RuntimeError(f"model package checksum mismatch: {name}")

    def quantized(self, name):
        return QuantizedTensor(self.root, self.tensors[name])

    def bf16_projection(self, name):
        spec = self.tensors[name]
        if "bf16_file" not in spec:
            raise RuntimeError(
                f"{name} has no BF16 weights; rerun convert_smollm2.py"
            )
        return np.memmap(
            self.root / spec["bf16_file"],
            mode="r",
            dtype=bfloat16,
            shape=tuple(spec["shape"]),
        )

    def raw(self, name):
        spec = self.tensors[name]
        return np.memmap(
            self.root / spec["file"],
            mode="r",
            dtype=np.dtype(spec["dtype"]),
            shape=tuple(spec["shape"]),
        )

    def layer(self, index):
        prefix = f"layer{index:02d}"
        return {
            "qkv": self.quantized(f"{prefix}.qkv"),
            "o_proj": self.quantized(f"{prefix}.o_proj"),
            "gate_up": self.quantized(f"{prefix}.gate_up"),
            "down_proj": self.quantized(f"{prefix}.down_proj"),
            "input_norm": self.raw(f"{prefix}.input_norm"),
            "post_attn_norm": self.raw(f"{prefix}.post_attn_norm"),
        }
