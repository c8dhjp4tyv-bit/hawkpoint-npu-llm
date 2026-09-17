import hashlib
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


_MAX_TENSOR_ELEMENTS = 1 << 31


def _safe_model_path(root, name):
    """Resolve a manifest path while keeping it inside the model package."""
    if not isinstance(name, str) or not name:
        raise RuntimeError("model manifest contains an invalid file path")
    candidate = Path(name)
    if candidate.is_absolute():
        raise RuntimeError(f"absolute model path is not allowed: {name}")
    root = Path(root).resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"model path escapes package root: {name}") from exc
    return resolved


def _shape(shape, name):
    if not isinstance(shape, (list, tuple)) or not shape:
        raise RuntimeError(f"tensor {name} has an invalid shape")
    values = []
    elements = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise RuntimeError(f"tensor {name} has a non-integer shape")
        if dimension <= 0:
            raise RuntimeError(f"tensor {name} has a non-positive shape")
        elements *= dimension
        if elements > _MAX_TENSOR_ELEMENTS:
            raise RuntimeError(f"tensor {name} is too large")
        values.append(dimension)
    return tuple(values), elements


def _checked_memmap(root, name, *, dtype, shape, files):
    path = _safe_model_path(root, name)
    dtype = np.dtype(dtype)
    _, elements = _shape(shape, name)
    expected_size = elements * dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"tensor file size mismatch for {name}: "
            f"expected {expected_size}, got {actual_size}"
        )
    if files is not None and name not in files:
        raise RuntimeError(f"tensor file is missing from integrity manifest: {name}")
    return np.memmap(path, mode="r", dtype=dtype, shape=tuple(shape))


class QuantizedTensor:
    def __init__(self, root, spec, files=None):
        if not isinstance(spec, dict):
            raise RuntimeError("quantized tensor specification must be an object")
        self.shape, _ = _shape(spec.get("shape"), "quantized tensor")
        if len(self.shape) != 2:
            raise RuntimeError("quantized tensors must be rank two")
        self.weight = _checked_memmap(
            root,
            spec.get("weight"),
            dtype=np.int8,
            shape=self.shape,
            files=files,
        )
        self.scale = _checked_memmap(
            root,
            spec.get("scale"),
            dtype=np.float32,
            shape=(self.shape[0],),
            files=files,
        )


class XDNA1Model:
    def __init__(self, model_dir):
        self.root = Path(model_dir).resolve()
        metadata_path = _safe_model_path(self.root, "metadata.json")
        try:
            self.metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("model package has invalid metadata.json") from exc
        if not isinstance(self.metadata, dict):
            raise RuntimeError("model metadata must be a JSON object")
        tensors = self.metadata.get("tensors")
        if not isinstance(tensors, dict):
            raise RuntimeError("model metadata has no tensor manifest")
        metadata_digest = self.metadata.get("metadata_sha256")
        if not isinstance(metadata_digest, str) or len(metadata_digest) != 64:
            raise RuntimeError(
                "model package has no metadata_sha256; reconvert the model"
            )
        canonical = dict(self.metadata)
        canonical.pop("metadata_sha256", None)
        actual = hashlib.sha256(
            json.dumps(
                canonical, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()
        if actual != metadata_digest:
            raise RuntimeError("model metadata checksum mismatch")
        self._verify_integrity()
        self.tensors = tensors

    def _verify_integrity(self):
        files = self.metadata.get("files")
        if not isinstance(files, dict) or not files:
            raise RuntimeError(
                "model package has no integrity manifest; reconvert the model"
            )
        for name, expected in files.items():
            path = _safe_model_path(self.root, name)
            if not isinstance(expected, dict):
                raise RuntimeError(f"invalid integrity entry: {name}")
            expected_size = expected.get("size")
            expected_sha256 = expected.get("sha256")
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
            ):
                raise RuntimeError(f"invalid integrity entry: {name}")
            if not path.is_file() or path.stat().st_size != expected_size:
                raise RuntimeError(f"model package file is missing or truncated: {name}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise RuntimeError(f"model package checksum mismatch: {name}")

        # Every binary referenced by the tensor manifest must be covered by the
        # package checksum list.  Tokenizer/config files may also be present,
        # but no tensor may silently bypass integrity verification.
        for tensor_name, spec in self.metadata["tensors"].items():
            if not isinstance(spec, dict):
                raise RuntimeError(f"invalid tensor specification: {tensor_name}")
            for field in ("file", "weight", "scale", "bf16_file"):
                filename = spec.get(field)
                if filename is not None and filename not in files:
                    raise RuntimeError(
                        f"tensor file is missing from integrity manifest: {filename}"
                    )

    def quantized(self, name):
        return QuantizedTensor(self.root, self.tensors[name], self.metadata["files"])

    def bf16_projection(self, name):
        spec = self.tensors[name]
        if "bf16_file" not in spec:
            raise RuntimeError(
                f"{name} has no BF16 weights; rerun convert_smollm2.py"
            )
        shape, _ = _shape(spec.get("shape"), name)
        return _checked_memmap(
            self.root,
            spec["bf16_file"],
            dtype=bfloat16,
            shape=shape,
            files=self.metadata["files"],
        )

    def raw(self, name):
        spec = self.tensors[name]
        shape, _ = _shape(spec.get("shape"), name)
        try:
            dtype = np.dtype(spec["dtype"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"tensor {name} has an invalid dtype") from exc
        return _checked_memmap(
            self.root,
            spec["file"],
            dtype=dtype,
            shape=shape,
            files=self.metadata["files"],
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
