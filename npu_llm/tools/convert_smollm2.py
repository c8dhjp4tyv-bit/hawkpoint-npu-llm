"""Convert supported Hugging Face safetensors to the XDNA1 runtime format."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
from ml_dtypes import bfloat16  # noqa: F401 - registers BF16 with NumPy
from safetensors import safe_open


SMOLLM_ARCHITECTURE = {
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "num_hidden_layers": 30,
    "vocab_size": 49152,
}
QWEN_ARCHITECTURE = {
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "num_hidden_layers": 24,
    "vocab_size": 151936,
}
SUPPORTED_ARCHITECTURES = {
    "llama": SMOLLM_ARCHITECTURE,
    "qwen2": QWEN_ARCHITECTURE,
}
# Backward-compatible name used by the lightweight converter test.
EXPECTED_ARCHITECTURE = SMOLLM_ARCHITECTURE


def _tensor_files(model_dir: Path):
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        return {name: model_dir / file for name, file in mapping.items()}
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"no safetensors weights in {model_dir}")
    with safe_open(single, framework="np") as f:
        return {name: single for name in f.keys()}


class TensorReader:
    def __init__(self, model_dir: Path):
        self.files = _tensor_files(model_dir)

    def has(self, name):
        return name in self.files

    def get(self, name):
        path = self.files[name]
        with safe_open(path, framework="np") as f:
            return np.asarray(f.get_tensor(name))


def quantize_per_output_channel(weight):
    weight = np.asarray(weight, dtype=np.float32)
    max_abs = np.max(np.abs(weight), axis=1)
    scale = np.maximum(max_abs / 127.0, np.finfo(np.float32).tiny)
    quantized = np.clip(np.rint(weight / scale[:, None]), -127, 127).astype(np.int8)
    return quantized, scale.astype(np.float32)


def write_quantized(out_dir, name, weight, manifest):
    q, scale = quantize_per_output_channel(weight)
    bf16 = np.asarray(weight, dtype=bfloat16)
    q_path = out_dir / f"{name}.w8.bin"
    s_path = out_dir / f"{name}.scale.f32.bin"
    bf16_path = out_dir / f"{name}.bf16.bin"
    q.tofile(q_path)
    scale.tofile(s_path)
    bf16.tofile(bf16_path)
    manifest["tensors"][name] = {
        "weight": q_path.name,
        "scale": s_path.name,
        "shape": list(q.shape),
        "layout": "row_major",
        "weight_dtype": "int8",
        "scale_dtype": "float32",
        "quantization": "symmetric_per_output_channel",
        "bf16_file": bf16_path.name,
        "bf16_dtype": "bfloat16",
    }


def write_raw(out_dir, name, array, dtype, manifest):
    array = np.asarray(array, dtype=dtype)
    path = out_dir / f"{name}.{np.dtype(dtype).name}.bin"
    array.tofile(path)
    manifest["tensors"][name] = {
        "file": path.name,
        "shape": list(array.shape),
        "dtype": np.dtype(dtype).name,
    }


def _validate_architecture(config):
    family = config.get("model_type")
    expected_architecture = SUPPORTED_ARCHITECTURES.get(family)
    if expected_architecture is None:
        raise ValueError(
            f"unsupported model_type {family!r}; expected one of "
            + ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        )
    mismatches = {
        name: (config.get(name), expected)
        for name, expected in expected_architecture.items()
        if config.get(name) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected})"
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(
            "model is incompatible with the selected XDNA1 graph: "
            + details
        )
    head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    if head_dim != 64:
        raise ValueError(f"head_dim={head_dim} is unsupported; expected 64")
    return family


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _convert_into(
    model_dir: Path,
    out_dir: Path,
    *,
    model_id=None,
    source_model=None,
    source_revision=None,
    display_name=None,
):
    config = json.loads((model_dir / "config.json").read_text())
    model_family = _validate_architecture(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = TensorReader(model_dir)
    layers = int(config["num_hidden_layers"])
    manifest = {
        "format": (
            "xdna1-qwen2-w8a16-v1"
            if model_family == "qwen2"
            else "xdna1-smollm2-w8a16-v1"
        ),
        "model_family": model_family,
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "attention_heads": int(config["num_attention_heads"]),
        "kv_heads": int(config["num_key_value_heads"]),
        "head_dim": int(
            config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        ),
        "layers": layers,
        "vocab_size": int(config["vocab_size"]),
        "rope_theta": float(config.get("rope_theta", 10000.0)),
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-5)),
        "context_length": 64,
        "activation_dtype": "bfloat16",
        "accumulator_dtype": "float32",
        "tensors": {},
    }
    if model_id:
        manifest["model_id"] = model_id
    if source_model:
        manifest["source_model"] = source_model
    if source_revision:
        manifest["source_revision"] = source_revision
    if display_name:
        manifest["display_name"] = display_name

    embed = reader.get("model.embed_tokens.weight")
    write_raw(out_dir, "token_embedding", embed, np.float16, manifest)

    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        attn = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"
        qkv = np.concatenate(
            [
                reader.get(f"{attn}.q_proj.weight"),
                reader.get(f"{attn}.k_proj.weight"),
                reader.get(f"{attn}.v_proj.weight"),
            ],
            axis=0,
        )
        qkv_bias = np.concatenate(
            [
                (
                    reader.get(f"{attn}.{name}_proj.bias")
                    if reader.has(f"{attn}.{name}_proj.bias")
                    else np.zeros(
                        reader.get(f"{attn}.{name}_proj.weight").shape[0],
                        dtype=np.float32,
                    )
                )
                for name in ("q", "k", "v")
            ]
        )
        gate_up = np.concatenate(
            [
                reader.get(f"{mlp}.gate_proj.weight"),
                reader.get(f"{mlp}.up_proj.weight"),
            ],
            axis=0,
        )
        write_quantized(out_dir, f"layer{layer:02d}.qkv", qkv, manifest)
        write_raw(
            out_dir,
            f"layer{layer:02d}.qkv_bias",
            qkv_bias,
            np.float16,
            manifest,
        )
        write_quantized(
            out_dir,
            f"layer{layer:02d}.o_proj",
            reader.get(f"{attn}.o_proj.weight"),
            manifest,
        )
        write_quantized(out_dir, f"layer{layer:02d}.gate_up", gate_up, manifest)
        write_quantized(
            out_dir,
            f"layer{layer:02d}.down_proj",
            reader.get(f"{mlp}.down_proj.weight"),
            manifest,
        )
        write_raw(
            out_dir,
            f"layer{layer:02d}.input_norm",
            reader.get(f"{prefix}.input_layernorm.weight"),
            np.float16,
            manifest,
        )
        write_raw(
            out_dir,
            f"layer{layer:02d}.post_attn_norm",
            reader.get(f"{prefix}.post_attention_layernorm.weight"),
            np.float16,
            manifest,
        )

    write_raw(
        out_dir, "final_norm", reader.get("model.norm.weight"), np.float16, manifest
    )
    lm_name = "lm_head.weight"
    lm_head = reader.get(lm_name) if reader.has(lm_name) else embed
    write_quantized(out_dir, "lm_head", lm_head, manifest)

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "config.json",
    ]
    for filename in tokenizer_files:
        src = model_dir / filename
        if src.exists():
            (out_dir / filename).write_bytes(src.read_bytes())
    manifest["files"] = {
        path.name: {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != "metadata.json"
    }
    metadata = out_dir / "metadata.json"
    metadata.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with metadata.open("rb") as stream:
        os.fsync(stream.fileno())
    return manifest


def convert(model_dir: Path, out_dir: Path, **metadata):
    """Build and verify a complete model package, then publish it atomically."""
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent)
    )
    backup = out_dir.with_name(f".{out_dir.name}.backup-{uuid.uuid4().hex}")
    try:
        manifest = _convert_into(model_dir, staging, **metadata)
        for name, expected in manifest["files"].items():
            path = staging / name
            if path.stat().st_size != expected["size"]:
                raise OSError(f"size verification failed for {name}")
            if _sha256(path) != expected["sha256"]:
                raise OSError(f"checksum verification failed for {name}")
        for path in staging.iterdir():
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        _fsync_directory(staging)
        if out_dir.exists():
            os.replace(out_dir, backup)
        try:
            os.replace(staging, out_dir)
            _fsync_directory(out_dir.parent)
        except Exception:
            if backup.exists():
                os.replace(backup, out_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    manifest = convert(args.model_dir, args.output_dir)
    print(
        f"Converted {manifest['layers']} layers, "
        f"hidden={manifest['hidden_size']}, vocab={manifest['vocab_size']}"
    )


if __name__ == "__main__":
    main()
