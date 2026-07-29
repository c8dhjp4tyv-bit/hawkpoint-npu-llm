"""Convert SmolLM2 safetensors to the XDNA1 NPU runtime format."""

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16  # noqa: F401 - registers BF16 with NumPy
from safetensors import safe_open


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


def convert(model_dir: Path, out_dir: Path):
    config = json.loads((model_dir / "config.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = TensorReader(model_dir)
    layers = int(config["num_hidden_layers"])
    manifest = {
        "format": "xdna1-smollm2-w8a16-v1",
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "attention_heads": int(config["num_attention_heads"]),
        "kv_heads": int(config["num_key_value_heads"]),
        "head_dim": int(
            config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        ),
        "layers": layers,
        "vocab_size": int(config["vocab_size"]),
        "context_length": 64,
        "activation_dtype": "int16",
        "accumulator_dtype": "int32",
        "tensors": {},
    }

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
        gate_up = np.concatenate(
            [
                reader.get(f"{mlp}.gate_proj.weight"),
                reader.get(f"{mlp}.up_proj.weight"),
            ],
            axis=0,
        )
        write_quantized(out_dir, f"layer{layer:02d}.qkv", qkv, manifest)
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

    (out_dir / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
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
    return manifest


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
