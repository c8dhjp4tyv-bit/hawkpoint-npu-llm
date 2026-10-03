#!/usr/bin/env python3
"""Inspect, validate, and compute analytical hardware metrics for HawkPoint NPU models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def compute_parameter_breakdown(metadata: dict) -> dict:
    """Calculate exact parameter counts for each sub-block of the transformer."""
    hidden = int(metadata.get("hidden_size", 576))
    intermediate = int(metadata.get("intermediate_size", 1536))
    q_heads = int(metadata.get("attention_heads", 9))
    kv_heads = int(metadata.get("kv_heads", 3))
    head_dim = int(metadata.get("head_dim", 64))
    layers = int(metadata.get("layers", 30))
    vocab = int(metadata.get("vocab_size", 49152))

    # Attention weights per layer
    q_params = hidden * (q_heads * head_dim)
    k_params = hidden * (kv_heads * head_dim)
    v_params = hidden * (kv_heads * head_dim)
    o_params = (q_heads * head_dim) * hidden
    attn_per_layer = q_params + k_params + v_params + o_params

    # SwiGLU MLP weights per layer (gate, up, down)
    mlp_per_layer = (hidden * intermediate) * 2 + (intermediate * hidden)

    # RMSNorm weights (input_layernorm + post_attention_layernorm)
    norm_per_layer = hidden * 2

    # Cumulative layer parameters
    layer_total = (attn_per_layer + mlp_per_layer + norm_per_layer) * layers

    # Embeddings and output LM head
    embed_params = vocab * hidden
    final_norm_params = hidden
    lm_head_params = vocab * hidden

    total_params = embed_params + layer_total + final_norm_params + lm_head_params

    # Weight size estimations depending on quantization
    format_name = metadata.get("format", "xdna1-w8a16")
    is_w8 = "w8" in format_name
    bytes_per_param = 1 if is_w8 else 2

    # Embeddings and LM head are typically BF16 (2 bytes)
    weights_disk_bytes = (
        (embed_params * 2)
        + (layer_total * bytes_per_param)
        + (final_norm_params * 2)
        + (lm_head_params * 2)
    )

    return {
        "embedding_parameters": embed_params,
        "per_layer_attention_parameters": attn_per_layer,
        "per_layer_mlp_parameters": mlp_per_layer,
        "all_layers_parameters": layer_total,
        "lm_head_parameters": lm_head_params,
        "total_parameters": total_params,
        "total_parameters_millions": round(total_params / 1e6, 2),
        "estimated_weights_bytes": weights_disk_bytes,
        "estimated_weights_mib": round(weights_disk_bytes / (1024 * 1024), 2),
    }


def verify_model_files(model_dir: Path, manifest: dict, verify_checksums: bool = False) -> dict:
    """Check existence, sizes, and optional SHA-256 checksums of model files."""
    files_manifest = manifest.get("files", {})
    verified_files = []
    missing_files = []
    size_mismatches = []
    checksum_failures = []

    for name, expected in files_manifest.items():
        path = model_dir / name
        if not path.is_file():
            missing_files.append(name)
            continue

        actual_size = path.stat().st_size
        expected_size = expected.get("size")
        if expected_size is not None and actual_size != expected_size:
            size_mismatches.append({
                "file": name,
                "expected_size": expected_size,
                "actual_size": actual_size,
            })
            continue

        if verify_checksums and "sha256" in expected:
            hasher = hashlib.sha256()
            with path.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if digest != expected["sha256"]:
                checksum_failures.append({
                    "file": name,
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": digest,
                })
                continue

        verified_files.append(name)

    # Check tokenizer components
    tokenizer_present = (model_dir / "tokenizer.json").is_file()

    valid = (
        len(missing_files) == 0
        and len(size_mismatches) == 0
        and len(checksum_failures) == 0
        and tokenizer_present
    )

    return {
        "valid": valid,
        "total_manifest_files": len(files_manifest),
        "verified_files_count": len(verified_files),
        "missing_files": missing_files,
        "size_mismatches": size_mismatches,
        "checksum_failures": checksum_failures,
        "tokenizer_present": tokenizer_present,
    }


def inspect_model(model_dir: Path, verify_checksums: bool = False) -> dict:
    """Analyze a converted XDNA1 model directory and return complete diagnostic report."""
    model_dir = Path(model_dir)
    metadata_file = model_dir / "metadata.json"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")

    manifest = json.loads(metadata_file.read_text(encoding="utf-8"))
    params = compute_parameter_breakdown(manifest)
    file_check = verify_model_files(model_dir, manifest, verify_checksums=verify_checksums)

    # Hardware & Performance Estimates on XDNA1 (4 columns AIE2)
    # AIE2 core clock: ~1.0 - 1.3 GHz. 4 columns * 4 tiles = 16 compute tiles.
    context_length = int(manifest.get("context_length", 64))
    layers = int(manifest.get("layers", 30))
    hidden = int(manifest.get("hidden_size", 576))

    # Weight streaming per decode token (in bytes)
    weight_bytes = params["estimated_weights_bytes"]
    # At 75 tok/s, required streaming bandwidth in GB/s
    bandwidth_gb_at_75_tps = (weight_bytes * 75.0) / (1024**3)

    # KV Cache memory per token sequence (BF16 = 2 bytes per element)
    kv_heads = int(manifest.get("kv_heads", 3))
    head_dim = int(manifest.get("head_dim", 64))
    kv_cache_bytes = 2 * layers * 2 * kv_heads * head_dim * context_length * 2  # 2 for K and V, 2 bytes/BF16
    kv_cache_kib = kv_cache_bytes / 1024

    return {
        "model_id": manifest.get("model_id", model_dir.name),
        "model_dir": str(model_dir.resolve()),
        "model_family": manifest.get("model_family", "llama"),
        "format": manifest.get("format", "xdna1-w8a16"),
        "context_length": context_length,
        "architecture": {
            "layers": layers,
            "hidden_size": hidden,
            "intermediate_size": int(manifest.get("intermediate_size", 1536)),
            "attention_heads": int(manifest.get("attention_heads", 9)),
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "vocab_size": int(manifest.get("vocab_size", 49152)),
        },
        "parameter_metrics": params,
        "hardware_demands": {
            "kv_cache_kib": round(kv_cache_kib, 2),
            "weights_size_mib": params["estimated_weights_mib"],
            "streaming_bandwidth_gbps_at_75_tps": round(bandwidth_gb_at_75_tps, 2),
            "xdna1_supported": context_length <= 64 and (hidden % 64 == 0),
        },
        "integrity": file_check,
    }


def format_text(report: dict) -> str:
    """Format inspection report into terminal-friendly text."""
    arch = report["architecture"]
    params = report["parameter_metrics"]
    hw = report["hardware_demands"]
    integ = report["integrity"]

    status_str = "VALID [OK]" if integ["valid"] else "INVALID / CORRUPT [FAIL]"

    lines = [
        "==================================================",
        f" Model Inspection: {report['model_id']}",
        "==================================================",
        f"Directory:    {report['model_dir']}",
        f"Family:       {report['model_family']} ({report['format']})",
        f"Integrity:    {status_str}",
        "--------------------------------------------------",
        "Architecture Parameters:",
        f"  Layers:             {arch['layers']}",
        f"  Hidden Size:        {arch['hidden_size']}",
        f"  Intermediate Size:  {arch['intermediate_size']}",
        f"  Attention Heads:    {arch['attention_heads']} (Q) / {arch['kv_heads']} (KV)",
        f"  Head Dimension:     {arch['head_dim']}",
        f"  Context Length:     {report['context_length']} tokens",
        f"  Vocabulary:         {arch['vocab_size']} tokens",
        "--------------------------------------------------",
        "Parameters & Footprint:",
        f"  Total Parameters:   {params['total_parameters_millions']}M ({params['total_parameters']:,} params)",
        f"  Weight Footprint:   {params['estimated_weights_mib']} MiB",
        f"  KV Cache (64 ctx):  {hw['kv_cache_kib']} KiB",
        f"  BW @ 75 tok/s:      {hw['streaming_bandwidth_gbps_at_75_tps']} GB/s",
        f"  XDNA1 Compatible:   {'YES' if hw['xdna1_supported'] else 'NO'}",
        "--------------------------------------------------",
        f"Files Verified:       {integ['verified_files_count']}/{integ['total_manifest_files']}",
        f"Tokenizer:            {'Present' if integ['tokenizer_present'] else 'Missing'}",
    ]
    if integ["missing_files"]:
        lines.append(f"Missing Files:        {', '.join(integ['missing_files'])}")
    if integ["size_mismatches"]:
        lines.append(f"Size Mismatches:      {len(integ['size_mismatches'])} file(s)")
    if integ["checksum_failures"]:
        lines.append(f"Checksum Failures:    {len(integ['checksum_failures'])} file(s)")
    lines.append("==================================================")
    return "\n".join(lines)


def format_markdown(report: dict) -> str:
    """Format inspection report into a GitHub Markdown table."""
    arch = report["architecture"]
    params = report["parameter_metrics"]
    hw = report["hardware_demands"]
    integ = report["integrity"]

    status_badge = "✅ Passed" if integ["valid"] else "❌ Failed"

    lines = [
        f"## Model Specification: `{report['model_id']}`",
        "",
        f"**Status**: {status_badge} | **Format**: `{report['format']}` | **Family**: `{report['model_family']}`",
        "",
        "| Architecture Attribute | Specification |",
        "|:-----------------------|:--------------|",
        f"| **Layers** | `{arch['layers']}` |",
        f"| **Hidden / Intermediate** | `{arch['hidden_size']} / {arch['intermediate_size']}` |",
        f"| **Heads (Q / KV)** | `{arch['attention_heads']} / {arch['kv_heads']}` (`head_dim={arch['head_dim']}`) |",
        f"| **Context Window** | `{report['context_length']}` tokens |",
        f"| **Vocabulary Size** | `{arch['vocab_size']:,}` |",
        f"| **Total Parameters** | **`{params['total_parameters_millions']}M`** |",
        f"| **Weight Size** | `{params['estimated_weights_mib']} MiB` |",
        f"| **KV Cache Capacity** | `{hw['kv_cache_kib']} KiB` |",
        f"| **Streaming BW @ 75 tok/s** | `{hw['streaming_bandwidth_gbps_at_75_tps']} GB/s` |",
        f"| **XDNA1 Hardware Ready** | `{'Yes' if hw['xdna1_supported'] else 'No'}` |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate HawkPoint NPU model weights and architecture.")
    parser.add_argument("model_dir", type=Path, help="Directory containing converted model files and metadata.json")
    parser.add_argument("--verify-checksums", action="store_true", help="Perform full SHA-256 byte verification on weights")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text", help="Output format")
    parser.add_argument("-o", "--output", type=Path, help="Write output to file")
    args = parser.parse_args(argv)

    if not args.model_dir.is_dir():
        sys.stderr.write(f"Error: {args.model_dir} is not a valid directory\n")
        return 1

    try:
        report = inspect_model(args.model_dir, verify_checksums=args.verify_checksums)
    except Exception as exc:
        sys.stderr.write(f"Inspection error: {exc}\n")
        return 1

    if args.format == "json":
        output = json.dumps(report, indent=2)
    elif args.format == "markdown":
        output = format_markdown(report)
    else:
        output = format_text(report)

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Inspection report written to {args.output}")
    else:
        print(output)

    return 0 if report["integrity"]["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
