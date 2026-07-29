#!/usr/bin/env python3
"""Download and convert supported language models for XDNA1."""

import argparse
from pathlib import Path
import sys

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu_llm.model_catalog import DEFAULT_MODEL_ID, MODEL_PRESETS


def prepare(model_id, models_dir):
    preset = MODEL_PRESETS[model_id]
    repo_id = preset["repo_id"]
    models_dir = Path(models_dir)
    source = models_dir / "sources" / repo_id.rsplit("/", 1)[1]
    output = models_dir / preset["directory"]
    source.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {source}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=source,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
        ],
    )

    sys.path.insert(0, str(ROOT / "npu_llm/tools"))
    from convert_smollm2 import convert

    print(f"Converting model to {output}")
    manifest = convert(
        source,
        output,
        model_id=model_id,
        source_model=repo_id,
        display_name=preset["display_name"],
    )
    print(
        f"Ready: {model_id}, "
        f"{manifest['layers']} layers, "
        f"hidden={manifest['hidden_size']}, "
        f"context={manifest['context_length']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODEL_PRESETS),
        help="model to prepare; may be repeated",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="download and convert all supported models",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "npu_llm/models",
        help="source and converted model storage directory",
    )
    args = parser.parse_args()
    model_ids = (
        list(MODEL_PRESETS)
        if args.all
        else (args.model or [DEFAULT_MODEL_ID])
    )
    for model_id in model_ids:
        prepare(model_id, args.models_dir)


if __name__ == "__main__":
    main()
