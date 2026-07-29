#!/usr/bin/env python3
"""Download SmolLM2-135M-Instruct and convert it for the XDNA1 runtime."""

from pathlib import Path
import sys

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "npu_llm/models/SmolLM2-135M-Instruct"
OUTPUT = ROOT / "npu_llm/models/SmolLM2-135M-Instruct-xdna1-w8a16"


def main():
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HuggingFaceTB/SmolLM2-135M-Instruct to {SOURCE}")
    snapshot_download(
        repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        local_dir=SOURCE,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
        ],
    )

    sys.path.insert(0, str(ROOT / "npu_llm/tools"))
    from convert_smollm2 import convert

    print(f"Converting model to {OUTPUT}")
    manifest = convert(SOURCE, OUTPUT)
    print(
        "Ready: "
        f"{manifest['layers']} layers, "
        f"hidden={manifest['hidden_size']}, "
        f"context={manifest['context_length']}"
    )


if __name__ == "__main__":
    main()
