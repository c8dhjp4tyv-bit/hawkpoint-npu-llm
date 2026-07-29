#!/usr/bin/env python3
"""Build fixed per-model token references with the NumPy BF16 path."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "npu_llm"))

from model_catalog import discover_models  # noqa: E402
from runtime.generate import NPUDecoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    references = {}
    prompt = [{"role": "user", "content": "Reply with exactly OK."}]
    for model_id, record in discover_models(args.models_dir).items():
        decoder = NPUDecoder(record["path"], npu_layers=0)
        stats = None
        for _, candidate in decoder.generate_messages(prompt, args.tokens):
            if candidate is not None:
                stats = candidate
        if stats is None or not stats["generated_token_ids"]:
            raise RuntimeError(f"{model_id} produced no reference tokens")
        references[model_id] = stats["generated_token_ids"]
        del decoder
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(references, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(references, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
