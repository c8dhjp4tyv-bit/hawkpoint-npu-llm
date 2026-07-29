"""Known checkpoints compatible with the implemented XDNA1 graphs."""

import json
from pathlib import Path


MODEL_PRESETS = {
    "smollm2-135m-xdna1": {
        "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "directory": "SmolLM2-135M-Instruct-xdna1-w8a16",
        "display_name": "SmolLM2 135M Instruct (XDNA1)",
    },
    "smollm-135m-xdna1": {
        "repo_id": "HuggingFaceTB/SmolLM-135M-Instruct",
        "directory": "SmolLM-135M-Instruct-xdna1-w8a16",
        "display_name": "SmolLM 135M Instruct (XDNA1)",
    },
    "smollm2-135m-sft-xdna1": {
        "repo_id": "HuggingFaceTB/smollm2-135M-SFT-Only",
        "directory": "SmolLM2-135M-SFT-Only-xdna1-w8a16",
        "display_name": "SmolLM2 135M SFT-Only (XDNA1)",
    },
    "qwen2.5-0.5b-xdna1": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "directory": "Qwen2.5-0.5B-Instruct-xdna1-w8a16",
        "display_name": "Qwen2.5 0.5B Instruct (XDNA1 experimental)",
    },
}

DEFAULT_MODEL_ID = "smollm2-135m-xdna1"


def discover_models(models_dir):
    """Return installed runtime models as model-id -> metadata/path records."""
    models_dir = Path(models_dir)
    discovered = {}
    if not models_dir.exists():
        return discovered

    directory_to_id = {
        preset["directory"]: model_id
        for model_id, preset in MODEL_PRESETS.items()
    }
    for metadata_path in sorted(models_dir.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        model_dir = metadata_path.parent
        model_id = metadata.get("model_id") or directory_to_id.get(model_dir.name)
        if not model_id:
            continue
        preset = MODEL_PRESETS.get(model_id, {})
        discovered[model_id] = {
            "path": model_dir,
            "display_name": metadata.get(
                "display_name",
                preset.get("display_name", model_id),
            ),
            "source_model": metadata.get(
                "source_model",
                preset.get("repo_id"),
            ),
            "context_length": int(metadata.get("context_length", 64)),
        }
    return discovered
