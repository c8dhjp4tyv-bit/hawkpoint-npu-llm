"""Known checkpoints compatible with the implemented XDNA1 graphs."""

import json
import logging
from pathlib import Path


MODEL_PRESETS = {
    "smollm2-135m-xdna1": {
        "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "directory": "SmolLM2-135M-Instruct-xdna1-w8a16",
        "display_name": "SmolLM2 135M Instruct (XDNA1)",
    },
    "smollm-135m-xdna1": {
        "repo_id": "HuggingFaceTB/SmolLM-135M-Instruct",
        "revision": "fcc320f490e08fdb4b99d935b2c58d40bf35b0d0",
        "directory": "SmolLM-135M-Instruct-xdna1-w8a16",
        "display_name": "SmolLM 135M Instruct (XDNA1)",
    },
    "smollm2-135m-sft-xdna1": {
        "repo_id": "HuggingFaceTB/smollm2-135M-SFT-Only",
        "revision": "79528469cd11749bac3e8e9200fd9c192fbd8979",
        "directory": "SmolLM2-135M-SFT-Only-xdna1-w8a16",
        "display_name": "SmolLM2 135M SFT-Only (XDNA1)",
    },
    "qwen2.5-0.5b-xdna1": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "directory": "Qwen2.5-0.5B-Instruct-xdna1-w8a16",
        "display_name": "Qwen2.5 0.5B Instruct (XDNA1 experimental)",
    },
}

DEFAULT_MODEL_ID = "smollm2-135m-xdna1"


# The only context length the compiled IRON graphs and kernels support.
SUPPORTED_CONTEXT_LENGTH = 64


def _text(value, fallback):
    """Accept a non-empty string, otherwise fall back."""
    return value if isinstance(value, str) and value else fallback


def _context_length(value):
    """Return the supported context length, or None if the record is invalid."""
    if value is None:
        return SUPPORTED_CONTEXT_LENGTH
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value != SUPPORTED_CONTEXT_LENGTH:
        return None
    return value


def discover_models(models_dir):
    """Return installed runtime models as model-id -> metadata/path records.

    A directory whose ``metadata.json`` is unreadable, is not a JSON object,
    or carries fields of the wrong type is skipped with a warning. One broken
    package must never keep the server from serving the other installed
    models, so nothing here raises.
    """
    models_dir = Path(models_dir)
    discovered = {}
    if not models_dir.exists():
        return discovered

    directory_to_id = {
        preset["directory"]: model_id
        for model_id, preset in MODEL_PRESETS.items()
    }
    for metadata_path in sorted(models_dir.glob("*/metadata.json")):
        model_dir = metadata_path.parent
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _skip(model_dir, f"metadata.json could not be read: {exc}")
            continue
        if not isinstance(metadata, dict):
            _skip(model_dir, "metadata.json is not a JSON object")
            continue
        model_id = _text(
            metadata.get("model_id"), directory_to_id.get(model_dir.name)
        )
        if not model_id:
            _skip(model_dir, "no usable model_id")
            continue
        context_length = _context_length(metadata.get("context_length"))
        if context_length is None:
            _skip(
                model_dir,
                "context_length must be the supported "
                f"{SUPPORTED_CONTEXT_LENGTH}, got "
                f"{metadata.get('context_length')!r}",
            )
            continue
        preset = MODEL_PRESETS.get(model_id, {})
        discovered[model_id] = {
            "path": model_dir,
            "display_name": _text(
                metadata.get("display_name"),
                preset.get("display_name", model_id),
            ),
            "source_model": _text(
                metadata.get("source_model"), preset.get("repo_id")
            ),
            "context_length": context_length,
        }
    return discovered


def _skip(model_dir, reason):
    logging.warning("skipping model directory %s: %s", model_dir, reason)
