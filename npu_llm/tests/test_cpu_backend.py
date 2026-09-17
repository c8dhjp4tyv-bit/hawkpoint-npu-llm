#!/usr/bin/env python3
"""Numerical checks for the vectorized CPU attention fallback."""

from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.cpu_backend import CPUDecoderStage  # noqa: E402


def test_vectorized_attention_matches_reference():
    stage = object.__new__(CPUDecoderStage)
    stage.q_heads = 4
    stage.kv_heads = 2
    stage.q_per_kv = 2
    stage.head_dim = 4
    stage._rope_inv_frequency = np.array([1.0, 0.5], dtype=np.float32)
    stage.key_cache = {0: np.zeros((2, 5, 4), dtype=bfloat16)}
    stage.value_cache = {0: np.zeros((2, 5, 4), dtype=bfloat16)}

    rng = np.random.default_rng(42)
    qkv = rng.normal(size=32).astype(np.float32).astype(bfloat16)
    position = 3
    actual = np.asarray(stage._attention(qkv, 0, position), dtype=np.float32)

    q_size = stage.q_heads * stage.head_dim
    kv_size = stage.kv_heads * stage.head_dim
    query = stage._rotate(
        qkv[:q_size].reshape(stage.q_heads, stage.head_dim), position
    )
    scale = stage.head_dim**-0.5
    expected = np.empty((stage.q_heads, stage.head_dim), dtype=np.float32)
    for kv_head in range(stage.kv_heads):
        keys = np.asarray(stage.key_cache[0][kv_head, : position + 1], dtype=np.float32)
        values = np.asarray(
            stage.value_cache[0][kv_head, : position + 1], dtype=np.float32
        )
        for group_head in range(stage.q_per_kv):
            q_head = kv_head * stage.q_per_kv + group_head
            scores = keys @ np.asarray(query[q_head], dtype=np.float32) * scale
            scores -= np.max(scores)
            probabilities = np.exp(scores)
            probabilities /= np.sum(probabilities)
            expected[q_head] = probabilities @ values

    assert np.allclose(actual, expected.reshape(-1), atol=2e-2, rtol=2e-2)


def main():
    test_vectorized_attention_matches_reference()
    print("PASS CPU backend tests")


if __name__ == "__main__":
    main()
