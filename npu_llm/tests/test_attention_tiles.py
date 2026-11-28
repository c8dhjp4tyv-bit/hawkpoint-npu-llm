#!/usr/bin/env python3
"""Hardware-free checks for the tiled attention decomposition.

The hub kernels stream the K/V cache in ``KV_TILE`` position tiles and score a
token in three passes (maximum, exponential sums, weighted values) instead of
holding the whole history in data memory. This file mirrors that arithmetic in
NumPy and requires it to match the untiled computation exactly, including the
BF16 weight rounding and the 16-position lane grouping: any change to the
padding rule or the accumulation order that would move the release-gate tokens
fails here.

It is a structural check of the decomposition, not of the kernels: both sides
use NumPy's exponential and division here, while the AIE2 hub uses the
lookup-table versions in ``kernels/fp32_emulation.h`` and, for Qwen, a
different rounding sequence and three-part weights. Only
``tests/validate_engine_npu.py`` and ``tests/validate_qwen_tokens.py`` compare
real kernel output against the pinned token references.
"""

from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from context_window import KV_TILE, context_tiles  # noqa: E402


def _bf16(values):
    return np.asarray(values, dtype=bfloat16).astype(np.float32)


def _weighted_scores(scores, count, maximum):
    """exp(bf16(score - maximum) * 0.125), rounded to BF16 per element."""
    offset = _bf16(scores - np.float32(maximum))
    scaled = _bf16(offset * np.float32(0.125))
    return _bf16(np.exp(scaled.astype(np.float32)))


def untiled(scores, count):
    """The single-array softmax the kernels ran before tiling."""
    vectors = (count + 15) // 16
    padded = np.zeros(vectors * 16, dtype=np.float32)
    padded[:count] = scores
    padded[count:] = padded[0]
    maximum = np.float32(padded.max())
    weights = _weighted_scores(padded[:vectors * 16], vectors * 16, maximum)
    weights[count:vectors * 16] = 0.0
    lanes = np.zeros(16, dtype=np.float32)
    for vector in range(vectors):
        lanes = (lanes + weights[vector * 16:(vector + 1) * 16]).astype(np.float32)
    total = np.float32(lanes.sum(dtype=np.float32))
    inverse = _bf16(np.float32(1.0) / total)
    return maximum, total, _bf16(weights[:count] * inverse)


def tiled(scores, count, tile=KV_TILE):
    """The three-pass, streamed-tile softmax the hub kernels run."""
    maximum = np.float32(-3.4e38)
    for first in range(0, count, tile):
        end = min(first + tile, count)
        padded = np.zeros(tile, dtype=np.float32)
        padded[:end - first] = scores[first:end]
        padded[end - first:] = padded[0]
        maximum = max(maximum, np.float32(padded.max()))
    lanes = np.zeros(16, dtype=np.float32)
    weights = np.zeros(count, dtype=np.float32)
    for first in range(0, count, tile):
        end = min(first + tile, count)
        padded = np.zeros(tile, dtype=np.float32)
        padded[:end - first] = scores[first:end]
        padded[end - first:] = padded[0]
        exponentials = _weighted_scores(padded, tile, maximum)
        exponentials[end - first:tile] = 0.0
        for vector in range((end - first + 15) // 16):
            lanes = (
                lanes + exponentials[vector * 16:(vector + 1) * 16]
            ).astype(np.float32)
        weights[first:end] = exponentials[:end - first]
    total = np.float32(lanes.sum(dtype=np.float32))
    inverse = _bf16(np.float32(1.0) / total)
    return maximum, total, _bf16(weights * inverse)


def test_tiled_softmax_matches_untiled():
    rng = np.random.default_rng(0)
    for count in (1, 7, 16, 17, 64, 65, KV_TILE, KV_TILE + 1, 3 * KV_TILE + 37):
        scores = rng.standard_normal(count).astype(np.float32) * 4.0
        expected = untiled(scores, count)
        actual = tiled(scores, count)
        assert actual[0] == expected[0], count
        assert actual[1] == expected[1], count
        assert np.array_equal(actual[2], expected[2]), count


def _softmax_decomposition(scores, tile=KV_TILE, streamed=True):
    """Softmax over ``position`` cached scores plus the current token.

    The kernels never put the current token in the stream: the cached
    positions arrive as tiles, while the current token's score is added to its
    lane (``position % 16``) after the last tile. ``streamed=False`` computes
    the same thing as one flat array, which is what the code did before the
    cache was tiled.
    """
    position = len(scores) - 1
    cached, current = scores[:position], np.float32(scores[position])
    if streamed:
        maximum = np.float32(-3.4e38)
        for first in range(0, position, tile):
            end = min(first + tile, position)
            padded = np.zeros(tile, dtype=np.float32)
            padded[:end - first] = cached[first:end]
            padded[end - first:] = padded[0]
            maximum = max(maximum, np.float32(padded.max()))
        maximum = max(maximum, current)
        lanes = np.zeros(16, dtype=np.float32)
        weights = np.zeros(position, dtype=np.float32)
        for first in range(0, position, tile):
            end = min(first + tile, position)
            padded = np.zeros(tile, dtype=np.float32)
            padded[:end - first] = cached[first:end]
            padded[end - first:] = padded[0]
            exponentials = _weighted_scores(padded, tile, maximum)
            exponentials[end - first:tile] = 0.0
            for vector in range((end - first + 15) // 16):
                lanes = (
                    lanes + exponentials[vector * 16:(vector + 1) * 16]
                ).astype(np.float32)
            weights[first:end] = exponentials[:end - first]
        own = _weighted_scores(np.array([current], dtype=np.float32), 1, maximum)
        lane = np.zeros(16, dtype=np.float32)
        lane[position % 16] = np.float32(own[0])
        lanes = (lanes + lane).astype(np.float32)
    else:
        flat = np.zeros((position // 16 + 1) * 16, dtype=np.float32)
        flat[:position] = cached
        flat[position] = current
        # Padding repeats the first score, as the kernels' last vector does, so
        # an all-negative tail cannot change the maximum.
        flat[position + 1:] = flat[0]
        maximum = np.float32(flat.max())
        exponentials = _weighted_scores(flat, flat.size, maximum)
        exponentials[position + 1:] = 0.0
        lanes = np.zeros(16, dtype=np.float32)
        for vector in range(flat.size // 16):
            lanes = (
                lanes + exponentials[vector * 16:(vector + 1) * 16]
            ).astype(np.float32)
        weights = exponentials[:position]
        own = exponentials[position:position + 1]
    total = np.float32(lanes.sum(dtype=np.float32))
    inverse = _bf16(np.float32(1.0) / total)
    return maximum, total, _bf16(weights * inverse), _bf16(own * inverse)


def test_current_token_joins_its_lane():
    rng = np.random.default_rng(7)
    for position in (0, 1, 15, 16, 17, KV_TILE - 1, KV_TILE, KV_TILE + 1, 3 * KV_TILE):
        scores = rng.standard_normal(position + 1).astype(np.float32) * 4.0
        if position == 0:
            scores = np.array([1.5], dtype=np.float32)
        streamed = _softmax_decomposition(scores, streamed=True)
        flat = _softmax_decomposition(scores, streamed=False)
        assert streamed[0] == flat[0], position
        assert streamed[1] == flat[1], position
        assert np.array_equal(streamed[2], flat[2]), position
        assert np.array_equal(streamed[3], flat[3]), position


def test_tile_count_bounds_the_stream():
    for context, expected in ((1, 1), (64, 1), (128, 1), (129, 2), (8192, 64)):
        assert context_tiles(context) == expected
    assert context_tiles(8192) * KV_TILE == 8192


def _cache_position(position):
    return divmod(position, KV_TILE)


def test_tile_position_matches_flat_row():
    for position in (0, 1, KV_TILE - 1, KV_TILE, KV_TILE + 3, 8191):
        tile, row = _cache_position(position)
        assert tile * KV_TILE + row == position


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
