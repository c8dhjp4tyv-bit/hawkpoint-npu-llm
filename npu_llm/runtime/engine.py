"""Host side of the row-split SmolLM decoder engine (designs/engine.py).

The engine runs every decoder layer of one token in a single NPU dispatch.
This module owns the packed weights (designs/engine_layout.py), the K/V cache
buffer the hub tile streams, and appends each token's new keys and values after
the dispatch.
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

from context_window import CONTEXT_BUCKETS, KV_TILE, context_tiles
from designs.engine import decoder_engine
from designs.engine_layout import (
    CACHE_HEAD,
    HEAD_DIM,
    HEADER_HIDDEN,
    HIDDEN,
    KV_HEADS,
    KV_NEW,
    TILE_ELEMENTS,
    cache_elements,
    pack_weights,
    supports,
)


class DecoderEngine:
    """Runs all decoder layers plus the final RMSNorm for one token per call."""

    def __init__(self, model, rope_inv_freq, context_length):
        if not supports(model.metadata):
            raise ValueError("the decoder engine supports SmolLM 135M layouts only")
        self.layers = int(model.metadata["layers"])
        self.context_length = int(context_length)
        self.max_tiles = context_tiles(self.context_length)
        self.tiles = 0
        self.rope_inv_freq = np.asarray(rope_inv_freq, dtype=np.float32)
        self.weights = iron.tensor(
            pack_weights(model, self.layers), dtype=bfloat16, device="npu"
        )
        self.kv_new = iron.zeros(self.layers * KV_NEW, dtype=bfloat16, device="npu")
        self.hidden = iron.zeros(HIDDEN, dtype=bfloat16, device="npu")
        self.cache = None
        self._cache_host = None
        self._key_copies = None
        self._pairs = None
        self._ensure_tiles(CONTEXT_BUCKETS[0])

    def _tiles_for(self, position):
        """Smallest streaming bucket that covers ``position``."""
        needed = max(1, -(-(position + 1) // KV_TILE))
        for bucket in CONTEXT_BUCKETS:
            if bucket >= needed and bucket <= self.max_tiles:
                return bucket
        return self.max_tiles

    def _ensure_tiles(self, tiles):
        """Grow the streamed window, keeping the positions already cached.

        The graph streams the cache as one contiguous run, so the buffer has to
        be allocated for the window it is dispatched with; growing copies the
        rows written so far into the wider layout.
        """
        if tiles == self.tiles:
            return
        # Copy the cached rows out before the reallocation: the views below
        # point into the buffer that is about to be replaced.
        old_copies = None if self._key_copies is None else [
            np.array(view) for view in self._key_copies
        ]
        old_pairs = None if self._pairs is None else np.array(self._pairs)
        # Allocate first: if the NPU does not have room for the wider window the
        # decoder is left exactly as it was, with its old views and tile count.
        cache = iron.zeros(
            cache_elements(self.layers, tiles * KV_TILE),
            dtype=bfloat16,
            device="npu",
        )
        self._key_copies = self._pairs = self._cache_host = None
        self.cache = cache
        self.tiles = tiles
        self._cache_host = cache.data
        # Host view of the cache stream: the header tile, then per layer and
        # KV head the two key passes followed by the key/value pairs, in the
        # order the hub consumes them.
        objects = KV_HEADS * self.layers * tiles
        region = self._cache_host[
            CACHE_HEAD:CACHE_HEAD + 4 * objects * TILE_ELEMENTS
        ].reshape((self.layers, KV_HEADS, 4 * tiles, KV_TILE, HEAD_DIM))
        self._key_copies = [
            region[:, :, copy * tiles:(copy + 1) * tiles] for copy in range(2)
        ]
        self._pairs = region[:, :, 2 * tiles:4 * tiles]
        if old_copies is not None:
            for old, new in zip(old_copies, self._key_copies):
                kept = min(old.shape[2], new.shape[2])
                new[:, :, :kept] = old[:, :, :kept]
            kept = min(old_pairs.shape[2], self._pairs.shape[2])
            self._pairs[:, :, :kept] = old_pairs[:, :, :kept]

    def reset(self):
        """Forget every cached position (a new sequence starts at 0)."""
        self._cache_host.fill(0)

    def _write_header(self, embedding, position):
        # Same RoPE table layout as NPUDecoder._rope_lut: cos, sin, position.
        angle = np.float32(position) * self.rope_inv_freq
        header = self._cache_host[:CACHE_HEAD]
        header[:32] = np.cos(angle).astype(bfloat16)
        header[32:64] = np.sin(angle).astype(bfloat16)
        # Two BF16 parts keep the position exact past 255 (see the kernels).
        high = bfloat16(position)
        header[64] = high
        header[65] = bfloat16(position - float(high))
        header[HEADER_HIDDEN:HEADER_HIDDEN + HIDDEN] = embedding

    def step(self, embedding, position):
        """Run every layer for one token; return the final-RMSNorm output (BF16).

        Attention reads cached positions below ``position`` only, so a cache
        written by an earlier sequence never needs clearing.
        """
        if not 0 <= position < self.context_length:
            raise ValueError(
                f"position outside the {self.context_length}-token context"
            )
        self._ensure_tiles(self._tiles_for(position))
        self._write_header(embedding, position)
        # The header and the rows appended after the previous step were
        # written through the host mapping; flush them before the dispatch.
        self.cache._sync_to_device()
        decoder_engine(
            self.weights,
            self.cache,
            self.kv_new,
            self.hidden,
            layers=self.layers,
            tiles=self.tiles,
        )
        new = self.kv_new.numpy().reshape(self.layers, KV_HEADS, 2, HEAD_DIM)
        tile, row = divmod(position, KV_TILE)
        for view in self._key_copies:
            view[:, :, tile, row, :] = new[:, :, 0]
        self._pairs[:, :, 2 * tile, row, :] = new[:, :, 0]
        self._pairs[:, :, 2 * tile + 1, row, :] = new[:, :, 1]
        return self.hidden.numpy()

    def close(self):
        """Drop every XRT buffer so the owning decoder can release the NPU."""
        self.weights = None
        self.cache = None
        self.kv_new = None
        self.hidden = None
        self._cache_host = None
        self._key_copies = None
        self._pairs = None
