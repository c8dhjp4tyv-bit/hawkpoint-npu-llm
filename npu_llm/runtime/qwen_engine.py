"""Host side of the Qwen2.5 0.5B decoder engine (designs/qwen_engine.py)."""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

from context_window import CONTEXT_BUCKETS, KV_TILE, context_tiles
from designs.qwen_engine import qwen_engine
from designs.qwen_engine_layout import (
    CACHE_HEAD,
    HEAD_DIM,
    HEADER_COS,
    HEADER_HIDDEN,
    HEADER_POSITION,
    HEADER_SIN,
    HIDDEN,
    KV_HEADS,
    KV_NEW,
    TILE_ELEMENTS,
    VOCAB,
    cache_elements,
    pack_weights,
    supports,
)


def split3(values):
    """FP32 values as three BF16 parts whose sum carries every mantissa bit."""
    values = np.asarray(values, dtype=np.float32)
    hi = values.astype(bfloat16)
    rest = values - hi.astype(np.float32)
    mid = rest.astype(bfloat16)
    lo = (rest - mid.astype(np.float32)).astype(bfloat16)
    return hi, mid, lo


class QwenDecoderEngine:
    """Runs all 24 Qwen decoder layers, the final RMSNorm, and the LM head for
    one token per call."""

    def __init__(self, model, context_length):
        if not supports(model.metadata):
            raise ValueError("the Qwen engine supports Qwen2.5 0.5B layouts only")
        self.layers = int(model.metadata["layers"])
        self.context_length = int(context_length)
        self.max_tiles = context_tiles(self.context_length)
        self.tiles = 0
        theta = float(model.metadata.get("rope_theta", 10000.0))
        half = HEAD_DIM // 2
        # Same expression as runtime/cpu_backend.py, so RoPE angles match the
        # CPU reference bit for bit.
        self.rope_inv_frequency = 1.0 / (
            theta ** (np.arange(half, dtype=np.float32) / np.float32(half))
        )
        self.weights = iron.tensor(
            pack_weights(model, self.layers), dtype=bfloat16, device="npu"
        )
        self.kv_new = iron.zeros(self.layers * KV_NEW, dtype=bfloat16, device="npu")
        self.logits = iron.zeros(VOCAB, dtype=np.float32, device="npu")
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
        self._key_copies = self._pairs = self._cache_host = None
        self.cache = iron.zeros(
            cache_elements(self.layers, tiles * KV_TILE),
            dtype=bfloat16,
            device="npu",
        )
        self.tiles = tiles
        self._cache_host = self.cache.data
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
        self._cache_host.fill(0)

    def _write_header(self, embedding, position):
        angle = np.float32(position) * self.rope_inv_frequency
        header = self._cache_host[:CACHE_HEAD]
        for base, values in ((HEADER_COS, np.cos(angle)), (HEADER_SIN, np.sin(angle))):
            for index, part in enumerate(split3(values)):
                header[base + index * 32:base + (index + 1) * 32] = part
        # Two BF16 parts keep the position exact past 255 (see the kernels).
        high = bfloat16(position)
        header[HEADER_POSITION] = high
        header[HEADER_POSITION + 1] = bfloat16(position - float(high))
        header[HEADER_HIDDEN:HEADER_HIDDEN + HIDDEN] = embedding

    def step(self, embedding, position, logits=True):
        """Run every layer for one token and return its FP32 logits.

        With ``logits=False`` (prefill positions whose output is discarded) the
        dispatch stops after the last layer and returns None. Attention reads
        cached positions below ``position`` only.
        """
        if not 0 <= position < self.context_length:
            raise ValueError(
                f"position outside the {self.context_length}-token context"
            )
        self._ensure_tiles(self._tiles_for(position))
        self._write_header(embedding, position)
        self.cache._sync_to_device()
        qwen_engine(
            self.weights, self.cache, self.kv_new, self.logits,
            layers=self.layers, lm_head=bool(logits),
            tiles=self.tiles,
        )
        new = self.kv_new.numpy().reshape(self.layers, KV_HEADS, 2, HEAD_DIM)
        tile, row = divmod(position, KV_TILE)
        for view in self._key_copies:
            view[:, :, tile, row, :] = new[:, :, 0]
        self._pairs[:, :, 2 * tile, row, :] = new[:, :, 0]
        self._pairs[:, :, 2 * tile + 1, row, :] = new[:, :, 1]
        return np.array(self.logits.numpy(), copy=True) if logits else None

    def close(self):
        self.weights = None
        self.cache = None
        self.kv_new = None
        self.logits = None
        self._cache_host = None
        self._key_copies = None
        self._pairs = None
