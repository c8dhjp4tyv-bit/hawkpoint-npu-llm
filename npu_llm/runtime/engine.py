"""Host side of the row-split SmolLM decoder engine (designs/engine.py).

The engine runs every decoder layer of one token in a single NPU dispatch.
This module owns the packed weights (designs/engine_layout.py), the K/V cache
buffer the hub tile reads, and appends each token's new keys and values after
the dispatch.
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

from designs.engine import decoder_engine
from designs.engine_layout import (
    CACHE_HEAD,
    CONTEXT,
    HEAD_DIM,
    HEADER_HIDDEN,
    HIDDEN,
    KV_HEADS,
    KV_NEW,
    cache_elements,
    pack_weights,
    supports,
)


class DecoderEngine:
    """Runs all decoder layers plus the final RMSNorm for one token per call."""

    def __init__(self, model, rope_inv_freq):
        if not supports(model.metadata):
            raise ValueError("the decoder engine supports SmolLM 135M layouts only")
        self.layers = int(model.metadata["layers"])
        self.rope_inv_freq = np.asarray(rope_inv_freq, dtype=np.float32)
        self.weights = iron.tensor(
            pack_weights(model, self.layers), dtype=bfloat16, device="npu"
        )
        self.cache = iron.zeros(
            cache_elements(self.layers), dtype=bfloat16, device="npu"
        )
        self.kv_new = iron.zeros(self.layers * KV_NEW, dtype=bfloat16, device="npu")
        self.hidden = iron.zeros(HIDDEN, dtype=bfloat16, device="npu")
        # Host view of the cache: [header][layer][kv head][K 64x64 | V 64x64].
        self._cache_host = self.cache.data
        self._layers_view = self._cache_host[CACHE_HEAD:].reshape(
            self.layers, KV_HEADS, 2, CONTEXT, HEAD_DIM
        )

    def reset(self):
        """Forget every cached position (a new sequence starts at 0)."""
        self._cache_host.fill(0)

    def _write_header(self, embedding, position):
        # Same RoPE table layout as NPUDecoder._rope_lut: cos, sin, position.
        angle = np.float32(position) * self.rope_inv_freq
        header = self._cache_host[:CACHE_HEAD]
        header[:32] = np.cos(angle).astype(bfloat16)
        header[32:64] = np.sin(angle).astype(bfloat16)
        header[64] = bfloat16(position)
        header[65] = bfloat16(0.0)
        header[HEADER_HIDDEN:HEADER_HIDDEN + HIDDEN] = embedding

    def step(self, embedding, position):
        """Run every layer for one token; return the final-RMSNorm output (BF16).

        Attention reads cached positions below ``position`` only, so a cache
        written by an earlier sequence never needs clearing.
        """
        if not 0 <= position < CONTEXT:
            raise ValueError("position outside the 64-token context")
        self._write_header(embedding, position)
        # The header and the rows appended after the previous step were
        # written through the host mapping; flush them before the dispatch.
        self.cache._sync_to_device()
        decoder_engine(
            self.weights, self.cache, self.kv_new, self.hidden, layers=self.layers
        )
        new = self.kv_new.numpy().reshape(self.layers, KV_HEADS, 2, HEAD_DIM)
        self._layers_view[:, :, :, position, :] = new
        return self.hidden.numpy()

    def close(self):
        """Drop every XRT buffer so the owning decoder can release the NPU."""
        self.weights = None
        self.cache = None
        self.kv_new = None
        self.hidden = None
        self._cache_host = None
        self._layers_view = None
