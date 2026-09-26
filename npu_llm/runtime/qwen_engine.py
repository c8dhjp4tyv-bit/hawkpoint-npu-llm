"""Host side of the Qwen2.5 0.5B decoder engine (designs/qwen_engine.py)."""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

from designs.qwen_engine import qwen_engine
from designs.qwen_engine_layout import (
    CACHE_HEAD,
    CONTEXT,
    HEAD_DIM,
    HEADER_COS,
    HEADER_HIDDEN,
    HEADER_POSITION,
    HEADER_SIN,
    HIDDEN,
    KV_HEADS,
    KV_NEW,
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

    def __init__(self, model):
        if not supports(model.metadata):
            raise ValueError("the Qwen engine supports Qwen2.5 0.5B layouts only")
        self.layers = int(model.metadata["layers"])
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
        self.cache = iron.zeros(
            cache_elements(self.layers), dtype=bfloat16, device="npu"
        )
        self.kv_new = iron.zeros(self.layers * KV_NEW, dtype=bfloat16, device="npu")
        self.logits = iron.zeros(VOCAB, dtype=np.float32, device="npu")
        self._cache_host = self.cache.data
        self._layers_view = self._cache_host[CACHE_HEAD:].reshape(
            self.layers, KV_HEADS, 2, CONTEXT, HEAD_DIM
        )

    def reset(self):
        self._cache_host.fill(0)

    def _write_header(self, embedding, position):
        angle = np.float32(position) * self.rope_inv_frequency
        header = self._cache_host[:CACHE_HEAD]
        for base, values in ((HEADER_COS, np.cos(angle)), (HEADER_SIN, np.sin(angle))):
            for index, part in enumerate(split3(values)):
                header[base + index * 32:base + (index + 1) * 32] = part
        header[HEADER_POSITION] = bfloat16(position)
        header[HEADER_HIDDEN:HEADER_HIDDEN + HIDDEN] = embedding

    def step(self, embedding, position, logits=True):
        """Run every layer for one token and return its FP32 logits.

        With ``logits=False`` (prefill positions whose output is discarded) the
        dispatch stops after the last layer and returns None. Attention reads
        cached positions below ``position`` only.
        """
        if not 0 <= position < CONTEXT:
            raise ValueError("position outside the 64-token context")
        self._write_header(embedding, position)
        self.cache._sync_to_device()
        qwen_engine(
            self.weights, self.cache, self.kv_new, self.logits,
            layers=self.layers, lm_head=bool(logits),
        )
        new = self.kv_new.numpy().reshape(self.layers, KV_HEADS, 2, HEAD_DIM)
        self._layers_view[:, :, :, position, :] = new
        return np.array(self.logits.numpy(), copy=True) if logits else None

    def close(self):
        self.weights = None
        self.cache = None
        self.kv_new = None
        self.logits = None
        self._cache_host = None
        self._layers_view = None
