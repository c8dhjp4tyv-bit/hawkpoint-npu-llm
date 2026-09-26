#!/usr/bin/env python3
"""Hardware-free tests for the Qwen decoder engine's weight-stream layout.

The kernels in kernels/qwen_engine_bf16.cc assume a fixed block order per
GEMV tile, uneven row ranges per tile, and a down projection whose columns
follow the joined-slice layout. These tests rebuild every tensor from the
packed streams of a small random model and require an exact match.
"""

from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designs import qwen_engine_layout as L  # noqa: E402


LAYERS = 1
VOCAB_ROWS = L.VOCAB


METADATA = {
    "hidden_size": 896,
    "intermediate_size": 4864,
    "attention_heads": 14,
    "kv_heads": 2,
    "head_dim": 64,
    "context_length": 64,
    "model_family": "qwen2",
    "vocab_size": L.VOCAB,
    "layers": LAYERS,
}


class FakeModel:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.metadata = dict(METADATA)
        shapes = {
            "qkv": (1152, 896),
            "o_proj": (896, 896),
            "gate_up": (9728, 896),
            "down_proj": (896, 4864),
        }
        # Random BF16 bit patterns below 1.0, generated directly as 16-bit
        # integers: the 151936-row head would need 1 GB as float64.
        self.projections = {
            "lm_head": rng.integers(
                0, 0x3F80, size=(VOCAB_ROWS, 896), dtype=np.uint16
            ).view(bfloat16)
        }
        self.raws = {"final_norm": rng.standard_normal(896).astype(np.float32)}
        for layer in range(LAYERS):
            prefix = f"layer{layer:02d}"
            for name, shape in shapes.items():
                self.projections[f"{prefix}.{name}"] = rng.standard_normal(
                    shape
                ).astype(bfloat16)
            for name in ("input_norm", "post_attn_norm"):
                self.raws[f"{prefix}.{name}"] = rng.standard_normal(896).astype(
                    np.float32
                )
            self.raws[f"{prefix}.qkv_bias"] = rng.standard_normal(1152).astype(
                np.float32
            )

    def bf16_projection(self, name):
        return self.projections[name]

    def raw(self, name):
        return self.raws[name]


def same(a, b):
    return np.array_equal(
        np.asarray(a).astype(bfloat16).view(np.uint16),
        np.asarray(b).astype(bfloat16).view(np.uint16),
    )


def test_every_tensor_round_trips():
    model = FakeModel()
    packed = L.pack_weights(model, LAYERS)
    offsets, total = L.tile_offsets(LAYERS)
    assert packed.size == total
    qkv = np.empty((1152, 896), dtype=bfloat16)
    o_proj = np.empty((896, 896), dtype=bfloat16)
    gate = np.empty((4864, 896), dtype=bfloat16)
    up = np.empty((4864, 896), dtype=bfloat16)
    down = np.zeros((896, L.VECTOR), dtype=bfloat16)
    lm_head = np.empty((VOCAB_ROWS, 896), dtype=bfloat16)
    bias = np.empty(1152, dtype=np.float32)
    for tile in range(L.GEMV_TILES):
        blocks = packed[offsets[tile]:offsets[tile] + L.tile_blocks(tile, LAYERS) * L.BLOCK]
        blocks = blocks.reshape(-1, L.BLOCK)
        params = blocks[0]
        assert same(params[:896], model.raws["layer00.input_norm"])
        assert same(params[896:1792], model.raws["layer00.post_attn_norm"])
        bias[tile * 192:(tile + 1) * 192] = params[1792:1792 + 384].view(np.float32)
        cursor = 1
        qkv[tile * 192:(tile + 1) * 192] = blocks[cursor:cursor + 48].reshape(-1, 896)
        cursor += 48
        rows = L.OUT_ROWS[tile]
        start = L.OUT_START[tile]
        o_proj[start:start + rows] = blocks[cursor:cursor + rows // 4].reshape(-1, 896)
        cursor += rows // 4
        first = L.MLP_START[tile]
        for group in range(L.MLP_ROWS[tile] // 16):
            gate[first + group * 16:first + group * 16 + 16] = blocks[cursor:cursor + 4].reshape(16, 896)
            up[first + group * 16:first + group * 16 + 16] = blocks[cursor + 4:cursor + 8].reshape(16, 896)
            cursor += 8
        for row in range(0, rows, 4):
            for chunk in range(L.DOWN_CHUNKS):
                down[start + row:start + row + 4, chunk * 896:(chunk + 1) * 896] = (
                    blocks[cursor].reshape(4, 896)
                )
                cursor += 1
        assert cursor == L.layer_blocks(tile)
        assert same(blocks[cursor][:896], model.raws["final_norm"])
        cursor += 1
        lm_rows = L.LM_GROUPS[tile] * L.LM_GROUP
        lm_head[L.LM_START[tile]:L.LM_START[tile] + lm_rows] = blocks[cursor:].reshape(-1, 896)
    assert same(qkv, model.projections["layer00.qkv"])
    assert same(o_proj, model.projections["layer00.o_proj"])
    gate_up = model.projections["layer00.gate_up"]
    assert same(gate, gate_up[:4864]) and same(up, gate_up[4864:])
    assert np.array_equal(bias, model.raws["layer00.qkv_bias"])
    assert same(lm_head, model.projections["lm_head"])
    # Down columns follow the joined slices: tile t's activation rows sit at
    # slot t * 896; the padding columns between slices are zero.
    expected = L.down_padded(model.projections["layer00.down_proj"])
    assert same(down, expected)
    for tile in range(L.GEMV_TILES):
        pad = expected[:, tile * 896 + L.MLP_ROWS[tile]:(tile + 1) * 896]
        assert not np.any(pad.astype(np.float32))


def test_row_ranges_cover_every_row_once():
    assert list(L.OUT_START) + [L.HIDDEN] == list(np.cumsum((0,) + L.OUT_ROWS))
    assert list(L.MLP_START) + [L.INTERMEDIATE] == list(np.cumsum((0,) + L.MLP_ROWS))
    assert L.LM_START[-1] + L.LM_GROUPS[-1] * L.LM_GROUP == L.VOCAB
    # q_residual in kernels/qwen_engine_bf16.cc hard-codes these tables.
    source = (ROOT / "kernels/qwen_engine_common.h").read_text()
    assert "q_out_rows[6] = {" + ", ".join(map(str, L.OUT_ROWS)) + "}" in source
    assert "q_out_start[6] = {" + ", ".join(map(str, L.OUT_START)) + "}" in source


def test_supported_architectures():
    assert L.supports(METADATA)
    smollm = dict(METADATA, hidden_size=576, model_family="llama")
    assert not L.supports(smollm)


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"PASS {len(tests)} Qwen engine layout tests")


if __name__ == "__main__":
    main()
