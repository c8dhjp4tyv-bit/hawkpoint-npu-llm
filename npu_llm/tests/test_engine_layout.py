#!/usr/bin/env python3
"""Hardware-free tests for the decoder engine's weight-stream layout.

The engine kernels assume a fixed order of weight blocks per GEMV tile. These
tests rebuild every projection from the packed streams of a small random model
and require an exact match, so a packing change that the kernels would read
wrongly fails here instead of producing wrong tokens on an NPU.
"""

from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designs import engine_layout as L  # noqa: E402


LAYERS = 2


class FakeModel:
    """Random BF16 tensors with the SmolLM 135M shapes."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.metadata = {
            "hidden_size": 576,
            "intermediate_size": 1536,
            "attention_heads": 9,
            "kv_heads": 3,
            "head_dim": 64,
            "context_length": 64,
            "layers": LAYERS,
        }
        shapes = {
            "qkv": (960, 576),
            "o_proj": (576, 576),
            "gate_up": (3072, 576),
            "down_proj": (576, 1536),
        }
        self.projections = {}
        self.raws = {"final_norm": rng.standard_normal(576).astype(np.float32)}
        for layer in range(LAYERS):
            prefix = f"layer{layer:02d}"
            for name, shape in shapes.items():
                self.projections[f"{prefix}.{name}"] = rng.standard_normal(
                    shape
                ).astype(bfloat16)
            for name in ("input_norm", "post_attn_norm"):
                self.raws[f"{prefix}.{name}"] = rng.standard_normal(576).astype(
                    np.float32
                )

    def bf16_projection(self, name):
        return self.projections[name]

    def raw(self, name):
        return self.raws[name]


def unpack(packed):
    """Invert the tile stream layout back into full projections per layer."""
    view = packed.reshape(L.GEMV_TILES, L.tile_blocks(LAYERS), L.BLOCK)
    layers = []
    for layer in range(LAYERS):
        qkv = np.empty((960, 576), dtype=bfloat16)
        o_proj = np.empty((576, 576), dtype=bfloat16)
        gate = np.empty((1536, 576), dtype=bfloat16)
        up = np.empty((1536, 576), dtype=bfloat16)
        down = np.empty((576, 1536), dtype=bfloat16)
        gammas = []
        for tile in range(L.GEMV_TILES):
            blocks = view[tile, layer * L.BLOCKS_PER_LAYER:(layer + 1) * L.BLOCKS_PER_LAYER]
            gammas.append(blocks[0])
            cursor = 1
            rows = blocks[cursor:cursor + L.QKV_BLOCKS].reshape(-1, 576)
            qkv[tile * L.QKV_ROWS:(tile + 1) * L.QKV_ROWS] = rows
            cursor += L.QKV_BLOCKS
            rows = blocks[cursor:cursor + L.O_BLOCKS].reshape(-1, 576)
            o_proj[tile * L.OUT_ROWS:(tile + 1) * L.OUT_ROWS] = rows
            cursor += L.O_BLOCKS
            for group in range(L.MLP_GROUPS):
                first = tile * L.MLP_ROWS + group * 16
                gate[first:first + 16] = blocks[cursor:cursor + 2].reshape(16, 576)
                up[first:first + 16] = blocks[cursor + 2:cursor + 4].reshape(16, 576)
                cursor += 4
            rows = blocks[cursor:cursor + L.DOWN_BLOCKS].reshape(-1, 1536)
            down[tile * L.OUT_ROWS:(tile + 1) * L.OUT_ROWS] = rows
            cursor += L.DOWN_BLOCKS
            assert cursor == L.BLOCKS_PER_LAYER
        layers.append((qkv, o_proj, np.concatenate([gate, up]), down, gammas))
    return layers, view[:, -1]


def test_every_projection_round_trips():
    model = FakeModel()
    packed = L.pack_weights(model, LAYERS)
    assert packed.size == L.weight_elements(LAYERS)
    layers, final_blocks = unpack(packed)
    for layer, (qkv, o_proj, gate_up, down, gammas) in enumerate(layers):
        prefix = f"layer{layer:02d}"
        for name, rebuilt in (
            ("qkv", qkv), ("o_proj", o_proj), ("gate_up", gate_up), ("down_proj", down)
        ):
            assert np.array_equal(
                rebuilt.view(np.uint16),
                model.projections[f"{prefix}.{name}"].view(np.uint16),
            ), f"{prefix}.{name} does not round-trip"
        expected_input = model.raws[f"{prefix}.input_norm"].astype(bfloat16)
        expected_post = model.raws[f"{prefix}.post_attn_norm"].astype(bfloat16)
        for block in gammas:
            assert np.array_equal(block[:576], expected_input)
            assert np.array_equal(block[576:1152], expected_post)
    final = model.raws["final_norm"].astype(bfloat16)
    for block in final_blocks:
        assert np.array_equal(block[:576], final)


def test_slices_fit_the_join_and_kernel_constants():
    # Each tile's per-round result must fit one 256-slot join slice.
    for rows in (L.QKV_ROWS, L.OUT_ROWS, L.MLP_ROWS):
        assert rows <= L.SLICE
    # The hub gathers qkv heads in 16-value chunks; a chunk must never
    # straddle two tiles' slices.
    assert L.QKV_ROWS % 16 == 0 and L.HEAD_DIM % 16 == 0
    assert L.GEMV_TILES * L.QKV_ROWS == 960
    assert L.GEMV_TILES * L.OUT_ROWS == L.HIDDEN
    assert L.GEMV_TILES * L.MLP_ROWS == L.INTERMEDIATE
    assert L.BLOCK == 8 * L.HIDDEN == 3 * L.INTERMEDIATE


def test_supported_architectures():
    assert L.supports(FakeModel().metadata)
    qwen = dict(FakeModel().metadata, hidden_size=896, intermediate_size=4864)
    assert not L.supports(qwen)
    assert not L.supports(dict(FakeModel().metadata, context_length=128))


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"PASS {len(tests)} engine layout tests")


if __name__ == "__main__":
    main()
