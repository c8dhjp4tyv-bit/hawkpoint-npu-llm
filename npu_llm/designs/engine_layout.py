"""Data layout shared by the decoder engine design and its host runtime.

Pure NumPy, no MLIR-AIE import, so the weight packing can be tested without
an NPU. See designs/engine.py for the dataflow these layouts feed.
"""

import numpy as np
from ml_dtypes import bfloat16


GEMV_TILES = 6
SLICE = 256                 # result slots per GEMV tile and round
VECTOR = GEMV_TILES * SLICE
HIDDEN = 576
INTERMEDIATE = 1536
BLOCK = 4608                # 8 rows x 576, or 3 rows x 1536, BF16 values
QKV_BLOCKS, O_BLOCKS, MLP_GROUPS, DOWN_BLOCKS = 20, 12, 16, 32
BLOCKS_PER_LAYER = 1 + QKV_BLOCKS + O_BLOCKS + 4 * MLP_GROUPS + DOWN_BLOCKS
QKV_ROWS = QKV_BLOCKS * 8   # 160 qkv rows per tile
OUT_ROWS = O_BLOCKS * 8     # 96 o_proj / down_proj rows per tile
MLP_ROWS = MLP_GROUPS * 16  # 256 gate/up rows per tile
CONTEXT = 64
HEAD_DIM = 64
KV_HEADS = 3
CACHE_HEAD = 2 * CONTEXT * HEAD_DIM  # keys [64 x 64] then values [64 x 64]
KV_NEW = KV_HEADS * 2 * HEAD_DIM
HEADER_HIDDEN = 128         # header object: [LUT | pad | hidden]

# Architecture the layout is written for (SmolLM and SmolLM2 135M).
SUPPORTED = {
    "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE,
    "attention_heads": 9,
    "kv_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "context_length": CONTEXT,
}


def supports(metadata):
    """True when a converted model matches the engine's fixed layout."""
    return all(metadata.get(key) == value for key, value in SUPPORTED.items())


def tile_blocks(layers):
    """Weight blocks per tile stream: every layer plus the final-norm block."""
    return layers * BLOCKS_PER_LAYER + 1


def weight_elements(layers):
    return GEMV_TILES * tile_blocks(layers) * BLOCK


def cache_elements(layers):
    return (1 + KV_HEADS * layers) * CACHE_HEAD


def tile_layer_blocks(model, layer, tile):
    """One GEMV tile's weight stream for one layer, as BLOCK-sized rows.

    Order: both RMSNorm gammas, this tile's 160 qkv rows, 96 o_proj rows,
    16 groups of (16 gate rows, 16 matching up rows), then 96 down_proj rows.
    """
    prefix = f"layer{layer:02d}"
    qkv = np.asarray(model.bf16_projection(f"{prefix}.qkv"))
    o_proj = np.asarray(model.bf16_projection(f"{prefix}.o_proj"))
    gate_up = np.asarray(model.bf16_projection(f"{prefix}.gate_up"))
    down = np.asarray(model.bf16_projection(f"{prefix}.down_proj"))
    gate, up = gate_up[:INTERMEDIATE], gate_up[INTERMEDIATE:]

    gammas = np.zeros(BLOCK, dtype=bfloat16)
    gammas[:HIDDEN] = np.asarray(model.raw(f"{prefix}.input_norm")).astype(bfloat16)
    gammas[HIDDEN:2 * HIDDEN] = np.asarray(
        model.raw(f"{prefix}.post_attn_norm")
    ).astype(bfloat16)

    parts = [gammas.reshape(1, BLOCK)]
    parts.append(qkv[tile * QKV_ROWS:(tile + 1) * QKV_ROWS].reshape(-1, BLOCK))
    parts.append(o_proj[tile * OUT_ROWS:(tile + 1) * OUT_ROWS].reshape(-1, BLOCK))
    first = tile * MLP_ROWS
    for group in range(MLP_GROUPS):
        rows = slice(first + group * 16, first + group * 16 + 16)
        parts.append(gate[rows].reshape(-1, BLOCK))
        parts.append(up[rows].reshape(-1, BLOCK))
    parts.append(down[tile * OUT_ROWS:(tile + 1) * OUT_ROWS].reshape(-1, BLOCK))
    blocks = np.concatenate(parts).astype(bfloat16, copy=False)
    if blocks.shape != (BLOCKS_PER_LAYER, BLOCK):
        raise RuntimeError(f"{prefix} packed into {blocks.shape} engine blocks")
    return blocks


def pack_weights(model, layers):
    """All six tile streams, tile-major: every layer in order, then the final
    RMSNorm gamma (only tile 0 uses it)."""
    packed = np.zeros(weight_elements(layers), dtype=bfloat16)
    view = packed.reshape(GEMV_TILES, tile_blocks(layers), BLOCK)
    final_gamma = np.asarray(model.raw("final_norm")).astype(bfloat16)
    for layer in range(layers):
        first = layer * BLOCKS_PER_LAYER
        for tile in range(GEMV_TILES):
            view[tile, first:first + BLOCKS_PER_LAYER] = tile_layer_blocks(
                model, layer, tile
            )
    view[:, -1, :HIDDEN] = final_gamma
    return packed
