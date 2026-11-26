"""Data layout shared by the decoder engine design and its host runtime.

Pure NumPy, no MLIR-AIE import, so the weight packing can be tested without
an NPU. See designs/engine.py for the dataflow these layouts feed.
"""

import numpy as np
from ml_dtypes import bfloat16

from context_window import (
    DEFAULT_CONTEXT_LENGTH,
    KV_TILE,
    MAX_CONTEXT_LENGTH,
    context_tiles,
)


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
HEAD_DIM = 64
KV_HEADS = 3
TILE_ELEMENTS = KV_TILE * HEAD_DIM  # one streamed tile of keys, or of values
HEADER_HIDDEN = 128         # header tile: [LUT | pad | hidden | free]
# The header travels as the first object of the key stream, so it is exactly
# one tile; the fields above need 704 slots and the rest stays unused.
CACHE_HEAD = TILE_ELEMENTS
STREAM_COPIES = 4           # two key passes, then one key/value pair per tile
KV_NEW = KV_HEADS * 2 * HEAD_DIM
Q_PER_KV = 3
# Softmax bookkeeping the hub tile keeps across streamed tiles: the running
# maximum, the inverse of the summed exponentials, and 16 partial sums per
# query head. Laid out exactly as ENG_STATE_* in kernels/engine_bf16.cc.
STATE_SIZE = 32 + 16 * KV_HEADS * Q_PER_KV
# One tile's normalized attention weights per query head of a KV head.
WEIGHTS_SIZE = Q_PER_KV * KV_TILE
ACC_SIZE = KV_HEADS * Q_PER_KV * HEAD_DIM

# Architecture the layout is written for (SmolLM and SmolLM2 135M).
SUPPORTED = {
    "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE,
    "attention_heads": 9,
    "kv_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
}


def supports(metadata):
    """True when a converted model matches the engine's fixed layout."""
    if not all(metadata.get(key) == value for key, value in SUPPORTED.items()):
        return False
    context_length = metadata.get("context_length", DEFAULT_CONTEXT_LENGTH)
    return (
        isinstance(context_length, int)
        and not isinstance(context_length, bool)
        and 1 <= context_length <= MAX_CONTEXT_LENGTH
    )


def tile_blocks(layers):
    """Weight blocks per tile stream: every layer plus the final-norm block."""
    return layers * BLOCKS_PER_LAYER + 1


def weight_elements(layers):
    return GEMV_TILES * tile_blocks(layers) * BLOCK


def cache_elements(layers, context_length=DEFAULT_CONTEXT_LENGTH):
    """Slots of the cache stream: the header, then ``STREAM_COPIES`` regions.

    Every layer and KV head owns ``context_tiles`` tiles of keys followed by
the same tiles of values, and the design streams each tile in position order.
The stream is one contiguous walk: the header, two key passes (maximum and
sums), then a third key pass paired with the values. Keys are therefore stored
three times and values once. A tile's data memory has two input DMA channels,
one of which the joined qkv vector already uses, so everything attention reads
has to arrive on a single channel.
    """
    tiles = KV_HEADS * layers * context_tiles(context_length)
    return CACHE_HEAD + STREAM_COPIES * tiles * TILE_ELEMENTS


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
