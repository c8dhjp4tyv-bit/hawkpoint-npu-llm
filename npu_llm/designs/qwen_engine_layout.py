"""Data layout for the Qwen2.5 0.5B decoder engine (designs/qwen_engine.py).

Pure NumPy, no MLIR-AIE import, so the weight packing can be tested without
an NPU. The engine uses the same dataflow as the SmolLM engine: six GEMV
tiles, each with its own weight stream and a fixed slice of every
projection's output rows, a MemTile join, and a hub tile for attention.

Qwen's dimensions do not divide evenly by six, so tiles own uneven row
ranges, and each tile's result slice sits at a fixed 896-slot offset in the
joined vector. The down projection is streamed in 896-column chunks that
line up with those slices, so the MLP activation never needs compacting.
"""

import numpy as np
from ml_dtypes import bfloat16


GEMV_TILES = 6
HIDDEN = 896
INTERMEDIATE = 4864
QKV = 1152
HEAD_DIM = 64
Q_HEADS = 14
KV_HEADS = 2
CONTEXT = 64
SLICE = HIDDEN                   # result slots per tile and round
VECTOR = GEMV_TILES * SLICE      # joined / broadcast vector
ROWS = 4                         # projection rows per weight block
BLOCK = ROWS * HIDDEN            # 3584 BF16 values per weight block

QKV_ROWS = QKV // GEMV_TILES                  # 192 per tile: three heads
OUT_ROWS = (152, 152, 148, 148, 148, 148)     # o_proj and down_proj rows
MLP_ROWS = (816, 816, 816, 816, 800, 800)     # gate/up rows (16-row groups)
OUT_START = tuple(int(sum(OUT_ROWS[:t])) for t in range(GEMV_TILES))
MLP_START = tuple(int(sum(MLP_ROWS[:t])) for t in range(GEMV_TILES))
DOWN_CHUNKS = GEMV_TILES                      # one 896-slot chunk per slice
VOCAB = 151936
LM_GROUP = 64                                 # logits per output object
LM_GROUPS = (396, 396, 396, 396, 395, 395)    # 64-row LM-head groups per tile
LM_START = tuple(int(sum(LM_GROUPS[:t])) * LM_GROUP for t in range(GEMV_TILES))

# Parameter block at the start of each layer's stream:
# [input_norm gamma | post_attention_norm gamma | this tile's qkv bias (FP32)]
PARAM_GAMMA_IN = 0
PARAM_GAMMA_POST = HIDDEN
PARAM_BIAS = 2 * HIDDEN

CACHE_HEAD = 2 * CONTEXT * HEAD_DIM           # keys [64 x 64] then values
KV_NEW = KV_HEADS * 2 * HEAD_DIM
# Header object (first cache-stream object): RoPE cos/sin split into three
# BF16 parts each, the position, and the token's embedding.
HEADER_COS = 0          # cos hi, mid, lo (3 x 32)
HEADER_SIN = 96         # sin hi, mid, lo (3 x 32)
HEADER_POSITION = 192
HEADER_HIDDEN = 256

SUPPORTED = {
    "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE,
    "attention_heads": Q_HEADS,
    "kv_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "context_length": CONTEXT,
    "model_family": "qwen2",
    "vocab_size": VOCAB,
}

assert sum(OUT_ROWS) == HIDDEN and all(r % ROWS == 0 for r in OUT_ROWS)
assert sum(MLP_ROWS) == INTERMEDIATE and all(r % 16 == 0 for r in MLP_ROWS)
assert max(QKV_ROWS, *OUT_ROWS, *MLP_ROWS) <= SLICE
assert sum(LM_GROUPS) * LM_GROUP == VOCAB


def supports(metadata):
    return all(metadata.get(key) == value for key, value in SUPPORTED.items())


def layer_blocks(tile):
    """Weight blocks one tile consumes per layer."""
    return (
        1
        + QKV_ROWS // ROWS
        + OUT_ROWS[tile] // ROWS
        + (MLP_ROWS[tile] // 16) * 8
        + (OUT_ROWS[tile] // ROWS) * DOWN_CHUNKS
    )


def head_blocks(tile):
    """Final-norm block plus this tile's LM-head row blocks."""
    return 1 + LM_GROUPS[tile] * LM_GROUP // ROWS


def tile_blocks(tile, layers, lm_head=True):
    return layers * layer_blocks(tile) + (head_blocks(tile) if lm_head else 0)


def tile_offsets(layers):
    """Element offset of each tile's stream in the packed weight buffer.

    Every stream holds the decoder layers followed by the LM head, so a
    dispatch that stops before the LM head reads a prefix of the same stream.
    """
    offsets, total = [], 0
    for tile in range(GEMV_TILES):
        offsets.append(total)
        total += tile_blocks(tile, layers) * BLOCK
    return offsets, total


def cache_elements(layers):
    return (1 + KV_HEADS * layers) * CACHE_HEAD


def down_padded(down):
    """down_proj [896, 4864] -> [896, 6 x 896] in joined-slice column order."""
    padded = np.zeros((HIDDEN, VECTOR), dtype=bfloat16)
    for tile in range(GEMV_TILES):
        first = MLP_START[tile]
        padded[:, tile * SLICE:tile * SLICE + MLP_ROWS[tile]] = down[
            :, first:first + MLP_ROWS[tile]
        ]
    return padded


def tile_layer_blocks(model, layer, tile):
    prefix = f"layer{layer:02d}"
    qkv = np.asarray(model.bf16_projection(f"{prefix}.qkv"))
    o_proj = np.asarray(model.bf16_projection(f"{prefix}.o_proj"))
    gate_up = np.asarray(model.bf16_projection(f"{prefix}.gate_up"))
    down = down_padded(np.asarray(model.bf16_projection(f"{prefix}.down_proj")))
    gate, up = gate_up[:INTERMEDIATE], gate_up[INTERMEDIATE:]

    params = np.zeros(BLOCK, dtype=bfloat16)
    params[PARAM_GAMMA_IN:PARAM_GAMMA_IN + HIDDEN] = np.asarray(
        model.raw(f"{prefix}.input_norm")
    ).astype(bfloat16)
    params[PARAM_GAMMA_POST:PARAM_GAMMA_POST + HIDDEN] = np.asarray(
        model.raw(f"{prefix}.post_attn_norm")
    ).astype(bfloat16)
    bias = np.asarray(model.raw(f"{prefix}.qkv_bias"), dtype=np.float32)
    rows = slice(tile * QKV_ROWS, (tile + 1) * QKV_ROWS)
    params[PARAM_BIAS:PARAM_BIAS + 2 * QKV_ROWS] = bias[rows].view(bfloat16)

    parts = [params.reshape(1, BLOCK)]
    parts.append(qkv[rows].reshape(-1, BLOCK))
    out_rows = slice(OUT_START[tile], OUT_START[tile] + OUT_ROWS[tile])
    parts.append(o_proj[out_rows].reshape(-1, BLOCK))
    first = MLP_START[tile]
    for group in range(MLP_ROWS[tile] // 16):
        group_rows = slice(first + group * 16, first + group * 16 + 16)
        parts.append(gate[group_rows].reshape(-1, BLOCK))
        parts.append(up[group_rows].reshape(-1, BLOCK))
    tile_down = down[out_rows]
    for row in range(0, OUT_ROWS[tile], ROWS):
        for chunk in range(DOWN_CHUNKS):
            parts.append(
                tile_down[row:row + ROWS, chunk * SLICE:(chunk + 1) * SLICE].reshape(
                    1, BLOCK
                )
            )
    blocks = np.concatenate(parts).astype(bfloat16, copy=False)
    if blocks.shape != (layer_blocks(tile), BLOCK):
        raise RuntimeError(f"{prefix} tile {tile} packed into {blocks.shape}")
    return blocks


def tile_head_blocks(model, tile):
    """Final RMSNorm gamma, then this tile's LM-head rows in 4-row blocks."""
    lm_head = np.asarray(model.bf16_projection("lm_head"))
    params = np.zeros(BLOCK, dtype=bfloat16)
    params[:HIDDEN] = np.asarray(model.raw("final_norm")).astype(bfloat16)
    rows = lm_head[LM_START[tile]:LM_START[tile] + LM_GROUPS[tile] * LM_GROUP]
    return np.concatenate([params.reshape(1, BLOCK), rows.reshape(-1, BLOCK)])


def pack_weights(model, layers):
    offsets, total = tile_offsets(layers)
    packed = np.zeros(total, dtype=bfloat16)
    for tile in range(GEMV_TILES):
        per_layer = layer_blocks(tile) * BLOCK
        for layer in range(layers):
            start = offsets[tile] + layer * per_layer
            packed[start:start + per_layer] = tile_layer_blocks(
                model, layer, tile
            ).reshape(-1)
        start = offsets[tile] + layers * per_layer
        head = tile_head_blocks(model, tile).reshape(-1)
        packed[start:start + head.size] = head
    return packed
