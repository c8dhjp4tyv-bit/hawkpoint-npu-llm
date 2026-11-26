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

from context_window import DEFAULT_CONTEXT_LENGTH, KV_TILE, context_tiles


GEMV_TILES = 6
HIDDEN = 896
INTERMEDIATE = 4864
QKV = 1152
HEAD_DIM = 64
Q_HEADS = 14
KV_HEADS = 2
Q_PER_KV = Q_HEADS // KV_HEADS
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

TILE_ELEMENTS = KV_TILE * HEAD_DIM           # one streamed K or V tile
# The header travels as the first object of the key stream, so it is exactly
# one tile; the fields below need 1152 slots and the rest stays unused.
CACHE_HEAD = TILE_ELEMENTS
STREAM_COPIES = 4           # two key passes, then one key/value pair per tile
KV_NEW = KV_HEADS * 2 * HEAD_DIM
# Header object (first cache-stream object): RoPE cos/sin split into three
# BF16 parts each, the position, and the token's embedding.
HEADER_COS = 0          # cos hi, mid, lo (3 x 32)
HEADER_SIN = 96         # sin hi, mid, lo (3 x 32)
HEADER_POSITION = 192
HEADER_HIDDEN = 256
# Softmax bookkeeping the hub keeps across streamed tiles: the running score
# maximum, the inverse of the summed exponentials, and 16 partial sums per
# query head. Laid out exactly as the STATE_* constants in
# kernels/qwen_engine_hub_bf16.cc.
STATE_SIZE = 32 + 16 * Q_HEADS
# One tile's normalized weights per query head, as three BF16 parts each.
WEIGHTS_SIZE = Q_PER_KV * 3 * KV_TILE
ACC_SIZE = Q_HEADS * HEAD_DIM

# The streamed window is capped by the hub tile's *program* memory, not by its
# data memory: attention scores every cached position three times (maximum,
# summed exponentials, weighted values) and also carries the current token's
# own bookkeeping, which fills 16 KB of instructions at four 128-position
# tiles. The SmolLM engine's hub is smaller and streams the full 8K.
MAX_CONTEXT_TILES = 4
MAX_CONTEXT_LENGTH = KV_TILE * MAX_CONTEXT_TILES

SUPPORTED = {
    "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE,
    "attention_heads": Q_HEADS,
    "kv_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "model_family": "qwen2",
    "vocab_size": VOCAB,
}

assert sum(OUT_ROWS) == HIDDEN and all(r % ROWS == 0 for r in OUT_ROWS)
assert sum(MLP_ROWS) == INTERMEDIATE and all(r % 16 == 0 for r in MLP_ROWS)
assert max(QKV_ROWS, *OUT_ROWS, *MLP_ROWS) <= SLICE
assert sum(LM_GROUPS) * LM_GROUP == VOCAB


def supports(metadata):
    if not all(metadata.get(key) == value for key, value in SUPPORTED.items()):
        return False
    context_length = metadata.get("context_length", DEFAULT_CONTEXT_LENGTH)
    return (
        isinstance(context_length, int)
        and not isinstance(context_length, bool)
        and 1 <= context_length <= MAX_CONTEXT_LENGTH
    )


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


def cache_elements(layers, context_length=DEFAULT_CONTEXT_LENGTH):
    """Slots of the cache stream: the header, then ``STREAM_COPIES`` regions.

    Keys are stored three times because attention reads them once per pass and
    a single DMA pattern cannot loop back; the copies make the whole key
    stream one contiguous walk and leave the hub two input channels.
    """
    tiles = KV_HEADS * layers * context_tiles(context_length)
    return CACHE_HEAD + STREAM_COPIES * tiles * TILE_ELEMENTS


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
