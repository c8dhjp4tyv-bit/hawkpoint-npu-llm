"""Row-split SmolLM decoder engine: every decoder layer in one NPU dispatch.

Weight streaming is the dominant cost of a decode step, and one AIE stream
tops out near 5-7 GB/s. The layer-pipelined decoder (designs/decoder_layer.py)
feeds each projection from a single stream, so only one stream is busy at a
time. Here six GEMV tiles each own a fixed slice of *every* projection's
output rows and read their own weight stream, so all six streams are busy in
every phase of the layer.

Per layer, every GEMV tile runs four rounds. In each round it writes its
result slice (256 slots); a MemTile joins the six slices and hands them to
the hub tile, which answers with a broadcast vector:

    round 1  qkv slice      -> hub runs RoPE + attention -> attended heads
    round 2  o_proj slice   -> hub forwards              -> o output
    round 3  SwiGLU slice   -> hub forwards              -> MLP activation
    round 4  down slice     -> hub forwards              -> down output

Every GEMV tile keeps its own copy of the residual stream and applies the
residual adds and RMSNorms locally, so the hidden state never leaves the
array between layers. The hub reads each layer's K/V cache from DDR one
tile of positions at a time and emits only the current position's key and
value; the host appends them.

A head's cache does not fit in tile memory once the context grows past a few
hundred tokens, so attention runs over streamed tiles: the hub reads all key
tiles to find the score maximum, reads them again to sum the exponentials,
and reads them once more alongside the value tiles to accumulate the output.
The current token's key and value travel through ``kv_new`` instead of DDR and
join the same three passes. See kernels/engine_bf16.cc for why the passes are
kept separate rather than fused into an online softmax.
"""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils import config


from context_window import DEFAULT_CONTEXT_LENGTH, KV_TILE, context_tiles
from designs.engine_layout import (
    ACC_SIZE,
    BLOCK,
    CACHE_HEAD,
    DOWN_BLOCKS,
    GEMV_TILES,
    HIDDEN,
    KV_HEADS,
    KV_NEW,
    MLP_GROUPS,
    O_BLOCKS,
    QKV_BLOCKS,
    SLICE,
    STATE_SIZE,
    STREAM_COPIES,
    WEIGHTS_SIZE,
    TILE_ELEMENTS,
    VECTOR,
    cache_elements,
    tile_blocks,
)


KERNELS = Path(__file__).resolve().parents[1] / "kernels"
SRC = KERNELS / "engine_bf16.cc"
HUB_SRC = KERNELS / "engine_hub_bf16.cc"

GEMV_PLACEMENT = [Tile(0, 2), Tile(0, 3), Tile(1, 2), Tile(1, 3), Tile(2, 2), Tile(2, 3)]
HUB_PLACEMENT = Tile(3, 2)


class _KernelSource(ExternalFunction):
    """ExternalFunction whose call-site dtype check accepts ObjectFifo buffers.

    IRON compares an acquired buffer's MLIR element type (``bf16``) with the
    declared NumPy dtype (``bfloat16``) and rejects the match; ``Kernel``
    calls on the same object file are not checked at all.
    """

    def _validate_arg(self, index, arg, expected_ty):
        return None


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def decoder_engine(
    Weights: In,
    Cache: In,
    KvNew: Out,
    Hidden: Out,
    *,
    layers: CompileTime[int] = 30,
    tiles: CompileTime[int] = 0,
):
    bf16 = np.dtype[bfloat16]
    f32 = np.dtype[np.float32]
    block_ty = np.ndarray[(BLOCK,), bf16]
    slice_ty = np.ndarray[(SLICE,), bf16]
    vector_ty = np.ndarray[(VECTOR,), bf16]
    header_ty = np.ndarray[(CACHE_HEAD,), bf16]
    tile_ty = np.ndarray[(TILE_ELEMENTS,), bf16]
    kv_ty = np.ndarray[(KV_NEW,), bf16]
    hidden_ty = np.ndarray[(HIDDEN,), bf16]
    lut_ty = np.ndarray[(80,), bf16]
    up_ty = np.ndarray[(16,), bf16]
    query_ty = np.ndarray[(3 * 64,), bf16]
    state_ty = np.ndarray[(STATE_SIZE,), f32]
    acc_ty = np.ndarray[(ACC_SIZE,), f32]
    weights_ty = np.ndarray[(WEIGHTS_SIZE,), bf16]
    # ``tiles`` is the window this graph streams and reads. The host grows it
    # in steps and reallocates the cache, so a short prompt does not pay for an
    # 8k window.
    tiles = tiles or context_tiles(DEFAULT_CONTEXT_LENGTH)

    source = _KernelSource(
        "eng_load_hidden", source_file=str(SRC),
        arg_types=[vector_ty, hidden_ty],
        include_dirs=[
            config.cxx_header_path(),
            str(Path(config.root_path()) / "aie_runtime_lib/AIE2"),
        ],
    )
    obj = source.object_file_name
    # The hub tile's attention kernels live in their own object, compiled with
    # -Os: together they exceed 16 KB of program memory at -O2.
    hub_source = _KernelSource(
        "eng_hub_init", source_file=str(HUB_SRC),
        arg_types=[tile_ty, lut_ty, vector_ty],
        compile_flags=["-Os"],
        include_dirs=[
            config.cxx_header_path(),
            str(Path(config.root_path()) / "aie_runtime_lib/AIE2"),
        ],
    )
    hub_obj = hub_source.object_file_name
    load_hidden = source
    store_hidden = Kernel("eng_store_hidden", obj, [hidden_ty, hidden_ty])
    final_norm = Kernel("eng_final_norm", obj, [hidden_ty, block_ty, hidden_ty])
    save_gamma = Kernel("eng_save_gamma", obj, [block_ty, hidden_ty])
    rmsnorm_block = Kernel("eng_rmsnorm_block", obj, [hidden_ty, block_ty, hidden_ty])
    gemv576_x = Kernel(
        "eng_gemv8_hidden", obj, [block_ty, hidden_ty, slice_ty, np.int32]
    )
    copy_attended = Kernel("eng_copy_attended", obj, [vector_ty, hidden_ty])
    copy_mlp = Kernel("eng_copy_mlp", obj, [vector_ty, vector_ty])
    gemv576_up = Kernel(
        "eng_gemv8_up", obj, [block_ty, hidden_ty, up_ty, np.int32]
    )
    gemv1536 = Kernel(
        "eng_gemv3_k1536", obj, [block_ty, vector_ty, slice_ty, np.int32]
    )
    swiglu = Kernel("eng_swiglu16", obj, [up_ty, slice_ty, np.int32])
    residual = Kernel("eng_residual96", obj, [hidden_ty, vector_ty])
    rmsnorm_local = Kernel("eng_rmsnorm_local", obj, [hidden_ty, hidden_ty, hidden_ty])
    hub_init = Kernel("eng_hub_init", hub_obj, [header_ty, lut_ty, vector_ty])
    copy_vector = Kernel("eng_copy1536", hub_obj, [vector_ty, vector_ty])
    prepare = Kernel(
        "eng_attention_prepare", hub_obj,
        [vector_ty, lut_ty, query_ty, kv_ty, state_ty, acc_ty, np.int32],
    )
    attn_max = Kernel(
        "eng_attention_max", hub_obj,
        [query_ty, tile_ty, lut_ty, state_ty, np.int32, np.int32],
    )
    attn_max_new = Kernel(
        "eng_attention_max_new", hub_obj, [query_ty, kv_ty, state_ty, np.int32]
    )
    attn_sums = Kernel(
        "eng_attention_sums", hub_obj,
        [query_ty, tile_ty, lut_ty, state_ty, np.int32, np.int32],
    )
    attn_sums_new = Kernel(
        "eng_attention_sums_new", hub_obj,
        [query_ty, kv_ty, lut_ty, state_ty, np.int32],
    )
    sums_finish = Kernel("eng_attention_sums_finish", hub_obj, [state_ty, np.int32])
    attn_weights = Kernel(
        "eng_attention_weights", hub_obj,
        [query_ty, tile_ty, lut_ty, state_ty, weights_ty, np.int32, np.int32],
    )
    attn_apply = Kernel(
        "eng_attention_apply", hub_obj,
        [weights_ty, tile_ty, lut_ty, acc_ty, np.int32, np.int32],
    )
    attn_values_new = Kernel(
        "eng_attention_values_new", hub_obj,
        [query_ty, kv_ty, lut_ty, state_ty, acc_ty, np.int32],
    )
    emit = Kernel("eng_attention_emit", hub_obj, [acc_ty, vector_ty, np.int32])

    weight_fifos = [
        ObjectFifo(block_ty, depth=2, name=f"weights{i}") for i in range(GEMV_TILES)
    ]
    joined = ObjectFifo(vector_ty, depth=2, name="joined")
    slices = joined.prod().join(
        [i * SLICE for i in range(GEMV_TILES)],
        obj_types=[slice_ty] * GEMV_TILES,
        names=[f"slice{i}" for i in range(GEMV_TILES)],
        depths=[2] * GEMV_TILES,
        tile=Tile(1, 1),
    )
    broadcast = ObjectFifo(vector_ty, depth=2, name="broadcast")
    # One stream: the header tile, the keys twice (maximum and sums), then a
    # key/value pair per tile for the values pass. The hub has two input DMA
    # channels and the joined qkv vector uses one of them.
    cache_tiles = ObjectFifo(tile_ty, depth=1, name="cache_tiles")
    kv_new = ObjectFifo(kv_ty, depth=1, name="kv_new")
    hidden_out = ObjectFifo(hidden_ty, depth=1, name="hidden_out")

    def gemv_tile(
        wf, bc, out, hidden, x, gamma_post, up, mlp_local,
        load_hidden, rmsnorm_block, save_gamma, gemv576_x, copy_attended, copy_mlp,
        gemv576_up, gemv1536, swiglu, residual, rmsnorm_local, *final,
    ):
        b = bc.acquire(1)
        load_hidden(b, hidden)
        bc.release(1)
        for _ in range_(layers):
            # Round 1: input RMSNorm and this tile's 160 qkv rows.
            w = wf.acquire(1)
            rmsnorm_block(hidden, w, x)
            save_gamma(w, gamma_post)
            wf.release(1)
            s = out.acquire(1)
            for blk in range_(QKV_BLOCKS):
                w = wf.acquire(1)
                gemv576_x(w, x, s, blk * 8)
                wf.release(1)
            out.release(1)
            # Round 2: o_proj rows from the attended heads. Broadcast vectors
            # are copied out and released at once so the hub can fill the
            # next broadcast buffer while this tile computes.
            attended = bc.acquire(1)
            copy_attended(attended, x)
            bc.release(1)
            s = out.acquire(1)
            for blk in range_(O_BLOCKS):
                w = wf.acquire(1)
                gemv576_x(w, x, s, blk * 8)
                wf.release(1)
            out.release(1)
            o_rows = bc.acquire(1)
            residual(hidden, o_rows)
            bc.release(1)
            rmsnorm_local(hidden, gamma_post, x)
            # Round 3: gate and up rows, fused SwiGLU.
            s = out.acquire(1)
            for group in range_(MLP_GROUPS):
                w = wf.acquire(1)
                gemv576_x(w, x, s, group * 16)
                wf.release(1)
                w = wf.acquire(1)
                gemv576_x(w, x, s, group * 16 + 8)
                wf.release(1)
                w = wf.acquire(1)
                gemv576_up(w, x, up, 0)
                wf.release(1)
                w = wf.acquire(1)
                gemv576_up(w, x, up, 8)
                wf.release(1)
                swiglu(up, s, group * 16)
            out.release(1)
            # Round 4: down rows from the full MLP activation.
            mlp = bc.acquire(1)
            copy_mlp(mlp, mlp_local)
            bc.release(1)
            s = out.acquire(1)
            for blk in range_(DOWN_BLOCKS):
                w = wf.acquire(1)
                gemv1536(w, mlp_local, s, blk * 3)
                wf.release(1)
            out.release(1)
            down_rows = bc.acquire(1)
            residual(hidden, down_rows)
            bc.release(1)
        # Every stream ends with the final RMSNorm gamma; tile 0 emits the
        # normalized hidden state the host feeds to the LM head.
        w = wf.acquire(1)
        if final:
            final_fifo, store_hidden, final_norm = final
            final_norm(hidden, w, x)
            h = final_fifo.acquire(1)
            store_hidden(x, h)
            final_fifo.release(1)
        wf.release(1)

    def hub_tile(
        cf, jn, bc, kv, lut, queries, state, acc, tile_weights,
        hub_init, prepare, attn_max, attn_max_new, attn_sums, attn_sums_new,
        sums_finish, attn_weights, attn_apply, attn_values_new, emit,
        copy_vector,
    ):
        first = cf.acquire(1)
        start = bc.acquire(1)
        hub_init(first, lut, start)
        cf.release(1)
        bc.release(1)
        for _ in range_(layers):
            qkv = jn.acquire(1)
            attended = bc.acquire(1)
            new_rows = kv.acquire(1)
            for head in range_(KV_HEADS):
                prepare(qkv, lut, queries, new_rows, state, acc, head)
                # Pass 1: the score maximum over every cached position.
                for tile in range_(tiles):
                    keys = cf.acquire(1)
                    attn_max(queries, keys, lut, state, head, tile)
                    cf.release(1)
                attn_max_new(queries, new_rows, state, head)
                # Pass 2: the exponential sum against that maximum.
                for tile in range_(tiles):
                    keys = cf.acquire(1)
                    attn_sums(queries, keys, lut, state, head, tile)
                    cf.release(1)
                attn_sums_new(queries, new_rows, lut, state, head)
                sums_finish(state, head)
                # Pass 3: the normalized weights against the value tiles.
                for tile in range_(tiles):
                    keys = cf.acquire(1)
                    attn_weights(queries, keys, lut, state, tile_weights, head, tile)
                    cf.release(1)
                    values = cf.acquire(1)
                    attn_apply(tile_weights, values, lut, acc, head, tile)
                    cf.release(1)
                attn_values_new(queries, new_rows, lut, state, acc, head)
                emit(acc, attended, head)
            kv.release(1)
            jn.release(1)
            bc.release(1)
            for _ in range_(3):
                rows = jn.acquire(1)
                forwarded = bc.acquire(1)
                copy_vector(rows, forwarded)
                jn.release(1)
                bc.release(1)

    workers = []
    for i in range(GEMV_TILES):
        args = [
            weight_fifos[i].cons(), broadcast.cons(), slices[i].prod(),
            Buffer(hidden_ty, name=f"hidden{i}"),
            Buffer(hidden_ty, name=f"normed{i}"),
            Buffer(hidden_ty, name=f"gamma_post{i}"),
            Buffer(up_ty, name=f"up{i}"),
            Buffer(vector_ty, name=f"mlp{i}"),
            load_hidden, rmsnorm_block, save_gamma, gemv576_x, copy_attended, copy_mlp,
            gemv576_up, gemv1536, swiglu, residual, rmsnorm_local,
        ]
        if i == 0:
            args += [hidden_out.prod(), store_hidden, final_norm]
        workers.append(Worker(gemv_tile, args, tile=GEMV_PLACEMENT[i], stack_size=2048,
                              dynamic_objfifo_lowering=False))
    workers.append(Worker(
        hub_tile,
        [cache_tiles.cons(), joined.cons(), broadcast.prod(), kv_new.prod(),
         Buffer(lut_ty, name="lut"),
         Buffer(query_ty, name="queries"), Buffer(state_ty, name="state"),
         Buffer(acc_ty, name="acc"), Buffer(weights_ty, name="tile_weights"),
         hub_init, prepare, attn_max, attn_max_new, attn_sums, attn_sums_new,
         sums_finish, attn_weights, attn_apply, attn_values_new, emit,
         copy_vector],
        tile=HUB_PLACEMENT,
        stack_size=4096,
        dynamic_objfifo_lowering=False,
    ))

    per_tile = tile_blocks(layers) * BLOCK
    total_weights = GEMV_TILES * per_tile
    total_cache = cache_elements(layers, tiles * KV_TILE)
    # [header][keys][keys][key/value pairs], the header being the first tile
    cache_objects = KV_HEADS * layers * tiles
    rt = Runtime()
    with rt.sequence(
        np.ndarray[(total_weights,), bf16],
        np.ndarray[(total_cache,), bf16],
        np.ndarray[(layers * KV_NEW,), bf16],
        hidden_ty,
    ) as (weights, cache_in, kv_out, hidden):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(GEMV_TILES):
            tap = TensorAccessPattern(
                (total_weights,), i * per_tile,
                [1, 1, tile_blocks(layers), BLOCK], [0, 0, BLOCK, 1],
            )
            rt.fill(weight_fifos[i].prod(), weights, tap, task_group=tg,
                    tile=Tile(i // 2, 0))
        rt.fill(
            cache_tiles.prod(), cache_in,
            TensorAccessPattern(
                (total_cache,), 0,
                [1, 1, 1 + STREAM_COPIES * cache_objects, TILE_ELEMENTS],
                [0, 0, TILE_ELEMENTS, 1],
            ),
            task_group=tg, tile=Tile(3, 0),
        )
        rt.drain(
            kv_new.cons(), kv_out,
            TensorAccessPattern(
                (layers * KV_NEW,), 0, [1, 1, layers, KV_NEW], [0, 0, KV_NEW, 1]
            ),
            wait=True, task_group=tg, tile=Tile(3, 0),
        )
        rt.drain(hidden_out.cons(), hidden, wait=True, task_group=tg, tile=Tile(0, 0))
        rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()
