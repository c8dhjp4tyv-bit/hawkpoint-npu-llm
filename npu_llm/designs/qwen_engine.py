"""Qwen2.5 0.5B decoder engine: all 24 decoder layers in one NPU dispatch.

Same dataflow as designs/engine.py (six GEMV tiles with their own weight
streams, a MemTile join, and a hub tile for attention), with Qwen's shapes:
14 query heads over 2 KV heads, a qkv bias, and a 4864-wide MLP. The layout
of weights, slices, and cache is in designs/qwen_engine_layout.py.

Per layer each GEMV tile runs four rounds:

    round 1  input RMSNorm, 192 qkv rows + bias -> hub: RoPE, attention
    round 2  o_proj rows                        -> hub forwards; residual, norm
    round 3  16-row gate/up groups and SiLU     -> hub forwards the activation
    round 4  down_proj rows in 896-column chunks-> hub forwards; residual

With ``lm_head`` set, every tile then applies the final RMSNorm and computes
its share of the 151936 LM-head rows, streaming FP32 logits straight to DDR.
Prefill positions whose logits are discarded use the variant without it.
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

from designs.qwen_engine_layout import (
    BLOCK,
    CACHE_HEAD,
    DOWN_CHUNKS,
    GEMV_TILES,
    HIDDEN,
    KV_HEADS,
    KV_NEW,
    LM_GROUP,
    LM_GROUPS,
    LM_START,
    MLP_ROWS,
    OUT_ROWS,
    QKV_ROWS,
    ROWS,
    SLICE,
    VECTOR,
    VOCAB,
    cache_elements,
    tile_blocks,
    tile_offsets,
)


KERNELS = Path(__file__).resolve().parents[1] / "kernels"
SRC = KERNELS / "qwen_engine_bf16.cc"
HUB_SRC = KERNELS / "qwen_engine_hub_bf16.cc"
GEMV_PLACEMENT = [Tile(0, 2), Tile(0, 3), Tile(1, 2), Tile(1, 3), Tile(2, 2), Tile(2, 3)]
HUB_PLACEMENT = Tile(3, 2)


class _KernelSource(ExternalFunction):
    """ExternalFunction whose call-site dtype check accepts ObjectFifo buffers
    (see designs/engine.py)."""

    def _validate_arg(self, index, arg, expected_ty):
        return None


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def qwen_engine(
    Weights: In,
    Cache: In,
    KvNew: Out,
    Logits: Out,
    *,
    layers: CompileTime[int] = 24,
    lm_head: CompileTime[bool] = True,
):
    bf16 = np.dtype[bfloat16]
    f32 = np.dtype[np.float32]
    block_ty = np.ndarray[(BLOCK,), bf16]
    slice_ty = np.ndarray[(SLICE,), bf16]
    vector_ty = np.ndarray[(VECTOR,), bf16]
    hidden_ty = np.ndarray[(HIDDEN,), bf16]
    bias_ty = np.ndarray[(QKV_ROWS,), f32]
    group_ty = np.ndarray[(16,), bf16]
    partial_ty = np.ndarray[(ROWS * 32,), f32]
    head_ty = np.ndarray[(CACHE_HEAD,), bf16]
    kv_ty = np.ndarray[(KV_NEW,), bf16]
    lut_ty = np.ndarray[(256,), bf16]
    logits_ty = np.ndarray[(LM_GROUP,), f32]

    include_dirs = [
        config.cxx_header_path(),
        str(Path(config.root_path()) / "aie_runtime_lib/AIE2"),
        str(KERNELS),
    ]
    # Two object files: each AIE2 core has 16 KB of program memory, and the
    # GEMV tiles must not carry the hub's attention code (or vice versa).
    load_hidden = _KernelSource(
        "q_load_hidden", source_file=str(SRC),
        arg_types=[vector_ty, hidden_ty], include_dirs=include_dirs,
    )
    obj = load_hidden.object_file_name
    hub_init = _KernelSource(
        "q_hub_init", source_file=str(HUB_SRC),
        arg_types=[head_ty, lut_ty, vector_ty], include_dirs=include_dirs,
    )
    hub_obj = hub_init.object_file_name
    norm_params = Kernel("q_norm_params", obj, [hidden_ty, block_ty, hidden_ty])
    norm_local = Kernel("q_norm_local", obj, [hidden_ty, hidden_ty, hidden_ty])
    save_params = Kernel("q_save_params", obj, [block_ty, hidden_ty, bias_ty])
    gemv_slice = Kernel("q_gemv_slice", obj, [block_ty, hidden_ty, slice_ty, np.int32])
    gemv_group = Kernel("q_gemv_gate", obj, [block_ty, hidden_ty, group_ty, np.int32])
    add_bias = Kernel("q_add_bias", obj, [slice_ty, bias_ty])
    residual = Kernel("q_residual", obj, [hidden_ty, vector_ty, hidden_ty])
    silu16 = Kernel("q_silu16", obj, [group_ty, group_ty, slice_ty, np.int32])
    down_chunk = Kernel("q_down_chunk", obj, [block_ty, vector_ty, partial_ty, np.int32])
    down_finish = Kernel("q_down_finish", obj, [partial_ty, slice_ty, np.int32])
    zero_tail = Kernel("q_zero_tail", obj, [slice_ty, np.int32])
    gemv_logits = Kernel("q_gemv_logits", obj, [block_ty, hidden_ty, logits_ty, np.int32])
    copy_vector = Kernel("q_copy_vector", hub_obj, [vector_ty, vector_ty])
    attention = Kernel(
        "q_attention", hub_obj, [vector_ty, head_ty, lut_ty, vector_ty, kv_ty, np.int32]
    )

    weight_fifos = [
        ObjectFifo(block_ty, depth=2, name=f"qweights{i}") for i in range(GEMV_TILES)
    ]
    joined = ObjectFifo(vector_ty, depth=2, name="qjoined")
    slices = joined.prod().join(
        [i * SLICE for i in range(GEMV_TILES)],
        obj_types=[slice_ty] * GEMV_TILES,
        names=[f"qslice{i}" for i in range(GEMV_TILES)],
        depths=[2] * GEMV_TILES,
        tile=Tile(1, 1),
    )
    broadcast = ObjectFifo(vector_ty, depth=2, name="qbroadcast")
    cache = ObjectFifo(head_ty, depth=1, name="qcache")
    kv_new = ObjectFifo(kv_ty, depth=1, name="qkv_new")
    logit_fifos = [
        ObjectFifo(logits_ty, depth=2, name=f"qlogits{i}") for i in range(GEMV_TILES)
    ]

    def gemv_program(out_blocks, mlp_groups, lm_groups):
        def gemv_tile(
            wf, bc, out, hidden, x, gamma_post, bias, gate, up, partial,
            load_hidden, norm_params, norm_local, save_params, gemv_slice,
            gemv_group, add_bias, residual, silu16,
            down_chunk, down_finish, zero_tail, *tail,
        ):
            b = bc.acquire(1)
            load_hidden(b, hidden)
            bc.release(1)
            for _ in range_(layers):
                # Round 1: input RMSNorm, qkv rows, bias.
                w = wf.acquire(1)
                norm_params(hidden, w, x)
                save_params(w, gamma_post, bias)
                wf.release(1)
                s = out.acquire(1)
                for blk in range_(QKV_ROWS // ROWS):
                    w = wf.acquire(1)
                    gemv_slice(w, x, s, blk * ROWS)
                    wf.release(1)
                add_bias(s, bias)
                out.release(1)
                # Round 2: o_proj rows from the attended heads.
                attended = bc.acquire(1)
                load_hidden(attended, x)
                bc.release(1)
                s = out.acquire(1)
                for blk in range_(out_blocks):
                    w = wf.acquire(1)
                    gemv_slice(w, x, s, blk * ROWS)
                    wf.release(1)
                out.release(1)
                o_rows = bc.acquire(1)
                residual(hidden, o_rows, x)
                bc.release(1)
                norm_local(hidden, gamma_post, x)
                # Round 3: 16-row gate and up groups, SiLU.
                s = out.acquire(1)
                for group in range_(mlp_groups):
                    for part in range_(4):
                        w = wf.acquire(1)
                        gemv_group(w, x, gate, part * ROWS)
                        wf.release(1)
                    for part in range_(4):
                        w = wf.acquire(1)
                        gemv_group(w, x, up, part * ROWS)
                        wf.release(1)
                    silu16(gate, up, s, group * 16)
                zero_tail(s, mlp_groups * 16)
                out.release(1)
                # Round 4: down rows over the six activation chunks.
                mlp = bc.acquire(1)
                s = out.acquire(1)
                for blk in range_(out_blocks):
                    for chunk in range_(DOWN_CHUNKS):
                        w = wf.acquire(1)
                        down_chunk(w, mlp, partial, chunk)
                        wf.release(1)
                    down_finish(partial, s, blk * ROWS)
                out.release(1)
                bc.release(1)
                down_rows = bc.acquire(1)
                residual(hidden, down_rows, x)
                bc.release(1)
            if len(tail) == 1:
                # Without the LM head, tile 0 signals the end of its last layer.
                # The dispatch waits on this drain; otherwise it would complete
                # once the last K/V rows are out, while the final rounds of the
                # layer still run into the next dispatch.
                tail[0].acquire(1)
                tail[0].release(1)
            elif tail:
                logits, gemv_logits = tail
                w = wf.acquire(1)
                norm_params(hidden, w, x)
                wf.release(1)
                for _ in range_(lm_groups):
                    out_logits = logits.acquire(1)
                    for part in range_(LM_GROUP // ROWS):
                        w = wf.acquire(1)
                        gemv_logits(w, x, out_logits, part * ROWS)
                        wf.release(1)
                    logits.release(1)

        return gemv_tile

    def hub_tile(cf, jn, bc, kv, lut, hub_init, attention, copy_vector):
        header = cf.acquire(1)
        first = bc.acquire(1)
        hub_init(header, lut, first)
        cf.release(1)
        bc.release(1)
        for _ in range_(layers):
            qkv = jn.acquire(1)
            attended = bc.acquire(1)
            new_rows = kv.acquire(1)
            for head in range_(KV_HEADS):
                c = cf.acquire(1)
                attention(qkv, c, lut, attended, new_rows, head)
                cf.release(1)
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
            Buffer(hidden_ty, name=f"qhidden{i}"),
            Buffer(hidden_ty, name=f"qnormed{i}"),
            Buffer(hidden_ty, name=f"qgamma_post{i}"),
            Buffer(bias_ty, name=f"qbias{i}"),
            Buffer(group_ty, name=f"qgate{i}"),
            Buffer(group_ty, name=f"qup{i}"),
            Buffer(partial_ty, name=f"qpartial{i}"),
            load_hidden, norm_params, norm_local, save_params, gemv_slice,
            gemv_group, add_bias, residual, silu16,
            down_chunk, down_finish, zero_tail,
        ]
        if lm_head:
            args += [logit_fifos[i].prod(), gemv_logits]
        elif i == 0:
            args += [logit_fifos[0].prod()]
        workers.append(Worker(
            gemv_program(OUT_ROWS[i] // ROWS, MLP_ROWS[i] // 16, LM_GROUPS[i]),
            args, tile=GEMV_PLACEMENT[i], stack_size=3072,
            dynamic_objfifo_lowering=True,
        ))
    workers.append(Worker(
        hub_tile,
        [cache.cons(), joined.cons(depth=1), broadcast.prod(), kv_new.prod(),
         Buffer(lut_ty, name="qlut"), hub_init, attention, copy_vector],
        tile=HUB_PLACEMENT, stack_size=4096,
        dynamic_objfifo_lowering=True,
    ))

    offsets, total_weights = tile_offsets(layers)
    total_cache = cache_elements(layers)
    rt = Runtime()
    with rt.sequence(
        np.ndarray[(total_weights,), bf16],
        np.ndarray[(total_cache,), bf16],
        np.ndarray[(layers * KV_NEW,), bf16],
        np.ndarray[(VOCAB,), f32],
    ) as (weights, cache_in, kv_out, logits):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(GEMV_TILES):
            tap = TensorAccessPattern(
                (total_weights,), offsets[i],
                [1, 1, tile_blocks(i, layers, lm_head), BLOCK], [0, 0, BLOCK, 1],
            )
            rt.fill(weight_fifos[i].prod(), weights, tap, task_group=tg,
                    tile=Tile(i // 2, 0))
        rt.fill(
            cache.prod(), cache_in,
            TensorAccessPattern(
                (total_cache,), 0,
                [1, 1, 1 + KV_HEADS * layers, CACHE_HEAD], [0, 0, CACHE_HEAD, 1],
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
        if not lm_head:
            rt.drain(
                logit_fifos[0].cons(), logits,
                TensorAccessPattern((VOCAB,), 0, [1, 1, 1, LM_GROUP], [0, 0, LM_GROUP, 1]),
                wait=True, task_group=tg, tile=Tile(0, 0),
            )
        if lm_head:
            for i in range(GEMV_TILES):
                rt.drain(
                    logit_fifos[i].cons(), logits,
                    TensorAccessPattern(
                        (VOCAB,), LM_START[i],
                        [1, 1, LM_GROUPS[i], LM_GROUP], [0, 0, LM_GROUP, 1],
                    ),
                    wait=True, task_group=tg, tile=Tile(i // 2, 0),
                )
        rt.finish_task_group(tg)
    return Program(iron.get_current_device(), rt).resolve_program()
