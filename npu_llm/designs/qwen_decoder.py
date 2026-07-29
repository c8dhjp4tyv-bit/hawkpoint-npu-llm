"""One persistent XDNA1 program for a fused Qwen2.5 decoder stack."""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, InOut, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils import config


SRC = Path(__file__).resolve().parents[1] / "kernels/qwen_decoder_layer_bf16.cc"
HIDDEN = 896
INTERMEDIATE = 4864
QKV = 1152
QKV_BLOCK = 16 * HIDDEN
DOWN_BLOCK = 4 * INTERMEDIATE
QKV_BLOCKS = QKV // 16
O_BLOCKS = HIDDEN // 16
GATE_BLOCKS = (2 * INTERMEDIATE) // 16
DOWN_BLOCKS = HIDDEN // 4
QKV_BYTES = QKV_BLOCKS * QKV_BLOCK
O_BYTES = O_BLOCKS * QKV_BLOCK
GATE_BYTES = GATE_BLOCKS * QKV_BLOCK
DOWN_BYTES = DOWN_BLOCKS * DOWN_BLOCK
WEIGHT_BYTES = QKV_BYTES + O_BYTES + GATE_BYTES + DOWN_BYTES
GAMMA_BIAS = 2 * HIDDEN + 2 * QKV


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def qwen_decoder(
    PackedWeights: In,
    GammasBias: In,
    LUT: In,
    Hidden: InOut,
    Cache: InOut,
    *,
    layers: CompileTime[int] = 1,
):
    weights_ty = np.ndarray[(layers * WEIGHT_BYTES,), np.dtype[bfloat16]]
    params_ty = np.ndarray[(layers * GAMMA_BIAS,), np.dtype[bfloat16]]
    params_layer_ty = np.ndarray[(GAMMA_BIAS,), np.dtype[bfloat16]]
    lut_ty = np.ndarray[(66,), np.dtype[bfloat16]]
    hidden_ty = np.ndarray[(HIDDEN,), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(layers, 2, 8192), np.dtype[bfloat16]]
    cache_layer_ty = np.ndarray[(2, 8192), np.dtype[bfloat16]]
    row64 = np.ndarray[(64,), np.dtype[bfloat16]]
    block896 = np.ndarray[(QKV_BLOCK,), np.dtype[bfloat16]]
    block4864 = np.ndarray[(DOWN_BLOCK,), np.dtype[bfloat16]]
    packed1156 = np.ndarray[(1156,), np.dtype[bfloat16]]
    packed578 = np.ndarray[(578,), np.dtype[bfloat16]]
    cache8192 = np.ndarray[(8192,), np.dtype[bfloat16]]
    attended448 = np.ndarray[(448,), np.dtype[bfloat16]]
    mlp4864 = np.ndarray[(INTERMEDIATE,), np.dtype[bfloat16]]
    row32 = np.ndarray[(32,), np.dtype[bfloat16]]

    rms = ExternalFunction(
        "qwen_rmsnorm896",
        source_file=str(SRC),
        arg_types=[hidden_ty, hidden_ty, hidden_ty],
        include_dirs=[
            config.cxx_header_path(),
            str(Path(config.root_path()) / "aie_runtime_lib/AIE2"),
        ],
    )
    obj = rms.object_file_name
    project896 = Kernel(
        "qwen_project16_k896_bf16",
        obj,
        [block896, hidden_ty, row64, np.int32],
    )
    project4864 = Kernel(
        "qwen_project4_k4864_bf16",
        obj,
        [block4864, mlp4864, row32, np.int32],
    )
    rope = Kernel(
        "qwen_pack_rope64",
        obj,
        [row64, lut_ty, np.ndarray[(2 * QKV,), np.dtype[bfloat16]],
         packed1156, np.int32, np.int32],
    )
    attention = Kernel(
        "qwen_attention7",
        obj,
        [packed578, cache8192, cache8192, attended448],
    )
    residual64 = Kernel(
        "qwen_residual64", obj, [row64, hidden_ty, hidden_ty, np.int32]
    )
    store_gate = Kernel(
        "qwen_store_gate64", obj, [row64, mlp4864, np.int32]
    )
    swiglu_up = Kernel(
        "qwen_swiglu_up64", obj, [row64, mlp4864, np.int32]
    )
    residual32 = Kernel(
        "qwen_residual32", obj, [row32, hidden_ty, hidden_ty, np.int32]
    )

    fwq = ObjectFifo(block896, depth=1, name="qwen_w_qkv")
    fwo = ObjectFifo(block896, depth=1, name="qwen_w_o")
    fwg = ObjectFifo(block896, depth=1, name="qwen_w_gate")
    fwd = ObjectFifo(block4864, depth=1, name="qwen_w_down")
    finitial = ObjectFifo(hidden_ty, depth=1, name="qwen_hidden_initial")
    ffinal = ObjectFifo(hidden_ty, depth=1, name="qwen_hidden_final")
    fp_parent = ObjectFifo(params_layer_ty, depth=1, name="qwen_params")
    fgq, fgp, fbias = fp_parent.cons().split(
        [0, HIDDEN, 2 * HIDDEN],
        obj_types=[
            hidden_ty,
            hidden_ty,
            np.ndarray[(2 * QKV,), np.dtype[bfloat16]],
        ],
        names=["qwen_gamma_q", "qwen_gamma_post", "qwen_qkv_bias"],
        depths=[1, 1, 1],
    )
    flut = ObjectFifo(lut_ty, depth=1, name="qwen_lut")
    fcache_in_parent = ObjectFifo(cache_layer_ty, depth=1, name="qwen_cache_in")
    cache_in = fcache_in_parent.cons().split(
        [0, 8192],
        obj_types=[cache8192] * 2,
        names=["qwen_cache_in_0", "qwen_cache_in_1"],
        depths=[1, 1],
    )
    fcache_out_parent = ObjectFifo(cache_layer_ty, depth=1, name="qwen_cache_out")
    cache_out = fcache_out_parent.prod().join(
        [0, 8192],
        obj_types=[cache8192] * 2,
        names=["qwen_cache_out_0", "qwen_cache_out_1"],
        depths=[1, 1],
    )

    fnorm_q = ObjectFifo(hidden_ty, depth=1, name="qwen_norm_q")
    fqkv_rows = ObjectFifo(row64, depth=1, name="qwen_qkv_rows")
    fpacked = ObjectFifo(packed1156, depth=1, name="qwen_packed_qkv")
    packed_heads = fpacked.cons().split(
        [0, 578],
        obj_types=[packed578, packed578],
        names=["qwen_packed_0", "qwen_packed_1"],
        depths=[1, 1],
    )
    fatt_parent = ObjectFifo(hidden_ty, depth=1, name="qwen_attended")
    attended_heads = fatt_parent.prod().join(
        [0, 448],
        obj_types=[attended448, attended448],
        names=["qwen_attended_0", "qwen_attended_1"],
        depths=[1, 1],
    )
    fo_rows = ObjectFifo(row64, depth=1, name="qwen_o_rows")
    fafter = ObjectFifo(hidden_ty, depth=1, name="qwen_after_attention")
    fnorm_post = ObjectFifo(hidden_ty, depth=1, name="qwen_norm_post")
    fgate_rows = ObjectFifo(row64, depth=1, name="qwen_gate_rows")
    fmlp = ObjectFifo(mlp4864, depth=1, name="qwen_mlp")
    fdown_rows = ObjectFifo(row32, depth=1, name="qwen_down_rows")

    def normalize(xp, gp, op, fn):
        for _ in range_(layers):
            x, g, out = xp.acquire(1), gp.acquire(1), op.acquire(1)
            fn(x, g, out)
            xp.release(1); gp.release(1); op.release(1)

    def project64(wp, xp, op, fn, chunks):
        for _ in range_(layers):
            x = xp.acquire(1)
            for _ in range_(chunks):
                out = op.acquire(1)
                for offset in range_(0, 64, 16):
                    w = wp.acquire(1)
                    fn(w, x, out, offset)
                    wp.release(1)
                op.release(1)
            xp.release(1)

    def pack(ip, lp, bp, op, fn):
        for _ in range_(layers):
            lut, bias, out = lp.acquire(1), bp.acquire(1), op.acquire(1)
            for head in range_(14):
                row = ip.acquire(1); fn(row, lut, bias, out, head, 0); ip.release(1)
            for head in range_(2):
                row = ip.acquire(1); fn(row, lut, bias, out, head, 1); ip.release(1)
            for head in range_(2):
                row = ip.acquire(1); fn(row, lut, bias, out, head, 2); ip.release(1)
            lp.release(1); bp.release(1); op.release(1)

    def attend(pp, cip, cop, op, fn):
        for _ in range_(layers):
            p, ci = pp.acquire(1), cip.acquire(1)
            co, out = cop.acquire(1), op.acquire(1)
            fn(p, ci, co, out)
            pp.release(1); cip.release(1); cop.release(1); op.release(1)

    def add64(pp, rp, op, fn):
        for _ in range_(layers):
            residual, out = rp.acquire(1), op.acquire(1)
            for block in range_(14):
                projection = pp.acquire(1)
                fn(projection, residual, out, block * 64)
                pp.release(1)
            rp.release(1); op.release(1)

    def activate(ip, op, store_fn, up_fn):
        for _ in range_(layers):
            out = op.acquire(1)
            for block in range_(76):
                gate = ip.acquire(1); store_fn(gate, out, block * 64); ip.release(1)
            for block in range_(76):
                up = ip.acquire(1); up_fn(up, out, block * 64); ip.release(1)
            op.release(1)

    def project32(wp, xp, op, fn):
        for _ in range_(layers):
            x = xp.acquire(1)
            for _ in range_(28):
                out = op.acquire(1)
                for offset in range_(0, 32, 4):
                    w = wp.acquire(1)
                    fn(w, x, out, offset)
                    wp.release(1)
                op.release(1)
            xp.release(1)

    def add32(pp, rp, op, fn):
        for _ in range_(layers):
            residual, out = rp.acquire(1), op.acquire(1)
            for block in range_(28):
                projection = pp.acquire(1)
                fn(projection, residual, out, block * 32)
                pp.release(1)
            rp.release(1); op.release(1)

    workers = [
        Worker(normalize, [finitial.cons(), fgq.cons(), fnorm_q.prod(), rms]),
        Worker(project64, [fwq.cons(), fnorm_q.cons(), fqkv_rows.prod(),
                           project896, 18]),
        Worker(pack, [fqkv_rows.cons(), flut.cons(), fbias.cons(),
                      fpacked.prod(), rope]),
    ]
    for head in range(2):
        workers.append(
            Worker(
                attend,
                [packed_heads[head].cons(), cache_in[head].cons(),
                 cache_out[head].prod(), attended_heads[head].prod(), attention],
            )
        )
    workers += [
        Worker(project64, [fwo.cons(), fatt_parent.cons(), fo_rows.prod(),
                           project896, 14]),
        Worker(add64, [fo_rows.cons(), finitial.cons(), fafter.prod(), residual64]),
        Worker(normalize, [fafter.cons(), fgp.cons(), fnorm_post.prod(), rms]),
        Worker(project64, [fwg.cons(), fnorm_post.cons(), fgate_rows.prod(),
                           project896, 152]),
        Worker(activate, [fgate_rows.cons(), fmlp.prod(), store_gate, swiglu_up]),
        Worker(project32, [fwd.cons(), fmlp.cons(), fdown_rows.prod(), project4864]),
        Worker(add32, [fdown_rows.cons(), fafter.cons(), ffinal.prod(), residual32]),
    ]

    rt = Runtime()
    with rt.sequence(weights_ty, params_ty, lut_ty, hidden_ty, cache_ty) as (
        weights, params, lut, hidden, cache
    ):
        rt.start(*workers)
        total_weights = layers * WEIGHT_BYTES

        q_base = 0
        o_base = layers * QKV_BYTES
        gate_base = o_base + layers * O_BYTES
        down_base = gate_base + layers * GATE_BYTES
        stream_tg = rt.task_group()
        def projection_tap(offset, blocks):
            return TensorAccessPattern(
                (total_weights,),
                offset,
                [layers, blocks, 16, HIDDEN],
                [16 * HIDDEN, layers * 16 * HIDDEN, HIDDEN, 1],
            )

        rt.fill(
            fwq.prod(), weights, projection_tap(q_base, QKV_BLOCKS),
            task_group=stream_tg, tile=Tile(0, 0),
        )
        rt.fill(
            fwo.prod(), weights, projection_tap(o_base, O_BLOCKS),
            task_group=stream_tg, tile=Tile(1, 0),
        )
        rt.fill(
            fwg.prod(), weights, projection_tap(gate_base, GATE_BLOCKS),
            task_group=stream_tg, tile=Tile(2, 0),
        )
        down_tap = TensorAccessPattern(
            (total_weights,),
            down_base,
            [layers, DOWN_BLOCKS * 4, 19, 256],
            [INTERMEDIATE, layers * INTERMEDIATE, 256, 1],
        )
        rt.fill(
            fwd.prod(), weights, down_tap,
            task_group=stream_tg, tile=Tile(3, 0),
        )
        params_tap = TensorAccessPattern(
            (layers * GAMMA_BIAS,),
            0,
            [layers, 32, 1, 128],
            [GAMMA_BIAS, 128, 0, 1],
        )
        lut_tap = TensorAccessPattern(
            (66,), 0, [layers, 1, 1, 66], [0, 0, 0, 1]
        )
        cache_tap = TensorAccessPattern(
            (layers * 2 * 8192,),
            0,
            [layers, 2, 32, 256],
            [2 * 8192, 8192, 256, 1],
        )
        rt.fill(
            fp_parent.prod(), params, params_tap,
            task_group=stream_tg, tile=Tile(0, 0),
        )
        rt.fill(
            flut.prod(), lut, lut_tap,
            task_group=stream_tg, tile=Tile(1, 0),
        )
        rt.fill(
            fcache_in_parent.prod(), cache, cache_tap,
            task_group=stream_tg, tile=Tile(3, 0),
        )
        for layer in range(layers):
            tg = rt.task_group()
            rt.fill(finitial.prod(), hidden, task_group=tg, tile=Tile(2, 0))
            rt.drain(
                ffinal.cons(), hidden, wait=True,
                task_group=tg, tile=Tile(0, 0),
            )
            rt.finish_task_group(tg)
        rt.drain(
            fcache_out_parent.cons(), cache, cache_tap,
            wait=True, task_group=stream_tg, tile=Tile(1, 0),
        )
        rt.finish_task_group(stream_tg)
    return Program(iron.get_current_device(), rt).resolve_program()
