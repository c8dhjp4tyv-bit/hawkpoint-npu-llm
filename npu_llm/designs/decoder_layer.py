"""One persistent xclbin for a complete SmolLM2 decoder layer."""

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


SRC = Path(__file__).resolve().parents[1] / "kernels/decoder_layer_bf16.cc"
QKV_BLOCK = 32 * 576
DOWN_BLOCK = 16 * 1536
QKV_BLOCKS, O_BLOCKS, GATE_BLOCKS, DOWN_BLOCKS = 30, 18, 96, 36
QKV_BYTES = QKV_BLOCKS * QKV_BLOCK
O_BYTES = O_BLOCKS * QKV_BLOCK
GATE_BYTES = GATE_BLOCKS * QKV_BLOCK
DOWN_BYTES = DOWN_BLOCKS * DOWN_BLOCK
WEIGHT_BYTES = QKV_BYTES + O_BYTES + GATE_BYTES + DOWN_BYTES

# The same kernels also expose an int8-weight/BF16-activation path.  A block
# carries the per-output-channel float32 scales immediately after its int8
# rows.  Keeping these constants next to the BF16 layout makes the host packer
# and the IRON taps auditable instead of relying on magic byte offsets.
QKV_W8_ROWS, DOWN_W8_ROWS = 64, 32
QKV_W8_BLOCK = QKV_W8_ROWS * 576 + QKV_W8_ROWS * 4
DOWN_W8_BLOCK = DOWN_W8_ROWS * 1536 + DOWN_W8_ROWS * 4
QKV_W8_BLOCKS, O_W8_BLOCKS, GATE_W8_BLOCKS, DOWN_W8_BLOCKS = 15, 9, 48, 18
QKV_W8_BYTES = QKV_W8_BLOCKS * QKV_W8_BLOCK
O_W8_BYTES = O_W8_BLOCKS * QKV_W8_BLOCK
GATE_W8_BYTES = GATE_W8_BLOCKS * QKV_W8_BLOCK
DOWN_W8_BYTES = DOWN_W8_BLOCKS * DOWN_W8_BLOCK
W8_WEIGHT_BYTES = QKV_W8_BYTES + O_W8_BYTES + GATE_W8_BYTES + DOWN_W8_BYTES

# The gate/up projection is the largest serialized stage (96 blocks). Three
# independent workers consume disjoint weight streams while sharing the
# normalized activation; a small merge worker restores row order for SwiGLU.
# This stays within the XDNA1 compute-peer/DMA budget alongside the pipeline
# workers.
GATE_PARALLEL = 3
GATE_OUTPUT_BLOCKS_PER_WORKER = (GATE_BLOCKS // 2) // GATE_PARALLEL


def _linear_tap(total, offset, blocks, block):
    return TensorAccessPattern(
        (total,), offset, [blocks, 1, 1, block], [block, 0, 0, 1]
    )


@iron.jit(
    aiecc_flags=["--alloc-scheme=basic-sequential"],
    compile_flags=["-O3", "-ffast-math"],
)
def decoder_layer(
    PackedWeights: In,
    Gammas: In,
    LUT: In,
    Hidden: InOut,
    Cache: InOut,
    *,
    layers: CompileTime[int] = 1,
    quantized: CompileTime[bool] = False,
):
    weight_dtype = np.uint8 if quantized else bfloat16
    qkv_block = QKV_W8_BLOCK if quantized else QKV_BLOCK
    down_block = DOWN_W8_BLOCK if quantized else DOWN_BLOCK
    qkv_blocks = 15
    o_blocks = 9
    gate_blocks = 48
    down_blocks = 18
    qkv_weight_blocks = QKV_W8_BLOCKS if quantized else 30
    o_weight_blocks = O_W8_BLOCKS if quantized else 18
    gate_weight_blocks = GATE_W8_BLOCKS if quantized else 96
    down_weight_blocks = DOWN_W8_BLOCKS if quantized else 36
    qkv_bytes = qkv_weight_blocks * qkv_block
    o_bytes = o_weight_blocks * qkv_block
    gate_bytes = gate_weight_blocks * qkv_block
    down_bytes = down_weight_blocks * down_block
    weight_bytes = qkv_bytes + o_bytes + gate_bytes + down_bytes
    weights_ty = np.ndarray[(layers * weight_bytes,), np.dtype[weight_dtype]]
    gammas_ty = np.ndarray[(layers * 1152,), np.dtype[bfloat16]]
    gamma_layer_ty = np.ndarray[(1152,), np.dtype[bfloat16]]
    lut_ty = np.ndarray[(66,), np.dtype[bfloat16]]
    hidden_ty = np.ndarray[(576,), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(layers, 3, 8192), np.dtype[bfloat16]]
    cache_layer_ty = np.ndarray[(3, 8192), np.dtype[bfloat16]]
    row64 = np.ndarray[(64,), np.dtype[bfloat16]]
    row32 = np.ndarray[(32,), np.dtype[bfloat16]]
    block576 = np.ndarray[(qkv_block,), np.dtype[weight_dtype]]
    block1536 = np.ndarray[(down_block,), np.dtype[weight_dtype]]
    gate_weight_group = np.ndarray[
        (GATE_PARALLEL * qkv_block,), np.dtype[weight_dtype]
    ]
    packed1008 = np.ndarray[(1008,), np.dtype[bfloat16]]
    packed336 = np.ndarray[(336,), np.dtype[bfloat16]]
    cache8192 = np.ndarray[(8192,), np.dtype[bfloat16]]
    attended192 = np.ndarray[(192,), np.dtype[bfloat16]]
    mlp1536 = np.ndarray[(1536,), np.dtype[bfloat16]]

    rms = ExternalFunction(
        "layer_rmsnorm576", source_file=str(SRC),
        arg_types=[hidden_ty, hidden_ty, hidden_ty],
        include_dirs=[
            config.cxx_header_path(),
            str(Path(config.root_path()) / "aie_runtime_lib/AIE2"),
        ],
    )
    obj = rms.object_file_name
    copy576 = Kernel("layer_copy576", obj, [hidden_ty, hidden_ty])
    copy64 = Kernel("layer_copy64", obj, [row64, row64])
    if quantized:
        proj576 = Kernel(
            "layer_project64_k576",
            obj, [block576, hidden_ty, row64]
        )
        proj1536 = Kernel(
            "layer_project32_k1536",
            obj, [block1536, mlp1536, row32]
        )
    else:
        proj576 = Kernel(
            "layer_project64_k576_pair_bf16",
            obj, [block576, hidden_ty, row64, np.int32]
        )
        proj1536 = Kernel(
            "layer_project16_k1536_bf16",
            obj, [block1536, mlp1536, row32, np.int32]
        )
    rope = Kernel(
        "layer_pack_rope64", obj,
        [row64, lut_ty, packed1008, np.int32, np.int32]
    )
    attn = Kernel(
        "layer_attention", obj,
        [packed336, cache8192, cache8192, attended192],
    )
    res64 = Kernel(
        "layer_residual64", obj, [row64, hidden_ty, hidden_ty, np.int32]
    )
    gate_store = Kernel(
        "layer_store_gate64", obj, [row64, mlp1536, np.int32]
    )
    swiglu_up = Kernel(
        "layer_swiglu_up64", obj, [row64, mlp1536, np.int32]
    )
    res32 = Kernel(
        "layer_residual32", obj, [row32, hidden_ty, hidden_ty, np.int32]
    )

    # Host-facing FIFOs.
    fwq = ObjectFifo(block576, depth=1, name="w_qkv")
    fwo = ObjectFifo(block576, depth=1, name="w_o")
    fgate_weight_parent = ObjectFifo(
        gate_weight_group, depth=1, name="w_gate_group"
    )
    fgate_weights = fgate_weight_parent.cons().split(
        [i * qkv_block for i in range(GATE_PARALLEL)],
        obj_types=[block576] * GATE_PARALLEL,
        names=[f"w_gate_{i}" for i in range(GATE_PARALLEL)],
        depths=[1] * GATE_PARALLEL,
    )
    fwd = ObjectFifo(block1536, depth=1, name="w_down")
    finitial = ObjectFifo(hidden_ty, depth=1, name="hidden_initial")
    ffinal = ObjectFifo(hidden_ty, depth=1, name="hidden_final")
    fg_parent = ObjectFifo(gamma_layer_ty, depth=1, name="gammas")
    fgq, fgp = fg_parent.cons().split(
        [0, 576], obj_types=[hidden_ty, hidden_ty],
        names=["gamma_q", "gamma_post"], depths=[1, 1],
    )
    flut = ObjectFifo(lut_ty, depth=1, name="lut")
    fcache_in_parent = ObjectFifo(cache_layer_ty, depth=1, name="cache_in")
    cache_in = fcache_in_parent.cons().split(
        [0, 8192, 16384], obj_types=[cache8192] * 3,
        names=["cache_in_0", "cache_in_1", "cache_in_2"], depths=[1] * 3,
    )
    fcache_out_parent = ObjectFifo(cache_layer_ty, depth=1, name="cache_out")
    cache_out = fcache_out_parent.prod().join(
        [0, 8192, 16384], obj_types=[cache8192] * 3,
        names=["cache_out_0", "cache_out_1", "cache_out_2"], depths=[1] * 3,
    )

    # Internal streaming graph.
    fnorm_q = ObjectFifo(hidden_ty, depth=1, name="norm_q")
    fqkv_rows = ObjectFifo(row64, depth=1, name="qkv_rows")
    fpacked = ObjectFifo(packed1008, depth=1, name="packed_qkv")
    packed_heads = fpacked.cons().split(
        [0, 336, 672], obj_types=[packed336] * 3,
        names=["packed_0", "packed_1", "packed_2"], depths=[1] * 3,
    )
    fatt_parent = ObjectFifo(hidden_ty, depth=1, name="attended")
    attended_heads = fatt_parent.prod().join(
        [0, 192, 384], obj_types=[attended192] * 3,
        names=["attended_0", "attended_1", "attended_2"], depths=[1] * 3,
    )
    fo_rows = ObjectFifo(row64, depth=1, name="o_rows")
    fafter = ObjectFifo(hidden_ty, depth=1, name="after_attention")
    fnorm_post = ObjectFifo(hidden_ty, depth=1, name="norm_post")
    fgate_rows = ObjectFifo(row64, depth=1, name="gate_rows")
    fgate_part_rows = [
        ObjectFifo(row64, depth=1, name=f"gate_rows_{i}")
        for i in range(GATE_PARALLEL)
    ]
    fmlp = ObjectFifo(mlp1536, depth=1, name="mlp")
    fdown_rows = ObjectFifo(row32, depth=1, name="down_rows")
    def normalize(xp, gp, op, fn):
        for _ in range_(layers):
            x, g, out = xp.acquire(1), gp.acquire(1), op.acquire(1)
            fn(x, g, out)
            xp.release(1); gp.release(1); op.release(1)

    def project64(wp, xp, op, fn, blocks):
        for _ in range_(layers):
            x = xp.acquire(1)
            for _ in range_(blocks):
                out = op.acquire(1)
                w = wp.acquire(1)
                if quantized:
                    fn(w, x, out)
                else:
                    fn(w, x, out, 0)
                wp.release(1)
                if not quantized:
                    w = wp.acquire(1); fn(w, x, out, 32); wp.release(1)
                op.release(1)
            xp.release(1)

    def pack_rope(ip, lp, op, fn):
        for _ in range_(layers):
            lut, out = lp.acquire(1), op.acquire(1)
            for head in range_(9):
                row = ip.acquire(1); fn(row, lut, out, head, 0); ip.release(1)
            for head in range_(3):
                row = ip.acquire(1); fn(row, lut, out, head, 1); ip.release(1)
            for head in range_(3):
                row = ip.acquire(1); fn(row, lut, out, head, 2); ip.release(1)
            lp.release(1); op.release(1)

    def attention(pp, cip, cop, op, fn):
        for _ in range_(layers):
            p = pp.acquire(1); ci = cip.acquire(1)
            co = cop.acquire(1); out = op.acquire(1)
            fn(p, ci, co, out)
            pp.release(1); cip.release(1); cop.release(1); op.release(1)

    def residual64(pp, rp, op, fn):
        for _ in range_(layers):
            residual, out = rp.acquire(1), op.acquire(1)
            for block in range_(9):
                projection = pp.acquire(1)
                fn(projection, residual, out, block * 64)
                pp.release(1)
            rp.release(1); op.release(1)

    def activate(ip, op, store_fn, up_fn):
        for _ in range_(layers):
            out = op.acquire(1)
            for block in range_(24):
                gate = ip.acquire(1); store_fn(gate, out, block * 64); ip.release(1)
            for block in range_(24):
                up = ip.acquire(1); up_fn(up, out, block * 64); ip.release(1)
            op.release(1)

    def project32(wp, xp, op, fn):
        for _ in range_(layers):
            x = xp.acquire(1)
            for _ in range_(down_blocks):
                out = op.acquire(1)
                w = wp.acquire(1)
                if quantized:
                    fn(w, x, out)
                else:
                    fn(w, x, out, 0)
                wp.release(1)
                if not quantized:
                    w = wp.acquire(1); fn(w, x, out, 16); wp.release(1)
                op.release(1)
            xp.release(1)

    def merge_gate_rows(p0, p1, p2, op, fn):
        # Each producer owns a round-robin block stream. Interleave rows in
        # worker order so the downstream activation sees the original order.
        parts = (p0, p1, p2)
        for _ in range_(layers):
            for _ in range_(GATE_OUTPUT_BLOCKS_PER_WORKER):
                for part in parts:
                    src, dst = part.acquire(1), op.acquire(1)
                    fn(src, dst)
                    part.release(1); op.release(1)

    def residual32(pp, rp, op, fn):
        for _ in range_(layers):
            residual, out = rp.acquire(1), op.acquire(1)
            for block in range_(18):
                projection = pp.acquire(1)
                fn(projection, residual, out, block * 32)
                pp.release(1)
            rp.release(1); op.release(1)

    def route_hidden(initial_port, feedback_port, current_port, final_port, fn):
        initial = initial_port.acquire(1)
        current = current_port.acquire(1)
        fn(initial, current)
        initial_port.release(1); current_port.release(1)
        for _ in range(layers - 1):
            feedback = feedback_port.acquire(1)
            current = current_port.acquire(1)
            fn(feedback, current)
            feedback_port.release(1); current_port.release(1)
        feedback = feedback_port.acquire(1)
        final = final_port.acquire(1)
        fn(feedback, final)
        feedback_port.release(1); final_port.release(1)

    workers = [
        Worker(normalize, [finitial.cons(), fgq.cons(), fnorm_q.prod(), rms]),
        Worker(project64, [fwq.cons(), fnorm_q.cons(), fqkv_rows.prod(), proj576, qkv_blocks]),
        Worker(pack_rope, [fqkv_rows.cons(), flut.cons(), fpacked.prod(), rope]),
    ]
    for head in range(3):
        workers.append(Worker(
            attention,
            [packed_heads[head].cons(), cache_in[head].cons(),
             cache_out[head].prod(), attended_heads[head].prod(), attn],
        ))
    workers += [
        Worker(project64, [fwo.cons(), fatt_parent.cons(), fo_rows.prod(), proj576, o_blocks]),
        Worker(residual64, [fo_rows.cons(), finitial.cons(), fafter.prod(), res64]),
        Worker(normalize, [fafter.cons(), fgp.cons(), fnorm_post.prod(), rms]),
        *[
            Worker(
                project64,
                [fgate_weights[i].cons(), fnorm_post.cons(),
                 fgate_part_rows[i].prod(), proj576,
                 gate_blocks // GATE_PARALLEL],
            )
            for i in range(GATE_PARALLEL)
        ],
        Worker(
            merge_gate_rows,
            [
                fgate_part_rows[0].cons(), fgate_part_rows[1].cons(),
                fgate_part_rows[2].cons(), fgate_rows.prod(), copy64,
            ],
        ),
        Worker(activate, [fgate_rows.cons(), fmlp.prod(), gate_store, swiglu_up]),
        Worker(project32, [fwd.cons(), fmlp.cons(), fdown_rows.prod(), proj1536]),
        Worker(residual32, [fdown_rows.cons(), fafter.cons(), ffinal.prod(), res32]),
    ]

    rt = Runtime()
    with rt.sequence(weights_ty, gammas_ty, lut_ty, hidden_ty, cache_ty) as (
        weights, gammas, lut, hidden, cache
    ):
        rt.start(*workers)
        stream_tg = rt.task_group()
        total_weights = layers * weight_bytes
        def layer_tap(offset, blocks, block, layer_stride):
            return TensorAccessPattern(
                (total_weights,), offset,
                [layers, blocks, 1, block],
                [layer_stride, block, 0, 1],
            )

        q_base = 0
        o_base = layers * qkv_bytes
        gate_base = o_base + layers * o_bytes
        down_base = gate_base + layers * gate_bytes
        rt.fill(
            fwq.prod(), weights,
            layer_tap(q_base, qkv_weight_blocks, qkv_block, qkv_bytes),
            task_group=stream_tg,
            tile=Tile(0, 0),
        )
        rt.fill(
            fwo.prod(), weights,
            layer_tap(o_base, o_weight_blocks, qkv_block, o_bytes),
            task_group=stream_tg,
            tile=Tile(1, 0),
        )
        gate_tap = TensorAccessPattern(
            (total_weights,), gate_base,
            [layers, gate_weight_blocks // GATE_PARALLEL, GATE_PARALLEL, qkv_block],
            [
                gate_bytes,
                GATE_PARALLEL * qkv_block,
                qkv_block,
                1,
            ],
        )
        rt.fill(
            fgate_weight_parent.prod(), weights, gate_tap,
            task_group=stream_tg, tile=Tile(2, 0),
        )
        rt.fill(
            fwd.prod(), weights,
            layer_tap(down_base, down_weight_blocks, down_block, down_bytes),
            task_group=stream_tg,
            tile=Tile(3, 0),
        )
        gamma_tap = TensorAccessPattern(
            (layers * 1152,), 0, [layers, 1, 1, 1152], [1152, 0, 0, 1]
        )
        lut_tap = TensorAccessPattern(
            (66,), 0, [layers, 1, 1, 66], [0, 0, 0, 1]
        )
        cache_tap = TensorAccessPattern(
            (layers * 3 * 8192,), 0,
            [layers, 1, 1, 3 * 8192], [3 * 8192, 0, 0, 1],
        )
        rt.fill(
            fg_parent.prod(), gammas, gamma_tap,
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
        for _ in range(layers):
            tg = rt.task_group()
            rt.fill(
                finitial.prod(), hidden,
                task_group=tg, tile=Tile(2, 0),
            )
            rt.drain(
                ffinal.cons(), hidden,
                wait=True, task_group=tg, tile=Tile(0, 0),
            )
            rt.finish_task_group(tg)
        rt.drain(
            fcache_out_parent.cons(), cache, cache_tap,
            wait=True, task_group=stream_tg, tile=Tile(1, 0),
        )
        rt.finish_task_group(stream_tg)
    return Program(iron.get_current_device(), rt).resolve_program()
