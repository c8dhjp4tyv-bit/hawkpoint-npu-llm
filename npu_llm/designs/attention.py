"""NPU-only KV append and grouped-query attention for SmolLM2."""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorTiler2D
from aie.iron import CompileTime, In, InOut, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile, from_name
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark, assert_pass


ROOT = Path(__file__).resolve().parents[1]
KV_SRC = ROOT / "kernels/kv_append_bf16.cc"
ATTN_SRC = ROOT / "kernels/attention_bf16.cc"
KV_HEADS = 3
Q_PER_KV = 3
HEAD_DIM = 64
CONTEXT = 64


def _slice_taps(width):
    return TensorTiler2D.simple_tiler(
        (KV_HEADS, width), (1, width), prune_step=False
    )


@iron.jit
def kv_append(Cache: InOut, Value: In, *, position: CompileTime[int]):
    cache_ty = np.ndarray[(KV_HEADS, CONTEXT * HEAD_DIM), np.dtype[bfloat16]]
    value_ty = np.ndarray[(KV_HEADS, HEAD_DIM), np.dtype[bfloat16]]
    cache_slice = np.ndarray[(CONTEXT * HEAD_DIM,), np.dtype[bfloat16]]
    value_slice = np.ndarray[(HEAD_DIM,), np.dtype[bfloat16]]
    fn = ExternalFunction(
        "kv_append_bf16",
        source_file=str(KV_SRC),
        arg_types=[cache_slice, value_slice, np.int32, cache_slice],
        include_dirs=[config.cxx_header_path()],
    )
    cache_in, value_in, cache_out, workers = [], [], [], []
    for i in range(KV_HEADS):
        ci = ObjectFifo(cache_slice, name=f"cache_in_{i}", depth=1)
        vi = ObjectFifo(value_slice, name=f"value_{i}", depth=1)
        co = ObjectFifo(cache_slice, name=f"cache_out_{i}", depth=1)
        cache_in.append(ci)
        value_in.append(vi)
        cache_out.append(co)

        def core_fn(of_cache, of_value, of_out, kernel):
            cache = of_cache.acquire(1)
            value = of_value.acquire(1)
            out = of_out.acquire(1)
            kernel(cache, value, position, out)
            of_cache.release(1)
            of_value.release(1)
            of_out.release(1)

        workers.append(
            Worker(
                core_fn,
                [ci.cons(), vi.cons(), co.prod(), fn],
            )
        )

    rt = Runtime()
    with rt.sequence(cache_ty, value_ty) as (cache, value):
        rt.start(*workers)
        for i, (ct, vt) in enumerate(
            zip(_slice_taps(CONTEXT * HEAD_DIM), _slice_taps(HEAD_DIM))
        ):
            rt.fill(cache_in[i].prod(), cache, ct, tile=Tile(i, 0))
            rt.fill(value_in[i].prod(), value, vt, tile=Tile(i, 0))
            rt.drain(
                cache_out[i].cons(), cache, ct, wait=True, tile=Tile(i, 0)
            )
    return Program(iron.get_current_device(), rt).resolve_program()


def _attention_design(symbol, position):
    q_ty = np.ndarray[(KV_HEADS, Q_PER_KV * HEAD_DIM), np.dtype[bfloat16]]
    cache_ty = np.ndarray[(KV_HEADS, CONTEXT * HEAD_DIM), np.dtype[bfloat16]]
    out_ty = np.ndarray[(KV_HEADS, Q_PER_KV * HEAD_DIM), np.dtype[bfloat16]]
    q_slice = np.ndarray[(Q_PER_KV * HEAD_DIM,), np.dtype[bfloat16]]
    cache_slice = np.ndarray[(CONTEXT * HEAD_DIM,), np.dtype[bfloat16]]
    out_slice = np.ndarray[(Q_PER_KV * HEAD_DIM,), np.dtype[bfloat16]]
    fn = ExternalFunction(
        symbol,
        source_file=str(ATTN_SRC),
        arg_types=[q_slice, cache_slice, np.int32, out_slice],
        include_dirs=[config.cxx_header_path()],
    )
    a_fifos, cache_fifos, c_fifos, workers = [], [], [], []
    for i in range(KV_HEADS):
        af = ObjectFifo(q_slice, name=f"a_{i}", depth=1)
        cf = ObjectFifo(cache_slice, name=f"cache_{i}", depth=1)
        of = ObjectFifo(out_slice, name=f"out_{i}", depth=1)
        a_fifos.append(af)
        cache_fifos.append(cf)
        c_fifos.append(of)

        def core_fn(of_a, of_cache, of_c, kernel):
            a = of_a.acquire(1)
            cache = of_cache.acquire(1)
            c = of_c.acquire(1)
            kernel(a, cache, position, c)
            of_a.release(1)
            of_cache.release(1)
            of_c.release(1)

        workers.append(
            Worker(
                core_fn,
                [af.cons(), cf.cons(), of.prod(), fn],
            )
        )
    rt = Runtime()
    with rt.sequence(q_ty, cache_ty, out_ty) as (a, cache, c):
        rt.start(*workers)
        for i, (at, ct) in enumerate(
            zip(_slice_taps(Q_PER_KV * HEAD_DIM), _slice_taps(CONTEXT * HEAD_DIM))
        ):
            rt.fill(a_fifos[i].prod(), a, at, tile=Tile(i, 0))
            rt.fill(cache_fifos[i].prod(), cache, ct, tile=Tile(i, 0))
            rt.drain(c_fifos[i].cons(), c, at, wait=True, tile=Tile(i, 0))
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def attention_scores(Q: In, KCache: In, P: Out, *, position: CompileTime[int]):
    return _attention_design("attention_scores_bf16", position)


@iron.jit
def attention_values(P: In, VCache: In, O: Out, *, position: CompileTime[int]):
    return _attention_design("attention_values_bf16", position)


def _parser():
    p = argparse.ArgumentParser()
    add_compile_args(p, short_dev=None)
    p.add_argument("--position", type=int, default=7)
    p.add_argument("--emit-mlir", action="store_true")
    add_benchmark_args(p)
    return p


def _run(o):
    rng = np.random.default_rng(13)
    q = rng.normal(0, 0.3, (KV_HEADS, Q_PER_KV * HEAD_DIM)).astype(bfloat16)
    knew = rng.normal(0, 0.3, (KV_HEADS, HEAD_DIM)).astype(bfloat16)
    vnew = rng.normal(0, 0.3, (KV_HEADS, HEAD_DIM)).astype(bfloat16)
    kc_np = rng.normal(
        0, 0.3, (KV_HEADS, CONTEXT, HEAD_DIM)
    ).astype(bfloat16)
    vc_np = rng.normal(
        0, 0.3, (KV_HEADS, CONTEXT, HEAD_DIM)
    ).astype(bfloat16)
    pos_np = np.array([o.position], np.int32)
    Q = iron.tensor(q, dtype=bfloat16, device="npu")
    KN = iron.tensor(knew, dtype=bfloat16, device="npu")
    VN = iron.tensor(vnew, dtype=bfloat16, device="npu")
    KC = iron.tensor(kc_np, dtype=bfloat16, device="npu")
    VC = iron.tensor(vc_np, dtype=bfloat16, device="npu")
    P = iron.zeros(q.shape, dtype=bfloat16, device="npu")
    O = iron.zeros(q.shape, dtype=bfloat16, device="npu")
    kv_append(KC, KN, position=o.position)
    kv_append(VC, VN, position=o.position)
    assert_pass(KC.numpy().reshape(KV_HEADS, CONTEXT, HEAD_DIM)[:, o.position], knew)
    assert_pass(VC.numpy().reshape(KV_HEADS, CONTEXT, HEAD_DIM)[:, o.position], vnew)
    bench = run_iters(
        attention_scores,
        Q,
        KC,
        P,
        position=o.position,
        warmup=o.warmup,
        iters=o.iters,
    )
    attention_values(P, VC, O, position=o.position)

    kc = KC.numpy().astype(np.float32).reshape(KV_HEADS, CONTEXT, HEAD_DIM)
    vc = VC.numpy().astype(np.float32).reshape(KV_HEADS, CONTEXT, HEAD_DIM)
    expected = np.empty_like(q, dtype=np.float32).reshape(KV_HEADS, Q_PER_KV, HEAD_DIM)
    for kv_head in range(KV_HEADS):
        for local_head in range(Q_PER_KV):
            query = q[kv_head, local_head * HEAD_DIM : (local_head + 1) * HEAD_DIM].astype(np.float32)
            scores = kc[kv_head, : o.position + 1] @ query / 8.0
            scores -= scores.max()
            probability = np.exp(scores)
            probability /= probability.sum()
            expected[kv_head, local_head] = probability @ vc[kv_head, : o.position + 1]
    assert_close_with_benchmark(
        O.numpy(),
        expected.reshape(KV_HEADS, Q_PER_KV * HEAD_DIM).astype(bfloat16),
        bench=bench,
        float_rtol=0.15,
        float_atol=0.08,
        fail_msg="grouped-query attention mismatch",
    )


def main():
    o = _parser().parse_args()
    run_design_cli(
        attention_scores,
        o,
        compile_kwargs={"position": o.position},
        run_and_verify=_run,
        device=lambda x: from_name(x.dev, n_cols=4),
    )


if __name__ == "__main__":
    main()
