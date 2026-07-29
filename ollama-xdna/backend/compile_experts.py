#!/usr/bin/env python3
"""Compile an 8-expert padded GEMV program for Hawk Point XDNA1.

Qwen3-Coder uses two expert shapes: 2048x768 for gate/up and 768x2048
for down.  The AIE program exposes one fixed 2048x2048 W8/BF16 shape and
processes the eight selected experts over four columns.  The host pads the
unused rows or columns.  Keeping one xclbin avoids reprogramming the NPU
between the three expert projections in every transformer layer.
"""

from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT
sys.path.insert(0, str(PROJECT))

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils import config
from aie.utils.hostruntime import set_current_device


KERNEL_SOURCE = PROJECT / "npu_llm" / "kernels" / "project_w8bf16.cc"
ROWS = 16
BATCH = 8
CORES = 2
EXPERTS_PER_CORE = BATCH // CORES
DMA_BLOCKS = 64


def linear_tap(length: int, offset: int, elements: int) -> TensorAccessPattern:
    return TensorAccessPattern(
        (length,), offset, [1, 1, 1, elements], [0, 0, 0, 1]
    )


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def expert_project(
    weights: In,
    scales: In,
    activations: In,
    output: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
):
    weight_ty = np.ndarray[(BATCH, M, K), np.dtype[np.int8]]
    scale_ty = np.ndarray[(BATCH, M), np.dtype[np.float32]]
    activation_ty = np.ndarray[(BATCH, K), np.dtype[bfloat16]]
    output_ty = np.ndarray[(BATCH, M), np.dtype[bfloat16]]
    weight_block_ty = np.ndarray[(ROWS, K), np.dtype[np.int8]]
    scale_block_ty = np.ndarray[(ROWS,), np.dtype[np.float32]]
    activation_block_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    accumulator_ty = np.ndarray[(ROWS,), np.dtype[np.float32]]
    output_block_ty = np.ndarray[(ROWS,), np.dtype[bfloat16]]

    accumulate = ExternalFunction(
        "project_w8bf16_acc32",
        source_file=str(KERNEL_SOURCE),
        arg_types=[weight_block_ty, activation_block_ty, accumulator_ty],
        include_dirs=[config.cxx_header_path()],
        compile_flags=[f"-DDIM_M={ROWS}", f"-DDIM_K={K}"],
    )
    dequantize = Kernel(
        "project_dequant_bf16",
        accumulate.object_file_name,
        [accumulator_ty, scale_block_ty, output_block_ty],
    )

    weight_fifos = []
    scale_fifos = []
    activation_fifos = []
    accumulator_fifos = []
    output_fifos = []
    workers = []

    chunks_per_expert = (M // ROWS) // DMA_BLOCKS

    def accumulate_worker(of_w, of_x, of_acc, fn):
        for _ in range_(EXPERTS_PER_CORE * chunks_per_expert):
            x = of_x.acquire(1)
            for _ in range_(DMA_BLOCKS):
                w = of_w.acquire(1)
                acc = of_acc.acquire(1)
                fn(w, x, acc)
                of_w.release(1)
                of_acc.release(1)
            of_x.release(1)

    def dequantize_worker(of_acc, of_s, of_out, fn):
        for _ in range_(EXPERTS_PER_CORE * chunks_per_expert * DMA_BLOCKS):
            acc = of_acc.acquire(1)
            scale = of_s.acquire(1)
            out = of_out.acquire(1)
            fn(acc, scale, out)
            of_acc.release(1)
            of_s.release(1)
            of_out.release(1)

    for core in range(CORES):
        weight_fifo = ObjectFifo(weight_block_ty, depth=1, name=f"weights_{core}")
        scale_fifo = ObjectFifo(scale_block_ty, depth=1, name=f"scales_{core}")
        activation_fifo = ObjectFifo(
            activation_block_ty, depth=1, name=f"activation_{core}"
        )
        accumulator_fifo = ObjectFifo(
            accumulator_ty, depth=1, name=f"accumulator_{core}"
        )
        output_fifo = ObjectFifo(output_block_ty, depth=1, name=f"output_{core}")
        weight_fifos.append(weight_fifo)
        scale_fifos.append(scale_fifo)
        activation_fifos.append(activation_fifo)
        accumulator_fifos.append(accumulator_fifo)
        output_fifos.append(output_fifo)
        workers.extend(
            [
                Worker(
                    accumulate_worker,
                    [
                        weight_fifo.cons(),
                        activation_fifo.cons(),
                        accumulator_fifo.prod(),
                        accumulate,
                    ],
                ),
                Worker(
                    dequantize_worker,
                    [
                        accumulator_fifo.cons(),
                        scale_fifo.cons(),
                        output_fifo.prod(),
                        dequantize,
                    ],
                ),
            ]
        )

    runtime = Runtime()
    with runtime.sequence(
        weight_ty, scale_ty, activation_ty, output_ty
    ) as (w, s, x, out):
        runtime.start(*workers)
        for expert_slot in range(EXPERTS_PER_CORE):
            for chunk in range(chunks_per_expert):
                start_row = chunk * DMA_BLOCKS * ROWS
                elements = DMA_BLOCKS * ROWS
                task_group = runtime.task_group()
                for core in range(CORES):
                    expert = core * EXPERTS_PER_CORE + expert_slot
                    weight_offset = (expert * M + start_row) * K
                    row_offset = expert * M + start_row
                    weight_tap = TensorAccessPattern(
                        (BATCH * M, K),
                        weight_offset,
                        [DMA_BLOCKS, 1, 1, ROWS * K],
                        [ROWS * K, 0, 0, 1],
                    )
                    runtime.fill(
                        weight_fifos[core].prod(),
                        w,
                        weight_tap,
                        task_group=task_group,
                    )
                    runtime.fill(
                        scale_fifos[core].prod(),
                        s,
                        linear_tap(BATCH * M, row_offset, elements),
                        task_group=task_group,
                    )
                    runtime.fill(
                        activation_fifos[core].prod(),
                        x,
                        linear_tap(BATCH * K, expert * K, K),
                        task_group=task_group,
                    )
                    runtime.drain(
                        output_fifos[core].cons(),
                        out,
                        linear_tap(BATCH * M, row_offset, elements),
                        wait=True,
                        task_group=task_group,
                    )
                runtime.finish_task_group(task_group)

    return Program(iron.get_current_device(), runtime).resolve_program()


def main() -> None:
    output = Path(__file__).resolve().parent / "artifacts" / "experts-8x2048x2048"
    output.mkdir(parents=True, exist_ok=True)
    set_current_device(from_name("npu", n_cols=4))
    design = expert_project.specialize(M=2048, K=2048)
    design.compile(
        xclbin_path=output / "experts.xclbin",
        inst_path=output / "insts.bin",
    )
    print(output)


if __name__ == "__main__":
    main()
