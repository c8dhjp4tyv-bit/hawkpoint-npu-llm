#!/usr/bin/env python3
"""Compile the native Q4_K/Q6_K expert program for Hawk Point XDNA1.

Unlike ``compile_experts.py`` this program receives GGML's quantized blocks
unchanged.  The AIE kernel performs the block dequantization while doing the
GEMV, so the corresponding XRT weight BO can remain resident between tokens.
The generated xclbin is intentionally separate from the validated W8 artifact:
it must be rebuilt and acceptance-tested on the exact MLIR-AIE/XRT stack before
being selected by the Ollama backend.
"""

from __future__ import annotations

import argparse
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
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.hostruntime import set_current_device


ROWS = 16
BATCH = 8
CORES = 2
EXPERTS_PER_CORE = BATCH // CORES
DMA_BLOCKS = 64
QK_K = 256
# The GGML payloads are 144 and 210 bytes.  The AIE DMA stream uses a 32-byte
# padded stride; the kernel reads only the payload fields and skips the tail.
BLOCK_BYTES = {"q4_k": 160, "q6_k": 224}
KERNEL_NAMES = {"q4_k": "q4k", "q6_k": "q6k"}


def linear_tap(length: int, offset: int, elements: int) -> TensorAccessPattern:
    return TensorAccessPattern(
        (length,), offset, [1, 1, 1, elements], [0, 0, 0, 1]
    )


def quantized_experts(quant: str):
    block_bytes = BLOCK_BYTES[quant]
    kernel_name = KERNEL_NAMES[quant]
    kernel_source = PROJECT / "npu_llm" / "kernels" / f"project_{kernel_name}_bf16.cc"
    function_name = f"project_{kernel_name}_bf16"

    @iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
    def expert_project(
        weights: In,
        activations: In,
        output: Out,
        *,
        M: CompileTime[int],
        K: CompileTime[int],
    ):
        blocks_per_row = K // QK_K
        row_bytes = blocks_per_row * block_bytes
        weight_ty = np.ndarray[(BATCH, M, row_bytes), np.dtype[np.uint8]]
        activation_ty = np.ndarray[(BATCH, K), np.dtype[bfloat16]]
        output_ty = np.ndarray[(BATCH, M), np.dtype[bfloat16]]
        weight_block_ty = np.ndarray[(ROWS, row_bytes), np.dtype[np.uint8]]
        activation_block_ty = np.ndarray[(K,), np.dtype[bfloat16]]
        output_block_ty = np.ndarray[(ROWS,), np.dtype[bfloat16]]

        project = ExternalFunction(
            function_name,
            source_file=str(kernel_source),
            arg_types=[weight_block_ty, activation_block_ty, output_block_ty],
            include_dirs=[config.cxx_header_path()],
            compile_flags=[f"-DDIM_M={ROWS}", f"-DDIM_K={K}"],
        )

        weight_fifos = []
        activation_fifos = []
        output_fifos = []
        workers = []

        chunks_per_expert = (M // ROWS) // DMA_BLOCKS

        def project_worker(of_w, of_x, of_out, fn):
            for _ in range_(EXPERTS_PER_CORE * chunks_per_expert):
                x = of_x.acquire(1)
                for _ in range_(DMA_BLOCKS):
                    w = of_w.acquire(1)
                    out = of_out.acquire(1)
                    fn(w, x, out)
                    of_w.release(1)
                    of_out.release(1)
                of_x.release(1)

        for core in range(CORES):
            weight_fifo = ObjectFifo(
                weight_block_ty, depth=1, name=f"qweights_{quant}_{core}"
            )
            activation_fifo = ObjectFifo(
                activation_block_ty, depth=1, name=f"qactivation_{quant}_{core}"
            )
            output_fifo = ObjectFifo(
                output_block_ty, depth=1, name=f"qoutput_{quant}_{core}"
            )
            weight_fifos.append(weight_fifo)
            activation_fifos.append(activation_fifo)
            output_fifos.append(output_fifo)
            workers.append(
                Worker(
                    project_worker,
                    [
                        weight_fifo.cons(),
                        activation_fifo.cons(),
                        output_fifo.prod(),
                        project,
                    ],
                )
            )

        runtime = Runtime()
        with runtime.sequence(weight_ty, activation_ty, output_ty) as (w, x, out):
            runtime.start(*workers)
            for expert_slot in range(EXPERTS_PER_CORE):
                for chunk in range(chunks_per_expert):
                    start_row = chunk * DMA_BLOCKS * ROWS
                    elements = DMA_BLOCKS * ROWS
                    task_group = runtime.task_group()
                    for core in range(CORES):
                        expert = core * EXPERTS_PER_CORE + expert_slot
                        weight_offset = (expert * M + start_row) * row_bytes
                        weight_tap = TensorAccessPattern(
                            (BATCH * M, row_bytes),
                            weight_offset,
                            [DMA_BLOCKS, 1, 1, ROWS * row_bytes],
                            [ROWS * row_bytes, 0, 0, 1],
                        )
                        runtime.fill(
                            weight_fifos[core].prod(),
                            w,
                            weight_tap,
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
                            linear_tap(BATCH * M, expert * M + start_row, elements),
                            wait=True,
                            task_group=task_group,
                        )
                    runtime.finish_task_group(task_group)

        return Program(iron.get_current_device(), runtime).resolve_program()

    return expert_project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=sorted(BLOCK_BYTES), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: backend/artifacts/experts-<quant>-8x2048x2048)",
    )
    args = parser.parse_args()

    output = args.output or (
        Path(__file__).resolve().parent
        / "artifacts"
        / f"experts-{args.quant}-8x2048x2048"
    )
    output.mkdir(parents=True, exist_ok=True)
    set_current_device(from_name("npu", n_cols=4))
    design = quantized_experts(args.quant).specialize(M=2048, K=2048)
    design.compile(
        xclbin_path=output / "experts.xclbin",
        inst_path=output / "insts.bin",
    )
    print(output)


if __name__ == "__main__":
    main()
