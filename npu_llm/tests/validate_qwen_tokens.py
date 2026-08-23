#!/usr/bin/env python3
"""Compare an autoregressive Qwen token sequence on CPU BF16 and XDNA1."""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from npu_llm.runtime.generate import NPUDecoder  # noqa: E402


PROMPT = [
    {
        "role": "user",
        "content": (
            "Explain why the sky appears blue during the day in one detailed "
            "paragraph with at least five sentences."
        ),
    }
]
# Pinned CPU BF16 reference for PROMPT under the production prompt path
# (NPUDecoder.prompt_ids -> encode_chat_within). It was captured while this
# script sliced the encoded ChatML stream itself, so it MUST be re-captured
# and deliberately reviewed on the pinned Hawk Point machine now that the
# retained prompt is a complete turn. A mismatch here is a real signal:
# re-record it from a hardware run, never edit it to match a failing run.
EXPECTED_CPU_BF16_TOKENS = [
    785,
    12884,
    7952,
    6303,
    2337,
    279,
    1899,
    4152,
    311,
    279,
    71816,
    315,
    39020,
    553,
    13673,
    3015,
    6973,
    89492,
    304,
    279,
    9237,
    594,
    16566,
    13,
    4220,
    6973,
    89492,
    11,
    892,
    525,
    13673,
    9853,
]


def sequence(model_dir, npu_layers, count):
    """Run the pinned prompt through one placement and return its tokens.

    The prompt is prepared with ``NPUDecoder.prompt_ids()`` -- the same call
    ``generate_messages()`` makes -- so this gate exercises the production
    trimming semantics instead of slicing the encoded ChatML stream itself.
    One decoder is live at a time; its NPU/XRT contexts are released before
    the caller builds the next placement.
    """
    with NPUDecoder(model_dir, npu_layers=npu_layers) as decoder:
        if npu_layers:
            decoder.warmup()
        prompt_ids = decoder.prompt_ids(PROMPT, count)
        prompt_text = decoder.tokenizer.decode(prompt_ids)
        if not prompt_text.startswith("<|im_start|>"):
            raise AssertionError(
                f"retained prompt is not well-formed ChatML: {prompt_text!r}"
            )
        if not prompt_text.endswith("<|im_start|>assistant\n"):
            raise AssertionError(
                f"retained prompt lost its generation prompt: {prompt_text!r}"
            )
        next_token = None
        timings = []
        position = 0
        for token_id in prompt_ids:
            next_token, elapsed = decoder.decode_token(token_id, position)
            timings.append(elapsed)
            position += 1
        generated = []
        for index in range(count):
            generated.append(next_token)
            if index + 1 < count:
                next_token, elapsed = decoder.decode_token(next_token, position)
                timings.append(elapsed)
                position += 1
        return generated, timings, decoder.tokenizer.eos_id, prompt_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens != len(EXPECTED_CPU_BF16_TOKENS):
        parser.error(
            f"--tokens must be {len(EXPECTED_CPU_BF16_TOKENS)} "
            "for the pinned reference"
        )

    cpu_tokens, cpu_times, eos_id, cpu_prompt_ids = sequence(
        args.model_dir, 0, args.tokens
    )
    gc.collect()
    npu_tokens, npu_times, npu_eos_id, npu_prompt_ids = sequence(
        args.model_dir,
        24,
        args.tokens,
    )
    if npu_eos_id != eos_id:
        raise AssertionError("CPU and NPU tokenizers disagree on EOS")
    if cpu_prompt_ids != npu_prompt_ids:
        raise AssertionError(
            "CPU and NPU placements were given different prompt tokens"
        )
    cpu_reference_differences = [
        index
        for index, pair in enumerate(
            zip(EXPECTED_CPU_BF16_TOKENS, cpu_tokens)
        )
        if pair[0] != pair[1]
    ]
    npu_reference_differences = [
        index
        for index, pair in enumerate(
            zip(EXPECTED_CPU_BF16_TOKENS, npu_tokens)
        )
        if pair[0] != pair[1]
    ]
    cpu_npu_differences = [
        index
        for index, pair in enumerate(zip(cpu_tokens, npu_tokens))
        if pair[0] != pair[1]
    ]
    cpu_eos_positions = [
        index for index, token_id in enumerate(cpu_tokens) if token_id == eos_id
    ]
    npu_eos_positions = [
        index for index, token_id in enumerate(npu_tokens) if token_id == eos_id
    ]
    report = {
        "schema_version": 1,
        "prompt": PROMPT,
        "token_count": args.tokens,
        "expected_cpu_bf16_token_ids": EXPECTED_CPU_BF16_TOKENS,
        "prompt_token_ids": cpu_prompt_ids,
        "actual_cpu_bf16_token_ids": cpu_tokens,
        "actual_npu_token_ids": npu_tokens,
        "agreement": args.tokens - len(npu_reference_differences),
        "cpu_reference_difference_positions": cpu_reference_differences,
        "npu_reference_difference_positions": npu_reference_differences,
        "cpu_npu_difference_positions": cpu_npu_differences,
        "eos_token_id": eos_id,
        "cpu_eos_positions": cpu_eos_positions,
        "npu_eos_positions": npu_eos_positions,
        "cpu_median_seconds": statistics.median(cpu_times),
        "npu_median_seconds": statistics.median(npu_times),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if cpu_eos_positions and cpu_eos_positions[0] < args.tokens - 1:
        raise AssertionError(
            "CPU reference ended before the required token sequence length"
        )
    if cpu_reference_differences:
        raise AssertionError(
            "Qwen CPU BF16 reference changed at positions "
            f"{cpu_reference_differences}"
        )
    if npu_reference_differences:
        raise AssertionError(
            "Qwen NPU reference mismatch at positions "
            f"{npu_reference_differences}"
        )


if __name__ == "__main__":
    main()
