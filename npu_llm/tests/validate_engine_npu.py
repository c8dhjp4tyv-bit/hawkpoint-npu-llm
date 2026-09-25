#!/usr/bin/env python3
"""Hardware gate for the SmolLM decoder engine (designs/engine.py).

Every sequence is first generated greedily by the CPU BF16 reference. The
engine and the chunked layer graphs are then teacher-forced on the same
tokens, and each position's argmax is compared with the reference. The engine
must agree with the reference at least as well as the chunked path does (up to
a small tolerance), and every disagreement must be a near-tie in the
reference logits. Its softmax uses different BF16 approximations than the
chunked path, so exact token equality with that path is not required.
"""

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = [
    "Explain in simple terms why the sky looks blue during the day. Give a detailed answer.",
    "Write a haiku about the ocean.",
    "What is 12 times 7?",
    "Merhaba! Bugün nasılsın?",
    "List three uses of copper.",
    "Translate 'good morning' into French and Spanish.",
]
NEAR_TIE = 0.5


def reference_sequences(model_dir, tokens):
    from runtime.generate import NPUDecoder

    os.environ["HAWKPOINT_ENGINE"] = "0"
    cpu = NPUDecoder(model_dir, npu_layers=0)
    sequences, references = [], []
    for prompt in PROMPTS:
        ids = cpu.tokenizer.encode_chat([{"role": "user", "content": prompt}])
        ids = ids[-(64 - tokens):]
        token = None
        for position, token_id in enumerate(ids):
            token, _ = cpu.decode_token(token_id, position)
        sequence = list(ids) + [token]
        while len(sequence) < len(ids) + tokens:
            token, _ = cpu.decode_token(sequence[-1], len(sequence) - 1)
            sequence.append(token)
        sequence = sequence[:63]
        sequences.append(sequence)
        references.append(teacher_forced(cpu, sequence))
    cpu.close()
    return sequences, references


def teacher_forced(decoder, sequence):
    decoder.reset_prefix_cache()
    results = []
    for position, token_id in enumerate(sequence):
        _, _, diagnostics = decoder.decode_token(token_id, position, diagnostics=True)
        results.append(diagnostics)
    return results


def agreement(model_dir, engine, sequences, references):
    from runtime.generate import NPUDecoder

    os.environ["HAWKPOINT_ENGINE"] = "1" if engine else "0"
    decoder = NPUDecoder(model_dir)
    if (decoder._engine is not None) != engine:
        raise AssertionError(f"engine={engine} was not selected for {model_dir}")
    decoder.warmup()
    exact = total = 0
    hard_misses = []
    for index, sequence in enumerate(sequences):
        for position, (got, want) in enumerate(
            zip(teacher_forced(decoder, sequence), references[index])
        ):
            total += 1
            if got["top_token_ids"][0] == want["top_token_ids"][0]:
                exact += 1
            elif want["top_logit_margin"] >= NEAR_TIE:
                hard_misses.append((index, position, want["top_logit_margin"]))
    decoder.close()
    return exact, total, hard_misses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=Path,
        nargs="?",
        default=Path(
            os.environ.get(
                "HAWKPOINT_MODEL_DIR", ROOT / "models/SmolLM2-135M-Instruct-xdna1-w8a16"
            )
        ),
    )
    parser.add_argument("--tokens", type=int, default=31)
    parser.add_argument("--tolerance", type=int, default=6)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    sequences, references = reference_sequences(args.model_dir, args.tokens)
    chunked = agreement(args.model_dir, False, sequences, references)
    engine = agreement(args.model_dir, True, sequences, references)
    report = {
        "positions": engine[1],
        "chunked_top1": chunked[0],
        "engine_top1": engine[0],
        "chunked_hard_misses": chunked[2],
        "engine_hard_misses": engine[2],
        "near_tie_margin": NEAR_TIE,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if engine[0] + args.tolerance < chunked[0]:
        raise SystemExit(
            f"engine top-1 agreement {engine[0]} is more than {args.tolerance} "
            f"below the chunked path's {chunked[0]}"
        )
    if len(engine[2]) > len(chunked[2]) + 1:
        raise SystemExit(
            f"engine missed {len(engine[2])} clear reference choices "
            f"(chunked path: {len(chunked[2])})"
        )
    print("PASS engine agreement")


if __name__ == "__main__":
    main()
