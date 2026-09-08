# Roadmap

This document tracks planned improvements and known gaps. Items are ordered by
expected impact, not a committed timeline. This is an experimental research
project — entries represent investigation directions.

## High-Impact Optimizations

### Persistent packed-weight kernel for Ollama backend
The default W8 path now packs each GGML row once and keeps dense tiles and
selected MoE experts in a bounded, persistent XRT/host cache. Repeated decode
steps therefore do not call `to_float` or upload unchanged expert slots.

The repository also includes native Q4_K and Q6_K AIE2 kernels and an IRON
compiler (`ollama-xdna/backend/compile_quantized.py`). When their matching
xclbin is built and selected with `GGML_XDNA_NATIVE_QUANT_*`, GGML blocks stay
quantized in the persistent XRT BO and dequantization is fused with GEMV on the
AIE array. The native artifacts remain opt-in until the exact hardware/toolchain
stack has passed the physical release gates.

**Target**: remove host-side dequantize/requantize and per-token weight
movement; measure the eventual 3–8× decode estimate on hardware rather than
assuming it.

### Multi-context NPU pipelining
Overlap host-side tokenization and LM head with NPU compute by submitting the
next layer's work while the current one finishes. Requires vertex-driven dispatch
and reworked IRON runtimes.

## Completed

### Sampling parameter support
Temperature, top-k, top-p, repetition/presence/frequency penalties, and a
reproducible `seed` are supported by the runtime, the terminal chat, and the
OpenAI-compatible API. Greedy decoding remains the default so the exact-token
release gates are unaffected.

Still missing from the completion API: `n > 1`, beam search, `logprobs`, and
stop sequences.

## XDNA2 / NPU4 Compatibility

### Current state
XDNA2 (Strix Point, Strix Halo) uses AIE4 tiles with 8 columns (vs XDNA1's 4),
new shared L2 buffer, and next-gen DMA. The current IRON graphs hardcode a
4-column array.

### Required for proof-of-concept
1. **Re-parameterize IRON layouts** — column count and ObjectFifo depths as
   configurable in `qwen_decoder.py`
2. **AIE2→AIE4 kernel rewrite** — wider vectors (512-bit), FP8 tensor blocks
3. **Auto-detect device column count** at runtime instead of fixed `n_cols=4`
4. Validate token-acceptance parity with existing CPU BF16 reference sequences

**Note**: XDNA2 porting requires physical hardware. No timeline commitment.

## Documentation & Observability

- [ ] IRON graph visualizer (DOT/SVG of tile dataflows)
- [ ] Offline benchmark harness (script repo for crossplacement checks)
- [ ] Dockerfile with pre-built MLIR-AIE (lowers entry barrier)

## Model Extensions

- SmolLM2-360M support (requires new hardware graph for 576→960 transition)
- FP8 experimental path for SmolLM (AIE2), worth while kernel
- Qwen3-Coder:30b full NPU merge (not just expert portions)

## Known Out-of-Scope

- Training / fine-tuning on NPU
- CUDA / ROCm / oneAPI NPU delegates
- Windows or macOS support
- LLM evaluation bench (MMLU, HumanEval) — 64-token context insufficient
