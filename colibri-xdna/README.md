# Colibri direct C/XDNA integration

This integration adds an opt-in XRT backend to the pure-C
[`JustVugg/colibri`](https://github.com/JustVugg/colibri) engine. It is not an
HTTP proxy: Colibri's C process calls the backend directly through its resident
tensor ABI.

The supported source revision is pinned in `release-pins.json`. Build it with:

```bash
./colibri-xdna/scripts/build.sh --build-root work/colibri-build --test-npu
```

Run the resulting engine using Colibri's existing GPU-tier controls, which the
upstream ABI currently names `CUDA` even for alternative implementations:

```bash
export COLI_XDNA_XCLBIN="$PWD/ollama-xdna/backend/artifacts/experts-8x2048x2048/experts.xclbin"
export COLI_XDNA_INSTS="$PWD/ollama-xdna/backend/artifacts/experts-8x2048x2048/insts.bin"
export COLI_XDNA_MEMORY_MB=512
export COLI_CUDA=1
export CUDA_EXPERT_GB=0.5
COLI_MODEL=/path/to/colibri-model work/colibri-build/colibri/c/colibri chat
```

## Implemented scope

- direct C-linkage XRT backend; no Python inference or network bridge;
- lazy dense decode matmul for one row;
- routed expert gate/up/SiLU/down execution for up to eight selected experts;
- Colibri f32, per-row int8, per-row int4, and grouped-int4 weight ingestion;
- one-time conversion to row-wise W8 storage and BF16 activations on AIE;
- bounded device-memory accounting via `COLI_XDNA_MEMORY_MB`;
- explicit CPU fallback for prefill batches, attention, unsupported formats,
  shapes, and resident-pipeline operations.

This is an accuracy/performance trade-off because grouped int4 weights are
requantized to row-wise W8. The standalone NPU test checks the complete
Colibri ABI → XRT → AIE path. Full GLM-5.2 validation still requires the
roughly 400 GB Colibri model and is therefore not claimed by this repository.
