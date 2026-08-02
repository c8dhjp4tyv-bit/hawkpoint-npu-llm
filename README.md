# Hawk Point NPU LLM

Run compatible SmolLM 135M and Qwen2.5 0.5B transformer checkpoints on the
first-generation AMD XDNA NPU in Phoenix and Hawk Point Ryzen AI processors.
The implementation uses MLIR-AIE/IRON and targets the `npu1` AIE2 array
directly.

The GPU is not used. The CPU handles tokenization, orchestration, argmax,
printing, reference validation, and Qwen's final LM head. Decoder projections,
RMSNorm, RoPE, attention, KV caches, residuals, and SwiGLU execute as AIE2
kernels.

> Experimental research project. It is not affiliated with or supported by
> AMD, Xilinx, Hugging Face, or the Open WebUI project.

The versioned support boundary is in [SUPPORT.md](SUPPORT.md), release changes
are in [CHANGELOG.md](CHANGELOG.md), and responsible disclosure is described
in [SECURITY.md](SECURITY.md). An alpha tag does not mean production-ready.

## Choose a runtime

This repository now contains two independent XDNA1 paths:

| Runtime | Best use |
|---|---|
| Native MLIR-AIE runtime below | Small supported checkpoints with a custom OpenAI-compatible server |
| [Ollama XDNA1 patch](ollama-xdna/README.md) | Existing Ollama Qwen models shared across CPU, GPU, and NPU |

For the Ollama path, install distro-specific build dependencies first:

```bash
./ollama-xdna/scripts/install-deps.sh
./ollama-xdna/scripts/verify-system.sh
```

Then patch a clean upstream Ollama v0.32.5 checkout, build all matching native
components, run the XDNA hardware test, and install:

```bash
./ollama-xdna/scripts/build-and-install.sh --backend cuda_v13
```

Use `--backend cpu`, `cuda_v12`, `cuda_v13`, `rocm_v7_2`, or `vulkan` to match
the machine. The complete driver/XRT prerequisites, distro commands, safe
dry-run, model test, update, API/Open WebUI, and rollback instructions are in
the [Ollama XDNA1 guide](ollama-xdna/README.md).

## Demonstrated hardware result

Validated on a Hawk Point XDNA1 NPU (`RyzenAI-npu1`, AIE2, 4 columns):

| Measurement | Result |
|---|---:|
| CPU/BF16 reference first-token argmax | 198 |
| NPU first-token argmax | 198 |
| Cold compile/load | 7.03 s |
| Warm `decode_token` smoke test | 10.72 token/s |
| 32-token streaming chat | 3.65 token/s |
| End-to-end TTFT for the 32-token acceptance prompt | 8.80 s |
| Peak host RAM during acceptance | 833.6 MiB |
| Fixed hardware context | 64 tokens |

The acceptance prompt generated:

```text
The sky looks blue during the day because the Earth's atmosphere scatters
the sunlight in all directions, including blue light. When sunlight enters
the Earth's atmosphere,
```

The fused Qwen2.5 0.5B path is checked against both the NumPy BF16 runtime and
the upstream BF16 checkpoint. Release candidates require an exact checked-in
32-token generated sequence. A separate 32-position prefill benchmark records
the top-five logits, BF16 near-ties, and CPU/NPU latency without treating a
performance regression as a correctness pass or claiming an unmeasured
speedup.

The API prewarms the default model, moving the roughly 4.2-second compile/load
cost to server startup. A real four-token `Hello` response produced
`Hello! How can` at 1.47 decode tokens/s after prompt ingestion, using about
2.4 GiB peak host RAM. Results are from the same Hawk Point system and vary
with memory pressure and CPU BLAS configuration.

## What is included

- Interactive, multi-turn terminal chat with `/reset`, `/stats`, and `/exit`
- Four selectable checkpoints across the SmolLM and Qwen families
- Configurable NPU/CPU layer offload for hybrid execution
- OpenAI-compatible `GET /v1/models`
- OpenAI-compatible `POST /v1/chat/completions`
- Streaming chat completions over server-sent events
- One-command API or API + Open WebUI launcher
- Weight converter and hardware component/acceptance tests
- AIE2 C++ kernels and IRON graph definitions

Conversation history is retained by the client and automatically trimmed to
the newest tokens that fit the current 64-token hardware context.

## Supported models

| API model ID | Hugging Face checkpoint | Intended use |
|---|---|---|
| `smollm2-135m-xdna1` | `HuggingFaceTB/SmolLM2-135M-Instruct` | Default assistant |
| `smollm-135m-xdna1` | `HuggingFaceTB/SmolLM-135M-Instruct` | Previous-generation assistant |
| `smollm2-135m-sft-xdna1` | `HuggingFaceTB/smollm2-135M-SFT-Only` | SFT comparison/research |
| `qwen2.5-0.5b-xdna1` | `Qwen/Qwen2.5-0.5B-Instruct` | Larger experimental assistant |

The converter rejects checkpoints whose hidden size, intermediate size, layer
count, attention layout, or vocabulary do not match one of the implemented
hardware graphs. A model name alone is not enough: unsupported variants such
as SmolLM2-360M, SmolLM2-1.7B, or larger Qwen checkpoints are not silently
accepted.

## Requirements

- A Phoenix or Hawk Point Ryzen AI system exposing `RyzenAI-npu1`
- Linux with a working `amdxdna`/XRT stack and matching NPU firmware
- A working MLIR-AIE/IRON environment with Peano
- Python 3.12 packages hash-locked in `requirements.lock`
- Docker with Compose support, only for the Open WebUI option

This project was validated with NPU firmware `1.5.5.391`. Driver, firmware,
XRT, Peano, and MLIR-AIE versions must be mutually compatible.

## Set up

First prepare MLIR-AIE using its upstream setup instructions. In a shell where
that checkout lives:

```bash
cd /path/to/mlir-aie
# Version used for the published hardware validation:
git checkout 57d7494e99c
source ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh >/dev/null
source utils/env_setup.sh "$PWD" "$PWD/peano"
```

Clone this repository and install the model-conversion dependencies into the
same environment:

```bash
git clone https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm.git
cd hawkpoint-npu-llm
python -m pip install --require-hashes -r requirements.lock
python scripts/prepare_model.py
```

`prepare_model.py` downloads the upstream Hugging Face model and creates the
XDNA1 runtime representation under `npu_llm/models/`. Model weights are not
stored in this Git repository. Checkpoints use immutable revisions recorded in
`npu_llm/model_catalog.py`. Conversion occurs in a sibling staging directory;
only a complete, checksum-verified package replaces the previous model.

Prepare every supported model:

```bash
python scripts/prepare_model.py --all
```

Or choose one or more explicitly:

```bash
python scripts/prepare_model.py \
  --model smollm2-135m-xdna1 \
  --model qwen2.5-0.5b-xdna1
```

Qwen2.5 0.5B creates roughly 1.7 GiB of converted runtime files. Its decoder
uses two-layer persistent BF16 XDNA1 programs with explicit round-to-nearest-
even conversion for numerically sensitive residual paths.

Use `--models-dir /path/to/storage` to keep the source and converted weights on
another disk. Pass that same directory to the launcher with `--models-dir`.

Verify the NPU:

```bash
xrt-smi examine
```

## Run

Choose interactively between the two server modes:

```bash
python launcher.py
```

Or select one directly:

```bash
# API only: http://localhost:8000/v1
python launcher.py api

# API + Open WebUI: http://localhost:3000
python launcher.py openwebui

# Models stored on another disk
python launcher.py openwebui --models-dir /path/to/storage

# Ollama-style hybrid offload: 60% of decoder layers on the NPU
python launcher.py openwebui --npu-percent 60

# Or select an exact number of leading NPU layers
python launcher.py openwebui --npu-layers 4
```

The pinned Open WebUI container uses Linux host networking so it can reach the
API while both processes stay bound to `127.0.0.1`. Its model picker displays
every installed checkpoint returned by `/v1/models`. Its data is kept in a
Docker volume, and the launcher gives it the same randomly generated API key
as the native server.

Run the terminal chatbot:

```bash
python npu_llm/chat.py
```

Example one-shot invocation:

```bash
python npu_llm/chat.py \
  --prompt "Explain in simple terms why the sky looks blue."
```

## OpenAI-compatible API

The launcher generates a random bearer token and prints it in API-only mode.
Set a stable token when another local client must reconnect:

```bash
export HAWKPOINT_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python launcher.py api
```

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $HAWKPOINT_API_KEY"
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HAWKPOINT_API_KEY" \
  -d '{
    "model": "smollm2-135m-xdna1",
    "messages": [
      {"role": "user", "content": "Why is the sky blue?"}
    ],
    "max_tokens": 16,
    "stream": false
  }'
```

For streaming output, set `"stream": true`.

Select another installed model by changing the request's `model` field:

```json
{
  "model": "smollm-135m-xdna1",
  "messages": [{"role": "user", "content": "Say hello."}]
}
```

An unknown or unprepared model returns `404 model_not_found`; it is never
silently routed to a different checkpoint. Only one checkpoint is retained by
the server at a time to limit host RAM usage. The default model is loaded and
prewarmed during server startup. A model selected later is prewarmed while it
is switched in. Pass `--no-prewarm` directly to `api_server.py` only when
deferred compilation is preferred.

## Hybrid NPU/CPU offload

Hybrid execution assigns a contiguous prefix of decoder layers to XDNA1 and
the remaining layers plus the LM head to NumPy on the CPU. It is analogous to
Ollama's GPU layer offload; percentages describe decoder-layer placement, not
an exact utilization or RAM split.

For Qwen2.5 0.5B, `--npu-percent 60` maps 14 of 24 layers to the NPU and 10 to
the CPU. The optimized default is now all 24 decoder layers on the NPU:

```bash
python launcher.py openwebui \
  --models-dir /path/to/models
```

Use `--npu-layers 0` for the CPU reference path and omit both options for the
optimized all-NPU decoder path. CPU-offloaded layers retain their own CPU KV
caches; NPU layers retain NPU-resident KV caches. Only the boundary hidden
state moves between devices once per generated token.

The server serializes requests because one physical NPU execution context is
shared. One active request and two queued requests are admitted by default;
additional work receives `429` instead of accumulating waiting inference
threads. Defaults also include a 1 MiB request limit, 120-second socket
timeout, 30 requests/minute/client rate limit, bearer authentication, safe
internal errors, and an explicit browser-origin allowlist.

The HTTP process never owns the XRT context. Inference runs in a persistent
spawned worker process. If a request deadline expires, the worker is
terminated, its NPU context is discarded, and the next request creates a fresh
worker. A worker-side inference error also terminates that worker so a possibly
corrupted XRT context is never reused. `GET /health` reports HTTP-process
liveness; `GET /ready` returns `200` only while a warmed inference worker is
available, and `503` after an error or timeout until recovery succeeds.

Both launcher modes bind the API to `127.0.0.1`; Open WebUI also binds only to
localhost. To run the server directly with TLS, pass `--tls-cert CERT.pem
--tls-key KEY.pem`. For non-local deployment, use a trusted TLS reverse proxy
and keep the backend private.

## Convert a model manually

```bash
python npu_llm/tools/convert_smollm2.py \
  npu_llm/models/SmolLM2-135M-Instruct \
  npu_llm/models/SmolLM2-135M-Instruct-xdna1-w8a16
```

A converted SmolLM directory is approximately 442 MiB; Qwen2.5 0.5B is
approximately 1.7 GiB. The format retains BF16 decoder weights for numerically
stable generation and per-output-channel INT8 weights for W8/BF16 kernels.
The runtime verifies package sizes and SHA-256 hashes before memory mapping
weights; packages created by an older converter must be reconverted.

## Tests

Tests are directly executable and do not require pytest:

```bash
python npu_llm/tests/test_converter.py
python npu_llm/tests/test_model_runtime.py
python tests/test_api_server.py
```

Run the complete hardware acceptance test:

```bash
python npu_llm/tests/validate_chat_npu.py
```

The tag-triggered release pipeline performs fresh pinned downloads and
conversion, Qwen token agreement, a bounded model-switch stress test of at
least 100 switches, a 1,000-completion endurance soak of 250 consecutive
requests per model, and an Ollama install/inference/rollback test with
`--jobs 8` before its publish job can start. See [SUPPORT.md](SUPPORT.md) for the gate and
[BENCHMARKS.md](BENCHMARKS.md) for the controlled comparison protocol.
`release-pins.json` is the machine-readable authority for the Ollama source
commit, Ollama model manifest, and accepted hardware/software stack.

Component examples:

```bash
python npu_llm/tests/test_elementwise_npu.py
python npu_llm/tests/test_qwen_components_npu.py
python npu_llm/tests/benchmark_qwen_fused.py /path/to/converted-qwen
python npu_llm/designs/rmsnorm.py --dev npu --size 576 -w 2 -i 5
python npu_llm/designs/rope.py --dev npu --heads 12 --position 7 -w 2 -i 5
```

## Architecture

Each generated token follows this path:

1. Fetch the token embedding into an XRT buffer.
2. Run the decoder layers using the fused SmolLM graph or persistent two-layer
   Qwen BF16 programs.
3. Keep the 64-token K/V cache in NPU-addressable XRT buffers.
4. Apply final RMSNorm and the model-specific LM head. Qwen uses a cached CPU
   LM-head matrix while its 24 decoder layers remain on XDNA1.
5. Select the next token and decode it on the host.

The decoder xclbin is compiled once and reused for every two-layer Qwen chunk
and token position. Layer weights and K/V caches remain in XRT buffer objects.

## Limitations

### Hardware

- **Fixed 64-token context window**: The KV cache is allocated as NPU/XRT
  buffer objects sized for 64 positions (`2 × 4096 BF16 per KV head`).
  This limit is currently hard-coded in the IRON graph and kernel definitions;
  it is not imposed by the physical tile memory itself. Extending it would
  require rebalancing the weight stream and cache buffer layout across the
  AIE tiles and recompiling the xclbin. Prompts longer than 64 tokens are
  automatically trimmed by the host.
- **Padlocked to XDNA1 (`npu1`, AIE2, 4 columns).** XDNA2/NPU4 silicon
  (Strix Point, Strix Halo — 8 columns, larger local memory, shared
  caches) is not targeted. The IRON graph layouts, tile counts, and ObjectFifo
  depths assume a 4-column array. Porting requires at minimum a re-parameterized
 IRON design and a full AIE2→AIE4 kernel rewrite.
- **No native BF16/FP16 tensor cores.** All AIE2 operations use a software
  BF16 multiply-accumulate path via `aie::mul` + `aie::accum`. There is no
  equivalent to NVIDIA Tensor Cores or Apple AMX blocks on this NPU
generation.
- **Two decoder layers per persistent invocation** — verified firmware
  limit. Qwen2.5 0.5B stacks 12 chunks of 2 layers each.

### Model support

- Only the exact four checkpoint architectures listed above are supported.
  The converter rejects any model whose hidden size, intermediate size,
  layer count, attention layout, or vocabulary differ from a known template.
  SmolLM2-360M, SmolLM2-1.7B, larger Qwen variants, and non-Llama architectures
  fail at conversion time, not silently at runtime.
- **No sampling parameters**. Generation is greedy. Temperature, top-k, top-p,
  and repetition penalty are accepted by neither the runtime nor the
  OpenAI-compatible API endpoint.

### Performance

- **Qwen2.5 0.5B NPU path is not faster than an eight-thread CPU baseline**
  on measured hardware. The first-generation XDNA AIE2 array has ~2.34 TFLOPS
  of BF16 peak throughput compared to a Zen 4 CPU core cluster at comparable
  throughput with much lower launch overhead. See [Performance
  Analysis](docs/PERFORMANCE-ANALYSIS.md) for a detailed breakdown.
- The **Ollama XDNA backend** converts GGML rows to W8 for each decoded token
  and streams padded weights through host→NPU DMA. This overhead makes the
  NPU path slower than the optimized CPU/GPU path for all measured models.
  A persistent packed-weight kernel (Q4_K/Q6_K→BF16 fused) would eliminate this
  bottleneck — this is the highest-value optimization and is tracked as
  a future work item in [ROADMAP.md](ROADMAP.md).
- **Warm decode is line-rate only for a single token stream**
  with no batching. The NPU has one execution context shared across all
  requests; the HTTP server serializes inference and returns `429 Too Many
  Requests` beyond a configurable queue depth.

### Software

- Requires a specific **XRT + firmware + kernel + MLIR-AIE version
  combination**. `release-pins.json` records the validated stack. Component
  version drift causes silent failures (DRM_IOCTL_AMDXDNA_CREATE_HWCTX
  failures, `aie2_alloc_resource` exhaustion).
- **No CUDA, ROCm, oneAPI, or Vulcan NPU delegates**: the AIE2 kernel follows
  the old MTBL path. IREE, TPU-MLIR, open-Silicon, and XDNA-API-based builds
  are not currently in scope.
- **Single-event completion model** — the inference worker uses
  `wait=True` on AIE command completion, not an interrupt-driven dispatch
  or multi-worker pipelining. Between-token gaps include an estimated XRT
  command-submission cost (~120 µs in the analytical model); this phase is not
  currently instrumented separately.

See [ROADMAP.md](ROADMAP.md) for planned improvements and [BENCHMARKS.md](BENCHMARKS.md) for the
controlled comparison protocol.

## License

Apache License 2.0 with LLVM exception. See `LICENSE`.

Model files are downloaded separately and remain subject to their respective
upstream licenses and terms.
