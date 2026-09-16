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
in [SECURITY.md](SECURITY.md). Deployment, monitoring, shutdown, recovery, and
release procedures are in the [operations guide](docs/OPERATIONS.md). An alpha
tag does not mean production-ready.

## Choose a runtime

This repository now contains three independent XDNA1 paths:

| Runtime | Best use |
|---|---|
| Native MLIR-AIE runtime below | Small supported checkpoints with a custom OpenAI-compatible server |
| [Ollama XDNA1 patch](ollama-xdna/README.md) | Existing Ollama Qwen models shared across CPU, GPU, and NPU |
| [Colibri direct C/XDNA backend](colibri-xdna/README.md) | Colibri routed-expert and decode matmul calls made directly from its C engine |

For the Ollama path, install distro-specific build dependencies first:

```bash
./ollama-xdna/scripts/install-deps.sh
./ollama-xdna/scripts/verify-system.sh
```

Then patch a clean upstream Ollama v0.33.3 checkout, build all matching native
components, run the XDNA hardware test, and install:

```bash
./ollama-xdna/scripts/build-and-install.sh --backend cuda_v13
```

Use `--backend cpu`, `cuda_v12`, `cuda_v13`, `rocm_v7_2`, or `vulkan` to match
the machine. The complete driver/XRT prerequisites, distro commands, safe
dry-run, model test, update, API/Open WebUI, and rollback instructions are in
the [Ollama XDNA1 guide](ollama-xdna/README.md).

For Colibri, the repository provides a pinned source patch and a native C++/XRT
implementation of Colibri's C backend ABI. Build it and exercise the NPU path
without a proxy using:

```bash
./colibri-xdna/scripts/build.sh --build-root work/colibri-build --test-npu
```

Supported operations, environment variables, CPU fallback behavior, and the
current full-model validation boundary are documented in the
[Colibri direct C/XDNA guide](colibri-xdna/README.md).

## Demonstrated hardware result

Validated on a Hawk Point XDNA1 NPU (`RyzenAI-npu1`, AIE2, 4 columns):

| Measurement | Result |
|---|---:|
| CPU/BF16 reference first-token argmax | 198 |
| NPU first-token argmax | 198 |
| Cold compile/load | 7.82 s |
| Warm `decode_token` smoke test | 13.25 token/s (0.0755 s/token) |
| 32-token streaming chat | 15.93 token/s |
| Native 16-token W8 decode, latest 3-run median | 14.42 token/s |
| Corrected-loop session medians / best individual run | 14.42–19.99 / 20.20 token/s |
| Native W8 TTFT, 35-token prompt | 1.47–1.62 s session medians |
| End-to-end TTFT for the 32-token acceptance prompt | 2.56 s |
| Peak host RAM during acceptance | 349.5 MiB |
| Fixed hardware context | 64 tokens |

The acceptance prompt generated:

```text
The sky appears blue during the day because our eyes are sensitive to the
color of the sky. When sunlight enters the Earth's atmosphere, it encounters
tiny particles called
```

The fused Qwen2.5 0.5B path is checked against both the NumPy BF16 runtime and
the upstream BF16 checkpoint. Release candidates require an exact checked-in
32-token generated sequence. A separate 32-position prefill benchmark records
the top-five logits, BF16 near-ties, and CPU/NPU latency without treating a
performance regression as a correctness pass or claiming an unmeasured
speedup.

The API prewarms the default model, moving the compile/load cost to server
startup. Results are from the same Hawk Point system and vary with temperature,
memory pressure, concurrent NPU activity, and CPU BLAS configuration. The
current loop skips the discarded LM-head work at non-final prefill positions
and does not execute an unused successor step after reaching `max_tokens`.
Consequently, the corrected measurements use 15 timed decode intervals for 16
emitted tokens after TTFT. Repeated three- and five-run sessions produced
14.42–19.99 token/s medians (20.20 token/s best individual run), demonstrating
the device's temperature/load sensitivity. The older 18.42 token/s result timed an
additional discarded successor step and is historical decoder-step evidence,
not a directly comparable current end-to-end rate.

The current controlled benchmark remains below the requested 50–75 token/s
range: the full 30-layer SmolLM decoder is the dominant measured phase on this
XDNA1 device. The best measured configuration is opt-in W8 projection plus
CPU final RMSNorm/LM-head (`HAWKPOINT_DECODER_W8=1
HAWKPOINT_CPU_FINAL_NORM=1`); it reports the miss rather than extrapolating an
unmeasured kernel rate.

## What is included

- Interactive, multi-turn terminal chat with `/reset`, `/stats`, and `/exit`
- Four selectable checkpoints across the SmolLM and Qwen families
- Configurable NPU/CPU layer offload for hybrid execution
- OpenAI-compatible `GET /v1/models`
- OpenAI-compatible `POST /v1/chat/completions`
- Streaming chat completions over server-sent events
- Temperature, top-k, top-p, penalty, and seed sampling (greedy by default)
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

For streaming output, set `"stream": true`. Add
`"stream_options": {"include_usage": true}` to receive a final chunk with
`choices: []` and token counts in `usage`, immediately before `[DONE]`.
Earlier chunks carry `usage: null` when this option is enabled. Without it,
the existing streaming format is preserved.

An inference failure after streaming headers produces a sanitized SSE `error`
event (`server_error` or `timeout_error`) and closes the stream without a
success `[DONE]` marker. Clients should treat a disconnected stream without
`[DONE]` as incomplete; final usage is not guaranteed for interrupted requests.

`max_tokens` or its modern alias `max_completion_tokens` must be a positive
JSON integer (then capped to the hardware context); supplying both is rejected.
`n` must be integer `1`, and `stream` must be a JSON boolean.
`stream_options` is only accepted with streaming enabled. Invalid types return
`400` before inference admission. Header and body reads are also bounded by
`--request-timeout`; a stalled body receives `408` when the connection remains
writable. Inference timeouts return `504` for non-streaming requests.

Responses include a unique `X-Request-ID`, `Cache-Control: no-store`, and
`X-Content-Type-Options: nosniff`. Streaming responses additionally disable
common reverse-proxy buffering with `X-Accel-Buffering: no`. Routes continue to
work when clients append query parameters.

Clients can retrieve one installed model with `GET /v1/models/{model_id}`.
Rate-limit and full-queue `429` responses include `Retry-After`. SIGINT and
SIGTERM trigger graceful HTTP shutdown and deterministic NPU worker cleanup,
which is particularly important under systemd and container supervisors.
During shutdown, newly arriving completions receive `503` while already
admitted inference drains for up to `--graceful-shutdown-timeout` seconds
(130 by default, slightly longer than the request deadline).
`GET /ready` switches to `503` immediately and reports `draining` plus the
active request count, allowing a reverse proxy or service manager to stop
routing new work before process exit. Access logs include the same request ID
returned to the client for correlation.

### Sampling

Generation is greedy unless a request opts in. `temperature: 0` (the default)
selects the highest-scoring token, which keeps the reproducible acceptance
sequences intact.

| Field | Default | Accepted range |
|---|---|---|
| `temperature` | `0` | `0`–`2`, `0` means greedy |
| `top_p` | `1.0` | greater than `0` through `1.0` |
| `top_k` | `0` | `0` (disabled) or a positive integer |
| `repetition_penalty` | `1.0` | `0.1`–`2.0` |
| `presence_penalty` | `0` | `-2.0`–`2.0` |
| `frequency_penalty` | `0` | `-2.0`–`2.0` |
| `seed` | none | `0`–`2^63 - 1` |

Filters are applied in the order penalties, temperature, `top_k`, `top_p`, and
then a multinomial draw. Penalties are scored over the prompt tokens as well as
the generated ones. An out-of-range value is rejected with `400
invalid_request_error` and never reaches the NPU worker.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HAWKPOINT_API_KEY" \
  -d '{
    "model": "smollm2-135m-xdna1",
    "messages": [{"role": "user", "content": "Write one sentence about rain."}],
    "max_tokens": 24,
    "temperature": 0.8,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "seed": 1234
  }'
```

A request that supplies `seed` reproduces its output exactly. An unseeded
sampled request draws a seed and reports it, so a run can be replayed:
`x_hawkpoint_stats.sampling` carries the effective mode, seed, and every
resolved parameter.

The terminal chat exposes the same options:

```bash
python npu_llm/chat.py \
  --prompt "Write one sentence about rain." \
  --temperature 0.8 --top-p 0.95 --repetition-penalty 1.1 --seed 1234
```

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
python npu_llm/tests/test_model_integrity.py
python npu_llm/tests/test_model_runtime.py
python npu_llm/tests/test_sampling.py
python npu_llm/tests/test_decode_loop.py
python npu_llm/tests/test_cpu_backend.py
python tests/test_api_server.py
```

`test_sampling.py` and `test_decode_loop.py` need no NPU: token selection is a
host operation, so the sampler and the decode loop's use of it are covered with
canned logits.

Run the complete hardware acceptance test:

```bash
python npu_llm/tests/validate_chat_npu.py
```

The tag-triggered release pipeline performs fresh pinned downloads and
conversion, Qwen token agreement, a bounded model-switch stress test of at
least 100 switches, a quick soak of 100 completions (25 consecutive per
model), an Ollama install/inference/rollback test, and a direct Colibri C/XDNA
build plus NPU ABI test with `--jobs 8` before its publish job can start. See
[SUPPORT.md](SUPPORT.md) for the gate and
[BENCHMARKS.md](BENCHMARKS.md) for the controlled comparison protocol.
The Ollama matrix uses 25 measured requests per placement (100 total), so the
two quick loops total 200 measured requests. Warm-up, correctness checks,
100 model switches, downloads, and builds are additional work.
For optional long-duration testing, manually run **Gated release** with
`profile: endurance` (1,000 native plus 4 × 1,000 Ollama requests).
Manual runs never publish a release. Tags use the quick profile, whose
success does not establish long-duration endurance.
`release-pins.json` is the machine-readable authority for the Ollama source tag
and commit, the Colibri source commit, Ollama model manifest, and the
hardware/software stack used for release certification. It is not an exact
kernel requirement for every runtime;
the compatibility validator checks capabilities and preserves observed version
differences in its report.

Component examples:

```bash
python npu_llm/tests/test_elementwise_npu.py
python npu_llm/tests/test_qwen_components_npu.py
python npu_llm/tests/benchmark_qwen_fused.py /path/to/converted-qwen
HAWKPOINT_DECODER_W8=1 HAWKPOINT_CPU_FINAL_NORM=1 HAWKPOINT_PROFILE=1 \
  python scripts/benchmark_native.py /path/to/converted-smollm \
  --tokens 16 --runs 3 --report benchmark.json
HAWKPOINT_PROFILE=1 python scripts/benchmark_native.py \
  /path/to/converted-qwen --tokens 32 --runs 3 --report benchmark.json
python npu_llm/designs/rmsnorm.py --dev npu --size 576 -w 2 -i 5
python npu_llm/designs/rope.py --dev npu --heads 12 --position 7 -w 2 -i 5
```

## Architecture

Each generated token follows this path:

1. Fetch the token embedding row into an XRT buffer.
2. Run SmolLM decoder layers in persistent two-layer graph chunks (the Phoenix
   firmware reliably completes this chunk size); Qwen uses its own two-layer
   BF16 program.
3. Keep the 64-token K/V cache in NPU-addressable XRT buffers.
4. Apply final RMSNorm and the model-specific LM head. SmolLM keeps the
   49k-vocabulary projection on host BLAS by default because it is faster on
   Hawk Point; set `HAWKPOINT_NPU_LM_HEAD=1` to force the NPU path.
5. Select the next token and decode it on the host.

The decoder xclbin is compiled once and reused for every two-layer chunk and
token position. Layer weights and K/V caches remain in XRT buffer objects. The
largest gate/up projection is split across three compute workers, and the
attention kernel copies only the cache prefix needed for the current position.
The optional `HAWKPOINT_DECODER_W8=1` mode uses the bundled int8 projection
kernels. On the validated Phoenix stack it is the fastest measured path; the
default BF16 decoder remains available as the numerical baseline. For SmolLM,
`HAWKPOINT_CPU_FINAL_NORM=1` moves only the final 576-element RMSNorm to the
host and is valid together with the CPU LM-head path.
W8 is an accuracy/performance trade-off: the release acceptance gate remains
on the default BF16 path, while the W8 benchmark is reported separately.
Set `HAWKPOINT_PROFILE=1` (or pass `--profile` to `benchmark_native.py`) to
include embedding, RoPE, decoder, LM-head, and CPU phase timings in the final
statistics. The benchmark reports whether the measured median reaches the
50 token/s target; it never fabricates a result when no NPU is available.

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
- **Greedy by default, with opt-in sampling**. `temperature`, `top_p`,
  `top_k`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, and
  `seed` are supported. A request that sets none of them decodes greedily and
  reproduces the checked-in acceptance sequences exactly. `n > 1`, beam search,
  `logprobs`, and stop sequences are still unsupported.

### Performance

- **Qwen2.5 0.5B NPU path is not faster than an eight-thread CPU baseline**
  on measured hardware. The first-generation XDNA AIE2 array has ~2.34 TFLOPS
  of BF16 peak throughput compared to a Zen 4 CPU core cluster at comparable
  throughput with much lower launch overhead. See [Performance
  Analysis](docs/PERFORMANCE-ANALYSIS.md) for a detailed breakdown.
- The **Ollama XDNA backend** now packs GGML rows once and retains dense tiles
  and selected MoE experts in a bounded persistent cache. Unchanged decode
  steps no longer repeat CPU dequantize/requantize or weight DMA. Native
  Q4_K/Q6_K AIE2 kernels are included under `npu_llm/kernels/` and can be
  compiled with `ollama-xdna/backend/compile_quantized.py`; their xclbins are
  opt-in until the physical release gates validate the exact toolchain stack.
- **Warm decode is line-rate only for a single token stream**
  with no batching. The NPU has one execution context shared across all
  requests; the HTTP server serializes inference and returns `429 Too Many
  Requests` beyond a configurable queue depth.

### Software

- Requires a working **XRT + firmware + amdxdna + MLIR-AIE combination**. The
  exact stack in `release-pins.json` is the reproducible release certificate,
  not a runtime kernel pin. Run `python scripts/verify_hardware_versions.py`
  for capability-based compatibility validation; the release workflow adds
  `--strict-release` to enforce the certified version strings as well. A
  compatible kernel/driver must still pass device open, hardware-context,
  buffer sync, xclbin load, and minimal kernel-submission checks.
- **No CUDA, ROCm, oneAPI, or Vulcan NPU delegates**: the AIE2 kernel follows
  the old MTBL path. IREE, TPU-MLIR, open-Silicon, and XDNA-API-based builds
  are not currently in scope.
- **Single-event completion model** — the inference worker uses
  `wait=True` on AIE command completion, not an interrupt-driven dispatch
  or multi-worker pipelining. Between-token gaps include XRT command submission
  overhead, but the current benchmark tooling does not instrument that phase
  separately.
- The Ollama backend enforces one global persistent-weight cache budget across
  dense, expert, and native Q4/Q6 caches. Set
  `GGML_XDNA_DEVICE_MEMORY_MB` only when the platform exposes a known capacity;
  otherwise device memory is reported as unknown rather than guessed.

See [ROADMAP.md](ROADMAP.md) for planned improvements and [BENCHMARKS.md](BENCHMARKS.md) for the
controlled comparison protocol.

## License

Apache License 2.0 with LLVM exception. See `LICENSE`.

Model files are downloaded separately and remain subject to their respective
upstream licenses and terms.
