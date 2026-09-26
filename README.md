# Hawk Point NPU LLM

Run compatible SmolLM 135M and Qwen2.5 0.5B transformer checkpoints on the
first-generation AMD XDNA NPU in Phoenix and Hawk Point Ryzen AI processors.
The implementation uses MLIR-AIE/IRON and targets the `npu1` AIE2 array
directly.

The GPU is not used. The CPU handles tokenization, orchestration, argmax,
printing, reference validation, and SmolLM's final LM head (host BLAS is
faster for its 49k-row head). Qwen's LM head runs on the NPU, except under
hybrid placement, which moves it to the CPU together with the trailing layers.
Decoder projections, RMSNorm, RoPE, attention, KV caches, residuals, and SwiGLU
execute as AIE2 kernels.

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

| Measurement (SmolLM2 135M, decoder engine) | Result |
|---|---:|
| CPU/BF16 reference first-token argmax | 198 |
| NPU first-token argmax | 198 |
| Native 16-token decode, 5-run median | 74.61 token/s |
| Individual runs | 73.17–75.83 token/s |
| TTFT, 35-token prompt | 0.28–0.29 s |
| TTFT, repeated prompt with `--prefix-cache` | 0.014 s |
| NPU time per token (all 30 layers, one dispatch) | 7.6–8.9 ms |
| CPU LM head per token | about 5 ms |
| Warmup (engine compile cached) | 5.1 s |
| Peak host RAM during the benchmark | 354 MiB |
| Quick soak / model-switch stress | 100/100 completions, 100 switches, 0 NPU errors |
| Fixed hardware context | 64 tokens |

The chunked layer graphs (`HAWKPOINT_ENGINE=0`) measured 17.1 token/s with
1.71–1.81 s TTFT on the same machine and benchmark.

Qwen2.5 0.5B runs on its own engine (`designs/qwen_engine.py`), including the
LM head:

| Measurement (Qwen2.5 0.5B, same benchmark) | Engine | Chunked graphs |
|---|---:|---:|
| Native 16-token decode, median | 28.83 token/s | 1.39 token/s |
| TTFT, 40-token prompt | 1.01–1.03 s | 25.1 s |
| Peak host RAM | 1.28 GiB | 1.71 GiB |
| 32-token CPU BF16 reference gate | 32/32 | 32/32 |

Each Qwen token reads about 1 GB of BF16 weights (24 layers plus the
151,936-row LM head), so the engine is bound by weight streaming.

The acceptance prompt generated:

```text
The sky looks blue during the day because the Earth's atmosphere scatters the
sunlight in all directions, including blue light. When sunlight enters our
atmosphere, it encounters
```

With the default system prompt this prompt does not fit the 32-token prompt
budget, so the system prompt is dropped and the question is kept whole. Before
turn-boundary truncation, the prompt kept only the newest 32 tokens (the tail
of the system prompt) and generated "The sky appears blue during the day
because our eyes are sensitive to the color of the sky…".

The Qwen2.5 0.5B path is checked against both the NumPy BF16 runtime and
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
Consequently, the benchmark times 15 decode intervals for 16 emitted tokens
after TTFT. Historical sessions of the opt-in W8 chunked decoder
(`HAWKPOINT_DECODER_W8=1`) produced 14.42–19.99 token/s medians (20.20 token/s
best individual run), demonstrating the device's temperature/load sensitivity;
they are separate from the decoder-engine and chunked BF16 results above. The
older 18.42 token/s result timed an additional discarded successor step and is
historical decoder-step evidence, not a directly comparable end-to-end rate.

The decoder engine meets the requested 50–75 token/s range on this
benchmark. It runs every SmolLM layer in one dispatch and streams the weights
over six NPU channels at once; the chunked graphs issued 15 dispatches per
token and streamed one projection at a time. The measurements, the bandwidth
limits, and the design are in the [performance
analysis](docs/PERFORMANCE-ANALYSIS.md).

## What is included

- Interactive, multi-turn terminal chat with `/reset`, `/stats`, and `/exit`
- Four selectable checkpoints across the SmolLM and Qwen families
- A single-dispatch SmolLM decoder engine that streams weights over six
  parallel NPU channels
- Configurable NPU/CPU layer offload for hybrid execution
- OpenAI-compatible `GET /v1/models`
- OpenAI-compatible `POST /v1/chat/completions`
- Streaming chat completions over server-sent events
- Temperature, top-k, top-p, penalty, and seed sampling (greedy by default)
- Stop sequences and per-token log probabilities
- Prompt-prefix reuse across turns, so a follow-up message only prefills its
  new tokens
- One-command API or API + Open WebUI launcher
- Weight converter and hardware component/acceptance tests
- AIE2 C++ kernels and IRON graph definitions

Conversation history is retained by the client. When it no longer fits the
64-token hardware context, the host keeps the newest message whole for as long
as possible: it drops whole older turns first, then shortens the system prompt
from its end (dropping it if not even its first word fits), and only then
removes the oldest characters of the newest message. The chat template's
control tokens are always kept intact. Message text is encoded as plain text, so a
literal `<|im_end|>` in a message cannot close the turn or forge another role.

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

### Stop sequences and log probabilities

`stop` accepts a string or up to four strings. Generation ends as soon as the
generated text contains one of them; the stop sequence itself is not returned
and `finish_reason` is `stop`. Streaming holds back only text that could still
become a stop sequence.

`logprobs: true` returns OpenAI-style `choices[].logprobs.content` entries
(`token`, `logprob`, `bytes`, `top_logprobs`); `top_logprobs` (`0`–`20`) sets
how many alternatives each entry lists. Values come from the model's raw
distribution, before penalties, temperature, and top-k/top-p filtering.
Requesting log probabilities never changes which tokens are selected.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HAWKPOINT_API_KEY" \
  -d '{
    "model": "smollm2-135m-xdna1",
    "messages": [{"role": "user", "content": "Count to five."}],
    "max_tokens": 24,
    "stop": ["4"],
    "logprobs": true,
    "top_logprobs": 3
  }'
```

Text is streamed only at UTF-8 character boundaries, so characters split across
byte-level BPE tokens (for example `ç` or emoji) are never emitted as `�`.

### Prompt-prefix reuse

The decoder remembers which tokens produced its current K/V cache. When the
next prompt for the same model starts with those tokens, as a multi-turn chat
does, prefill resumes after the shared prefix. `usage.prompt_tokens_details.cached_tokens`
and `x_hawkpoint_stats.cached_prompt_tokens` report how many positions were
reused, and `ttft_seconds` covers only the positions actually computed. Reused
positions hold exactly the values a full prefill would compute, so outputs are
unchanged. Set `HAWKPOINT_PREFIX_CACHE=0` to always prefill the whole prompt.

The terminal chat exposes the sampling options:

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
python npu_llm/tests/test_stopping.py
python npu_llm/tests/test_tokenizer.py
python npu_llm/tests/test_decode_loop.py
python npu_llm/tests/test_engine_layout.py
python npu_llm/tests/test_cpu_backend.py
python tests/test_api_server.py
```

`test_sampling.py`, `test_stopping.py`, `test_tokenizer.py`, and
`test_decode_loop.py` need no NPU: token selection, stop sequences, the chat
template, and prefix reuse are host operations, covered with canned logits and
a small tokenizer trained inside the test.

Run the complete hardware acceptance test:

```bash
python npu_llm/tests/validate_chat_npu.py
python npu_llm/tests/validate_engine_npu.py
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

For SmolLM checkpoints running entirely on the NPU with BF16 weights, the
default decoder is the row-split engine in `designs/engine.py`:

1. Write the token's embedding row and RoPE table into a small header in the
   K/V cache buffer.
2. Run all 30 decoder layers and the final RMSNorm in one dispatch. Six GEMV
   tiles each read their own weight stream from DDR and own a fixed slice of
   the output rows of every projection; a MemTile joins their slices and a
   hub tile runs RoPE and attention and broadcasts each combined vector back.
   The hidden state stays inside the array between layers.
3. Append the new key/value rows the hub returns to the K/V cache.
4. Compute the 49k-row LM head with host BLAS and select the next token.

Qwen2.5 0.5B uses the same dataflow in `designs/qwen_engine.py`, with uneven
row ranges per tile, and also runs the final RMSNorm and the 151,936-row LM
head on the NPU, streaming FP32 logits to the host. Its release gate requires
the CPU BF16 reference's exact greedy tokens, so its kernels follow the
reference's FP32 arithmetic; the AIE2 vector unit has no FP32 multiply, so
`kernels/fp32_emulation.h` builds FP32 products from exact BF16 partial
products.

`HAWKPOINT_ENGINE=0` switches both models back to the chunked layer graphs
described below, which also run hybrid NPU/CPU placements and the opt-in W8
decoder. The engine's projections, norms, and final RMSNorm compute the same
BF16 values as the chunked path; its attention softmax runs on the vector unit
with a lookup-table exponential, so a few near-tie tokens can differ. See the
[performance analysis](docs/PERFORMANCE-ANALYSIS.md) for the measurements and
design.

With the chunked layer graphs, each generated token follows this path:

1. Copy the token embedding row into a persistent XRT buffer.
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
  shortened by the host at turn boundaries.
- **Padlocked to XDNA1 (`npu1`, AIE2, 4 columns).** XDNA2 silicon (`npu2`,
  AIE2P: Strix Point, Strix Halo — 8 columns, larger local memory, shared
  caches) is not targeted. The IRON graph layouts, tile counts, and ObjectFifo
  depths assume a 4-column array. Porting requires at minimum a re-parameterized
  IRON design and porting the AIE2 kernels to AIE2P.
- **No native BF16/FP16 tensor cores.** All AIE2 operations use a software
  BF16 multiply-accumulate path via `aie::mul` + `aie::accum`. There is no
  equivalent to NVIDIA Tensor Cores or Apple AMX blocks on this NPU
generation.
- **Two decoder layers per persistent invocation for the chunked graphs** —
  larger chunks of that design time out on the validated firmware, so
  Qwen2.5 0.5B stacks 12 chunks of 2 layers each. The SmolLM decoder engine
  uses a different dataflow and runs all 30 layers in one invocation.

### Model support

- Only the exact four checkpoint architectures listed above are supported.
  The converter rejects any model whose hidden size, intermediate size,
  layer count, attention layout, or vocabulary differ from a known template.
  SmolLM2-360M, SmolLM2-1.7B, larger Qwen variants, and non-Llama architectures
  fail at conversion time, not silently at runtime.
- **Greedy by default, with opt-in sampling**. `temperature`, `top_p`,
  `top_k`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, and
  `seed` are supported. A request that sets none of them decodes greedily and
  reproduces the checked-in acceptance sequences exactly. Stop sequences and
  `logprobs` are supported; `n > 1` and beam search are not.

### Performance

- **Qwen2.5 0.5B is bound by weight streaming.** Its engine reaches 28.8
  token/s (the CPU BF16 reference runs at about 3.8 token/s on the same
  machine); each token moves about 1 GB of BF16 weights. The chunked Qwen
  graphs (`HAWKPOINT_ENGINE=0`) remain much slower, at 1.4 token/s. See
  [Performance Analysis](docs/PERFORMANCE-ANALYSIS.md).
- The **Ollama XDNA backend** now packs GGML rows once and retains dense tiles
  and selected MoE experts in a bounded persistent cache. Unchanged decode
  steps no longer repeat CPU dequantize/requantize or weight DMA. Native
  Q4_K/Q6_K AIE2 kernels are included under `npu_llm/kernels/` and can be
  compiled with `ollama-xdna/backend/compile_quantized.py`; their xclbins are
  opt-in until the physical release gates validate the exact toolchain stack.
- **The SmolLM engine is limited by weight streaming and the host LM head.**
  One token moves about 214 MB of BF16 weights through six streams (7.6-8.9 ms
  of NPU time depending on the position), and the CPU LM head adds about
  5 ms. Moving the LM head onto the NPU and an int8-weight engine are the
  measured next steps.
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

## Star History

<a href="https://www.star-history.com/?repos=c8dhjp4tyv-bit%2Fhawkpoint-npu-llm&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=c8dhjp4tyv-bit/hawkpoint-npu-llm&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=c8dhjp4tyv-bit/hawkpoint-npu-llm&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=c8dhjp4tyv-bit/hawkpoint-npu-llm&type=date&legend=top-left" />
 </picture>
</a>

