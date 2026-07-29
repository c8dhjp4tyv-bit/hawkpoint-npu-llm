# Hawk Point NPU LLM

Run compatible SmolLM 135M transformer checkpoints on the first-generation AMD
XDNA NPU in Phoenix and Hawk Point Ryzen AI processors. The implementation uses
MLIR-AIE/IRON and targets the `npu1` AIE2 array directly.

The GPU is not used. The CPU handles tokenization, orchestration, final argmax,
printing, and reference validation. Decoder projections, RMSNorm, RoPE,
attention, KV caches, residuals, and SwiGLU execute as AIE2 kernels.

> Experimental research project. It is not affiliated with or supported by
> AMD, Xilinx, Hugging Face, or the Open WebUI project.

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

## What is included

- Interactive, multi-turn terminal chat with `/reset`, `/stats`, and `/exit`
- Three selectable SmolLM 135M checkpoints
- OpenAI-compatible `GET /v1/models`
- OpenAI-compatible `POST /v1/chat/completions`
- Streaming chat completions over server-sent events
- One-command API or API + Open WebUI launcher
- SmolLM2 weight converter and hardware acceptance tests
- AIE2 C++ kernels and IRON graph definitions

Conversation history is retained by the client and automatically trimmed to
the newest tokens that fit the current 64-token hardware context.

## Supported models

| API model ID | Hugging Face checkpoint | Intended use |
|---|---|---|
| `smollm2-135m-xdna1` | `HuggingFaceTB/SmolLM2-135M-Instruct` | Default assistant |
| `smollm-135m-xdna1` | `HuggingFaceTB/SmolLM-135M-Instruct` | Previous-generation assistant |
| `smollm2-135m-sft-xdna1` | `HuggingFaceTB/smollm2-135M-SFT-Only` | SFT comparison/research |

The converter rejects checkpoints whose hidden size, intermediate size, layer
count, attention layout, or vocabulary do not match the fixed hardware graph.
SmolLM2-360M and SmolLM2-1.7B are therefore not silently accepted.

## Requirements

- A Phoenix or Hawk Point Ryzen AI system exposing `RyzenAI-npu1`
- Linux with a working `amdxdna`/XRT stack and matching NPU firmware
- A working MLIR-AIE/IRON environment with Peano
- Python 3.12 packages listed in `requirements.txt`
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
python -m pip install -r requirements.txt
python scripts/prepare_model.py
```

`prepare_model.py` downloads the upstream Hugging Face model and creates the
XDNA1 runtime representation under `npu_llm/models/`. Model weights are not
stored in this Git repository.

Prepare every supported model:

```bash
python scripts/prepare_model.py --all
```

Or choose one or more explicitly:

```bash
python scripts/prepare_model.py \
  --model smollm2-135m-xdna1 \
  --model smollm-135m-xdna1
```

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
```

The Open WebUI container is preconfigured to reach the host API at
`http://host.docker.internal:8000/v1`. Its model picker displays every
installed checkpoint returned by `/v1/models`. Its data is kept in a Docker
volume.

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

No API key is required. Clients that require one may use any non-empty value,
for example `local-npu`.

```bash
curl http://localhost:8000/v1/models
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-npu" \
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
silently routed to a different checkpoint. Models are loaded lazily, and only
one checkpoint is retained by the server at a time to limit host RAM usage.
The first request after switching models includes its load/compile cost.

The server serializes requests because one physical NPU execution context is
shared. It binds to `127.0.0.1` in API-only mode and `0.0.0.0` when it must be
reachable from the local Open WebUI container. Open WebUI itself is published
only on `127.0.0.1:3000`. The API has no authentication, so use the Open WebUI
mode only on a trusted network or protect port 8000 with a firewall.

## Convert a model manually

```bash
python npu_llm/tools/convert_smollm2.py \
  npu_llm/models/SmolLM2-135M-Instruct \
  npu_llm/models/SmolLM2-135M-Instruct-xdna1-w8a16
```

The converted directory is approximately 442 MiB. It retains BF16 decoder
weights for numerically stable generation and per-output-channel INT8 weights
for the W8/BF16 kernels and LM head.

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

Component examples:

```bash
python npu_llm/tests/test_elementwise_npu.py
python npu_llm/designs/rmsnorm.py --dev npu --size 576 -w 2 -i 5
python npu_llm/designs/rope.py --dev npu --heads 12 --position 7 -w 2 -i 5
```

## Architecture

Each generated token follows this path:

1. Fetch the token embedding into an XRT buffer.
2. Run all 30 SmolLM2 decoder layers using the fused AIE2 layer graph.
3. Keep the 64-token K/V cache in NPU-addressable XRT buffers.
4. Apply final RMSNorm and the W8/BF16 LM head on the NPU.
5. Copy logits to the host for argmax and token decoding.

The decoder-layer xclbin is compiled once and reused for all layers and token
positions. Layer weights, hidden states, and K/V caches remain in XRT buffer
objects; there is no CPU tensor-operation fallback.

## Limitations

- The attention kernel currently has a fixed 64-token context.
- Only the compatible SmolLM 135M checkpoints listed above are supported.
- Generation is greedy; sampling parameters are accepted neither by the
  runtime nor the API.
- The Python host currently invokes the resident decoder once per layer and
  token. A multi-layer sequence exists experimentally, but current XRT firmware
  is reliable with only two chained task groups.
- The project targets `npu1`; it is not an XDNA2 implementation.

## License

Apache License 2.0 with LLVM exception. See `LICENSE`.

SmolLM2 model files are downloaded separately and remain subject to their own
upstream license and terms.
