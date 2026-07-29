# Ollama XDNA1 offload

This experimental Linux integration patches a normal upstream Ollama checkout
so Qwen operations can be shared across CPU, GPU, and the first-generation AMD
Ryzen AI NPU. It adds an external GGML backend for `amdxdna`/XRT and preserves
Ollama's CLI, REST API, model store, and Open WebUI compatibility.

Validated combination:

| Component | Validated value |
|---|---|
| APU/NPU | AMD Ryzen 7 250, Hawk Point XDNA1 |
| NPU device | `/dev/accel/accel0`, `amdxdna` |
| Linux | Fedora 45 prerelease, kernel 7.2-rc5 |
| Ollama | `v0.32.5` |
| Model | `qwen3-coder:30b` |
| GPU | NVIDIA RTX 5060 Laptop, CUDA 13 |
| Placement | `23%/33%/44% CPU/GPU/NPU` |

The NPU path is real hardware execution, but it is not yet a performance win.
Read [Known limitations](#known-limitations) before installing.

## 1. Install build dependencies

Clone the project first:

```bash
git clone https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm.git
cd hawkpoint-npu-llm
```

Use the distribution-aware installer:

```bash
./ollama-xdna/scripts/install-deps.sh
```

For unattended package-manager confirmation:

```bash
./ollama-xdna/scripts/install-deps.sh --yes
```

The script detects `/etc/os-release` and supports:

- Fedora/RHEL family with `dnf`
- Ubuntu/Debian family with `apt-get`
- Arch family with `pacman`
- openSUSE family with `zypper`

It installs GCC/G++, Make, CMake, Ninja, ccache, Git, Go, curl, jq, pkg-config,
polkit, and PCI utilities. Ollama v0.32.5 requires Go 1.26 or newer. If a
distribution ships an older Go release, install a current Go toolchain from
[go.dev](https://go.dev/doc/install), then run the verifier again.

The script intentionally does not install an NPU kernel driver, firmware, or
XRT from an unverified third-party repository. Those components must match one
another and a wrong combination can stop the NPU from loading.

### Manual distro commands

Fedora/RHEL:

```bash
sudo dnf install gcc gcc-c++ make cmake ninja-build ccache git golang curl jq \
  pkgconf-pkg-config polkit pciutils
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build ccache git golang-go \
  curl jq \
  pkg-config policykit-1 pciutils
```

Arch:

```bash
sudo pacman -S --needed base-devel cmake ninja ccache git go curl jq pkgconf \
  polkit pciutils
```

openSUSE:

```bash
sudo zypper install gcc gcc-c++ make cmake ninja ccache git go curl jq \
  pkg-config polkit pciutils
```

## 2. Install and verify the XDNA host stack

You need all of the following before building:

- A kernel with the `amdxdna` driver loaded
- Matching Ryzen AI NPU firmware
- `/dev/accel/accel0` accessible to the current user
- XRT runtime and development files under `/opt/xilinx/xrt`
- `/opt/xilinx/xrt/include/xrt/xrt_bo.h`
- `/opt/xilinx/xrt/lib64/libxrt_coreutil.so`

Use AMD/Xilinx's XRT installation documentation for the runtime and development
packages. XRT 2026.1 and later split development headers into a separate
`-dev` Debian package or `-devel` RPM package:

<https://xilinx.github.io/XRT/master/html/install.html>

Verify the complete host:

```bash
./ollama-xdna/scripts/verify-system.sh
```

Every required line must report `PASS`. Useful manual checks are:

```bash
lspci -nnk | grep -A3 -i 'Signal processing controller'
ls -l /dev/accel/accel0
```

If device permissions are wrong, add the user to the group assigned to
`/dev/accel/accel0`, log out, and log back in. Do not make the device
world-writable as a permanent fix.

## 3. Patch a normal Ollama checkout, build, and install

The build script does not replace files inside an existing Ollama source tree.
By default it reuses the already installed, version-matched Ollama native
runtime. This avoids a large CPU/CUDA compilation and builds only the patched Go
binary plus the small XDNA backend. It performs these steps in a temporary
directory:

1. Clones the selected tag from `ollama/ollama`.
2. Applies `patches/ollama-v0.32.5-xdna.patch`.
3. Formats and tests all modified Go packages.
4. Confirms that installed Ollama matches the patch version and stages a copy
   of its native CPU/GPU runtime.
5. Builds `libggml-xdna.so` against matching llama.cpp headers and the staged
   GGML ABI.
6. Runs the standalone 8-expert XDNA hardware correctness test.
7. Confirms that staged `llama-server --list-devices` enumerates `XDNA0`.
8. Requests administrator authorization only after all validation passes.
9. Backs up the existing Ollama binary/runtime and installs atomically.
10. Restarts `ollama.service` and rolls back automatically if its health check
    fails.

CPU + NPU:

```bash
./ollama-xdna/scripts/build-and-install.sh --backend cpu
```

NVIDIA CUDA 13 + CPU + NPU:

```bash
./ollama-xdna/scripts/build-and-install.sh --backend cuda_v13
```

Other upstream Ollama GPU build choices:

```bash
# NVIDIA CUDA 12
./ollama-xdna/scripts/build-and-install.sh --backend cuda_v12

# AMD Radeon ROCm 7.2
./ollama-xdna/scripts/build-and-install.sh --backend rocm_v7_2

# Vulkan
./ollama-xdna/scripts/build-and-install.sh --backend vulkan
```

CUDA, ROCm, and Vulkan SDKs are optional and are not installed by the dependency
script. They are not needed when reusing a working installed Ollama runtime.
Install the selected GPU SDK only when using `--rebuild-runtime`. The NPU path
itself does not require a discrete GPU.

To rebuild Ollama's complete CPU/GPU native payload instead of reusing the
installed version, add `--rebuild-runtime`. This is primarily for developers
and can require many gigabytes of RAM and swap:

```bash
./ollama-xdna/scripts/build-and-install.sh \
  --backend cuda_v13 \
  --rebuild-runtime \
  --jobs 1
```

Fedora development releases may ship GCC newer than CUDA's supported host
compiler. When CUDA is selected and GCC is newer than 15, the build script adds
NVCC's `-allow-unsupported-compiler` flag and prints a warning. For a production
system, a CUDA-supported GCC toolchain is safer than relying on that override.

To build and validate without changing the installed Ollama:

```bash
./ollama-xdna/scripts/build-and-install.sh \
  --backend cuda_v13 \
  --no-install \
  --build-root "$PWD/ollama-xdna-build"
```

An interrupted named build can continue without cloning and recompiling
finished objects:

```bash
./ollama-xdna/scripts/build-and-install.sh \
  --backend cuda_v13 \
  --no-install \
  --build-root "$PWD/ollama-xdna-build" \
  --resume
```

The builder uses a fixed eight parallel jobs by default instead of `nproc`, so
high-thread-count CPUs do not launch one compiler per logical CPU. Override it
with `--jobs` when needed. For a full native Ollama/CUDA rebuild on a low-memory
desktop, use fewer jobs:

```bash
./ollama-xdna/scripts/build-and-install.sh \
  --backend cuda_v13 \
  --rebuild-runtime \
  --jobs 1
```

Increase `--jobs` only when enough free RAM is available. The same value limits
CMake/Ninja and Go parallelism. `ccache` speeds up interrupted and repeated
builds without increasing active compiler concurrency.

The only supported patch base is currently `v0.32.5`. This is deliberate:
external GGML backends must be rebuilt against the exact llama.cpp ABI bundled
with each Ollama version.

## 4. Run Qwen and prove NPU dispatch

Existing Ollama models remain in the normal Ollama model store. For the
validated model:

```bash
ollama pull qwen3-coder:30b
ollama run qwen3-coder:30b
```

In another terminal:

```bash
ollama ps
```

The validated machine reports:

```text
NAME               PROCESSOR
qwen3-coder:30b    23%/33%/44% CPU/GPU/NPU
```

The three numbers are normalized to 100%. They estimate active linear compute
from model placement and operation counts; they are not instantaneous hardware
utilization or memory percentages. XDNA1 has no dedicated VRAM. Its buffers use
system RAM.

Confirm actual hardware dispatch:

```bash
journalctl -u ollama -f | grep --line-buffered \
  'XDNA expert offload count'
```

Expected lines include:

```text
XDNA expert offload count: 1
XDNA expert offload count: 144
```

Confirm the API fields:

```bash
curl -s http://127.0.0.1:11434/api/ps |
  jq '.models[] | {name, npu, npu_percent}'
```

Expected for the validated placement:

```json
{
  "name": "qwen3-coder:30b",
  "npu": true,
  "npu_percent": 44
}
```

## 5. API and Open WebUI

The patched server keeps Ollama's normal API:

```bash
curl http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-coder:30b",
    "prompt": "Reply with only OK",
    "stream": false
  }'
```

No Open WebUI-specific patch is needed. Point Open WebUI at the same Ollama
endpoint. If Open WebUI runs in Docker on Linux, its container normally reaches
the host using `http://host.docker.internal:11434`; add the host-gateway mapping
when the container runtime does not provide that name.

Do not expose port 11434 directly to an untrusted network. Ollama's local API
does not provide authentication.

## Model compatibility

The backend selects operations by tensor type and shape, not by an Ollama model
filename.

- Qwen MoE: one to eight selected experts, input/output dimensions up to 2048
- Dense Qwen: tiled single-token matrix/vector projections
- Weight types: F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K
- Unsupported operations and shapes remain on Ollama's CPU/GPU backends

Hardware validation is strongest for `qwen3-coder:30b`. Dense
`qwen2.5-coder:7b` produced the correct tested answer but was slower than its
CPU/GPU path. Treat other Qwen architectures as experimental until correctness
and performance are measured for that exact model.

## Updating Ollama

To rebuild the currently supported release:

```bash
OLLAMA_XDNA_GPU_BACKEND=cuda_v13 \
  ./ollama-xdna/scripts/update-ollama-xdna.sh v0.32.5
```

To attempt the newest upstream release:

```bash
OLLAMA_XDNA_GPU_BACKEND=cuda_v13 \
  ./ollama-xdna/scripts/update-ollama-xdna.sh latest
```

For a newer tag, the updater attempts Git's three-way patch application, then
repeats all tests and ABI-matched builds. Any patch conflict, compilation
failure, hardware-test failure, or staged device-discovery failure stops before
the installed Ollama is touched. A successful build still requires explicit
administrator authorization.

An unvalidated tag is not automatically declared supported merely because it
builds. Re-run model correctness, dispatch, and performance tests before
publishing results.

## Recovery

Restore the newest pre-XDNA Ollama backup:

```bash
pkexec /usr/local/lib/ollama/xdna/rollback.sh
```

Backups use timestamped paths:

```text
/usr/local/bin/ollama.pre-xdna-YYYYMMDD-HHMMSS
/usr/local/lib/ollama.pre-xdna-YYYYMMDD-HHMMSS
/etc/systemd/system/ollama.service.d/xdna.conf.pre-xdna-YYYYMMDD-HHMMSS
```

The installer also performs this rollback automatically if the restarted API
does not become healthy within 30 seconds.

## Rebuild the AIE program

The repository includes a small prebuilt XDNA1 `xclbin` and instruction stream
so users do not need MLIR-AIE merely to run it. The corresponding IRON source is
`backend/compile_experts.py`, and its AIE kernel is
`../npu_llm/kernels/project_w8bf16.cc`.

To rebuild it, use the project's validated MLIR-AIE/Peano environment:

```bash
source /path/to/mlir-aie/ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
source /path/to/mlir-aie/utils/env_setup.sh \
  /path/to/mlir-aie /path/to/mlir-aie/peano
python ollama-xdna/backend/compile_experts.py
```

## Known limitations

- The current expert path converts selected GGML rows to W8 on the CPU and
  streams padded weights for every decoded token.
- That conversion and transfer overhead makes this proof-of-execution slower
  than the optimized CPU/GPU path on the validated machine.
- Prompt processing with batches larger than one usually remains on CPU/GPU;
  the NPU backend primarily targets single-token decoding operations.
- The `npu_percent` field is an operation-count estimate for Qwen3 MoE, not a
  reading from an NPU utilization counter.
- Only Linux/XDNA1 is supported. XDNA2 and Windows are not validated.
- The patch is maintained out of tree and may require adaptation after Ollama
  or its bundled llama.cpp changes ABI.

The highest-value optimization is a Q4_K/Q6_K-aware AIE kernel with persistent
or reusable packed weights. It would remove CPU dequantize/requantize work and
greatly reduce per-token host-to-NPU traffic.
