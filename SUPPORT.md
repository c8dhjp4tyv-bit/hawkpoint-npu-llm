# Support matrix

This is the tested boundary for `v0.1.0-alpha.1`. Anything outside it is
unsupported until a reproducible hardware report is added.

| Component | Validated | Status |
|---|---|---|
| AMD NPU | Hawk Point XDNA1, `RyzenAI-npu1`, AIE2, 4 columns | Native and Ollama paths tested |
| Phoenix XDNA1 | Same `npu1` architecture | Expected compatible; needs an independent report |
| XDNA2 / Strix Point | No | Unsupported |
| OS | Fedora Linux, x86-64, systemd | Validated |
| NPU firmware | `1.5.5.391` | Validated |
| Ollama | `v0.32.5`, commit `eec8e0b9458b8a01be0c216a9cc53eefde24ef50` | Patch target |
| GPU backends | CPU; CUDA v12/v13, ROCm v7.2, Vulkan build options | CPU and CUDA v13 exercised locally; others need reports |
| Python | CPython 3.12 | CI target |
| Open WebUI | `v0.11.0`, OCI index `sha256:72c0ba641ba75e7aa52655cb242570906ececd09b1140fb736483038a22b3228` | Pinned tag and digest |

Exact native checkpoint revisions are recorded in
`npu_llm/model_catalog.py`. See the README for the fixed model architecture
constraints.

## Release gate

A release candidate is accepted only after hosted CI passes. A maintainer with
a labeled Hawk Point runner must also run the manual `XDNA1 hardware
acceptance` workflow for fresh dependency installation, pinned model download,
conversion, token agreement, model switching, a 1,000-completion soak, Ollama
build validation, and rollback-script validation. Until that external hardware
run is attached to a release, the release remains alpha rather than
production-ready.
