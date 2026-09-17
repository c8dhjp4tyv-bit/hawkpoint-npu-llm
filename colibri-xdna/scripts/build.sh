#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT="a8f2ca623ffe9de9df11d56f34d11d2d501493d3"
build_root=""
source_dir=""
jobs="${COLIBRI_XDNA_JOBS:-8}"
test_npu=0

die() { echo "error: $*" >&2; exit 1; }
usage() {
    echo "Usage: $0 [--source PATH] [--build-root PATH] [--jobs N] [--test-npu]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) [[ $# -ge 2 ]] || die "--source requires a path"; source_dir="$2"; shift 2 ;;
        --build-root) [[ $# -ge 2 ]] || die "--build-root requires a path"; build_root="$2"; shift 2 ;;
        --jobs) [[ $# -ge 2 ]] || die "--jobs requires a value"; jobs="$2"; shift 2 ;;
        --test-npu) test_npu=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
[[ -r /opt/xilinx/xrt/include/xrt/xrt_bo.h ]] || die "XRT headers are missing"
[[ -r /opt/xilinx/xrt/lib64/libxrt_coreutil.so ]] || die "XRT runtime is missing"

if [[ -z "$build_root" ]]; then
    build_root="$(mktemp -d "${TMPDIR:-/tmp}/colibri-xdna.XXXXXXXX")"
else
    mkdir -p "$build_root"
    build_root="$(realpath "$build_root")"
fi
if [[ -z "$source_dir" ]]; then
    source_dir="$build_root/colibri"
    git clone --quiet --filter=blob:none https://github.com/JustVugg/colibri.git "$source_dir"
    git -C "$source_dir" checkout --quiet "$COMMIT"
else
    source_dir="$(realpath "$source_dir")"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$COMMIT" ]] ||
    die "Colibri checkout must be pinned to $COMMIT"
[[ -z "$(git -C "$source_dir" status --porcelain)" ]] ||
    die "Colibri checkout must be clean"

patch="$ROOT/patches/colibri-a8f2ca6-xdna.patch"
git -C "$source_dir" apply --check "$patch"
git -C "$source_dir" apply "$patch"
install -m 0644 "$ROOT/backend/backend_xdna.cpp" "$source_dir/c/backend_xdna.cpp"
make -C "$source_dir/c" colibri XDNA=1 XRT_ROOT=/opt/xilinx/xrt -j"$jobs"

if [[ "$test_npu" -eq 1 ]]; then
    test_bin="$build_root/test_backend_xdna"
    g++ -std=c++17 -O2 -Wall -Wextra -Wno-ignored-qualifiers \
        -I"$source_dir/c" -I/opt/xilinx/xrt/include \
        "$ROOT/backend/backend_xdna.cpp" "$ROOT/backend/test_backend_xdna.cpp" \
        -L/opt/xilinx/xrt/lib64 -Wl,-rpath,/opt/xilinx/xrt/lib64 \
        -lxrt_coreutil -pthread -o "$test_bin"
    "$test_bin" \
        "$ROOT/../ollama-xdna/backend/artifacts/experts-8x2048x2048/experts.xclbin" \
        "$ROOT/../ollama-xdna/backend/artifacts/experts-8x2048x2048/insts.bin"
fi

echo "Built Colibri XDNA C engine: $source_dir/c/colibri"
