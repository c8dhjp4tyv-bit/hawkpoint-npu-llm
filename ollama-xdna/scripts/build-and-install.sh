#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORTED_TAG="v0.32.5"
tag="${SUPPORTED_TAG}"
backend="cpu"
build_root=""
install_result=1
allow_unsupported=0
resume=0
jobs="${OLLAMA_XDNA_JOBS:-8}"
rebuild_runtime=0

die() {
    echo "error: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: $0 [options]
  --tag TAG             Ollama tag (default: ${SUPPORTED_TAG})
  --backend BACKEND     cpu, cuda_v12, cuda_v13, rocm_v7_2, or vulkan
  --build-root PATH     Keep build files at PATH
  --resume              Continue an interrupted --build-root build
  --jobs COUNT          Parallel jobs (default: 8)
  --rebuild-runtime     Rebuild CPU/GPU payload instead of reusing installed one
  --no-install          Build and validate without replacing system Ollama
  --allow-unsupported   Try the patch on a newer Ollama tag
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --tag)
            [[ "$#" -ge 2 ]] || die "--tag requires a value"
            tag="$2"
            shift 2
            ;;
        --backend)
            [[ "$#" -ge 2 ]] || die "--backend requires a value"
            backend="$2"
            shift 2
            ;;
        --build-root)
            [[ "$#" -ge 2 ]] || die "--build-root requires a value"
            build_root="$2"
            shift 2
            ;;
        --no-install)
            install_result=0
            shift
            ;;
        --resume)
            resume=1
            shift
            ;;
        --jobs)
            [[ "$#" -ge 2 ]] || die "--jobs requires a value"
            jobs="$2"
            shift 2
            ;;
        --rebuild-runtime)
            rebuild_runtime=1
            shift
            ;;
        --allow-unsupported)
            allow_unsupported=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "${backend}" in
    cpu|cuda_v12|cuda_v13|rocm_v7_2|vulkan) ;;
    *) die "unsupported backend: ${backend}" ;;
esac

[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"

if [[ "${tag}" != "${SUPPORTED_TAG}" && "${allow_unsupported}" -ne 1 ]]; then
    die "${tag} is unvalidated; use --allow-unsupported to attempt a 3-way apply"
fi

"${PROJECT_ROOT}/scripts/verify-system.sh"

if [[ -z "${build_root}" ]]; then
    build_root="$(mktemp -d "${TMPDIR:-/tmp}/ollama-xdna.XXXXXXXX")"
else
    mkdir -p "${build_root}"
    build_root="$(realpath "${build_root}")"
fi

source_dir="${build_root}/source"
native_build="${build_root}/native"
stage="${build_root}/stage"
patch_file="${PROJECT_ROOT}/patches/ollama-v0.32.5-xdna.patch"
export CMAKE_BUILD_PARALLEL_LEVEL="${jobs}"
export GOMAXPROCS="${jobs}"

if [[ "${resume}" -eq 1 ]]; then
    [[ -d "${source_dir}/.git" ]] ||
        die "--resume requested but ${source_dir} is not a Git checkout"
    expected_commit="$(git -C "${source_dir}" rev-list -n1 "${tag}")"
    actual_commit="$(git -C "${source_dir}" rev-parse HEAD)"
    [[ "${actual_commit}" == "${expected_commit}" ]] ||
        die "resume checkout is not based on ${tag}"
    git -C "${source_dir}" diff --check
else
    [[ ! -e "${source_dir}" ]] || die "${source_dir} already exists"
    git clone --quiet --filter=blob:none --branch "${tag}" \
        https://github.com/ollama/ollama.git "${source_dir}"

    if [[ "${tag}" == "${SUPPORTED_TAG}" ]]; then
        git -C "${source_dir}" apply --check "${patch_file}"
        git -C "${source_dir}" apply "${patch_file}"
    else
        git -C "${source_dir}" apply --3way "${patch_file}" ||
            die "the patch conflicts with ${tag}; installed Ollama was not changed"
    fi
fi

gofmt -w \
    "${source_dir}/api/types.go" \
    "${source_dir}/cmd/cmd.go" \
    "${source_dir}/cmd/cmd_test.go" \
    "${source_dir}/discover/llama_server.go" \
    "${source_dir}/discover/llama_server_test.go" \
    "${source_dir}/llm/llama_server.go" \
    "${source_dir}/llm/llama_server_test.go" \
    "${source_dir}/server/routes.go" \
    "${source_dir}/server/routes_test.go"

(
    cd "${source_dir}"
    go test ./api ./cmd ./discover ./llm ./server
)

if [[ "${rebuild_runtime}" -eq 1 ]]; then
    cmake_args=(
        -S "${source_dir}"
        -B "${native_build}"
        -G Ninja
        -DCMAKE_BUILD_TYPE=Release
        "-DOLLAMA_BUILD_PARALLEL=${jobs}"
    )
    if [[ "${backend}" != "cpu" ]]; then
        cmake_args+=("-DOLLAMA_LLAMA_BACKENDS=${backend}")
    fi
    if [[ "${backend}" == cuda_* ]]; then
        cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=native")
        gcc_major="$(g++ -dumpfullversion | cut -d. -f1)"
        if [[ "${gcc_major}" -gt 15 ]]; then
            echo "warning: CUDA does not officially support GCC ${gcc_major};" \
                "using -allow-unsupported-compiler"
            cmake_args+=("-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler")
        fi
    fi

    cmake "${cmake_args[@]}"
    cmake --build "${native_build}" --parallel "${jobs}"
    cmake --install "${native_build}" --prefix "${stage}"
    llama_source="${native_build}/_deps/llama_cpp-src"
    runtime_source="${native_build}/lib/ollama"
else
    [[ -x /usr/local/bin/ollama ]] ||
        die "installed Ollama is missing; use --rebuild-runtime"
    [[ -x /usr/local/lib/ollama/llama-server ]] ||
        die "installed Ollama runtime is missing; use --rebuild-runtime"
    installed_version="$(/usr/local/bin/ollama --version | awk '{print $NF}')"
    [[ "${installed_version}" == "${tag#v}"* ]] ||
        die "installed Ollama ${installed_version} does not match ${tag#v}"

    install -d "${stage}/bin" "${stage}/lib/ollama"
    cp -a /usr/local/lib/ollama/. "${stage}/lib/ollama/"
    runtime_source="/usr/local/lib/ollama"

    if [[ -d "${native_build}/_deps/llama_cpp-src/ggml/include" ]]; then
        llama_source="${native_build}/_deps/llama_cpp-src"
    else
        llama_source="${build_root}/llama.cpp"
        llama_commit="$(tr -d '[:space:]' < "${source_dir}/LLAMA_CPP_VERSION")"
        if [[ ! -d "${llama_source}/.git" ]]; then
            git clone --quiet --filter=blob:none \
                https://github.com/ggml-org/llama.cpp.git "${llama_source}"
        fi
        git -C "${llama_source}" fetch --quiet origin "${llama_commit}"
        git -C "${llama_source}" checkout --quiet "${llama_commit}"
    fi
fi

(
    cd "${source_dir}"
    go build -trimpath \
        -ldflags "-X github.com/ollama/ollama/version.Version=${tag#v}-xdna" \
        -o "${stage}/bin/ollama" .
)
[[ -d "${llama_source}/ggml/include" ]] ||
    die "matching llama.cpp headers were not generated"
[[ -x "${stage}/lib/ollama/llama-server" ]] ||
    die "staged llama-server is missing"

g++ -O3 -DNDEBUG -std=c++17 -fPIC -shared -DGGML_BACKEND_DL \
    -Wall -Wextra -Wpedantic \
    "${PROJECT_ROOT}/backend/ggml-xdna.cpp" \
    -I"${llama_source}/ggml/include" \
    -I"${llama_source}/ggml/src" \
    -isystem /opt/xilinx/xrt/include \
    -L"${runtime_source}" -lggml -lggml-base \
    -L/opt/xilinx/xrt/lib64 -lxrt_coreutil \
    -Wl,-rpath,/usr/local/lib/ollama:/opt/xilinx/xrt/lib64 \
    -o "${build_root}/libggml-xdna.so"

g++ -O2 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic \
    "${PROJECT_ROOT}/backend/test_experts.cpp" \
    -isystem /opt/xilinx/xrt/include \
    -L/opt/xilinx/xrt/lib64 -lxrt_coreutil \
    -Wl,-rpath,/opt/xilinx/xrt/lib64 \
    -o "${build_root}/test-experts"

install -d "${stage}/lib/ollama/xdna"
install -m 0755 "${build_root}/libggml-xdna.so" \
    "${stage}/lib/ollama/xdna/libggml-xdna.so"
install -m 0644 \
    "${PROJECT_ROOT}/backend/artifacts/experts-8x2048x2048/experts.xclbin" \
    "${stage}/lib/ollama/xdna/experts.xclbin"
install -m 0644 \
    "${PROJECT_ROOT}/backend/artifacts/experts-8x2048x2048/insts.bin" \
    "${stage}/lib/ollama/xdna/insts.bin"
ln -sfn xdna/libggml-xdna.so \
    "${stage}/lib/ollama/libggml-openvino.so"

"${build_root}/test-experts" \
    "${stage}/lib/ollama/xdna/experts.xclbin" \
    "${stage}/lib/ollama/xdna/insts.bin"

LD_LIBRARY_PATH="${stage}/lib/ollama:${stage}/lib/ollama/${backend}" \
GGML_BACKEND_PATH="${stage}/lib/ollama/xdna/libggml-xdna.so" \
GGML_XDNA_XCLBIN="${stage}/lib/ollama/xdna/experts.xclbin" \
GGML_XDNA_INSTS="${stage}/lib/ollama/xdna/insts.bin" \
    "${stage}/lib/ollama/llama-server" --list-devices |
    grep -q 'XDNA0' ||
    die "staged llama-server did not enumerate XDNA0"

echo "Validated stage: ${stage}"
if [[ "${install_result}" -eq 1 ]]; then
    pkexec "${PROJECT_ROOT}/scripts/install-stage.sh" \
        "${stage}" "${tag#v}" "${backend}"
else
    echo "System Ollama was not changed (--no-install)."
fi
