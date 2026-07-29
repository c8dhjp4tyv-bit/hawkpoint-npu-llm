#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tag="${1:-latest}"
backend="${OLLAMA_XDNA_GPU_BACKEND:-cpu}"

if [[ "${tag}" == "latest" ]]; then
    tag="$(
        curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest |
            jq -r '.tag_name'
    )"
fi

[[ -n "${tag}" && "${tag}" != "null" ]] || {
    echo "error: could not resolve the requested Ollama release" >&2
    exit 1
}

exec "${SCRIPT_DIR}/build-and-install.sh" \
    --tag "${tag}" \
    --backend "${backend}" \
    --allow-unsupported
