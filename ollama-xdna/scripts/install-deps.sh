#!/usr/bin/env bash
set -euo pipefail

yes_flag=()
if [[ "${1:-}" == "--yes" ]]; then
    yes_flag=(-y)
elif [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--yes]" >&2
    exit 2
fi

if [[ ! -r /etc/os-release ]]; then
    echo "error: /etc/os-release is missing; install the prerequisites manually" >&2
    exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
distro="${ID:-unknown} ${ID_LIKE:-}"

case "${distro}" in
    *fedora*|*rhel*)
        sudo dnf install "${yes_flag[@]}" \
            gcc gcc-c++ make cmake ninja-build ccache git golang curl jq \
            pkgconf-pkg-config polkit pciutils
        ;;
    *ubuntu*|*debian*)
        sudo apt-get update
        sudo apt-get install "${yes_flag[@]}" \
            build-essential cmake ninja-build ccache git golang-go curl jq \
            pkg-config policykit-1 pciutils
        ;;
    *arch*)
        sudo pacman -S --needed "${yes_flag[@]}" \
            base-devel cmake ninja ccache git go curl jq pkgconf polkit \
            pciutils
        ;;
    *opensuse*|*suse*)
        sudo zypper install "${yes_flag[@]}" \
            gcc gcc-c++ make cmake ninja ccache git go curl jq pkg-config \
            polkit pciutils
        ;;
    *)
        cat >&2 <<EOF
error: unsupported distribution: ${PRETTY_NAME:-unknown}
Install GCC or Clang, CMake >= 3.24, Ninja, Git, Go >= 1.26, curl, jq,
pkg-config, and polkit, then run verify-system.sh.
EOF
        exit 1
        ;;
esac

"$(dirname "${BASH_SOURCE[0]}")/verify-system.sh"
