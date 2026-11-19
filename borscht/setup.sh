#!/usr/bin/env bash
# Installs everything needed to build, test, and hack on borscht:
#
#     ./borscht/setup.sh            # from anywhere in the repo
#     cd borscht && uv run pytest   # make your edits to python/ or rust/
#                                   # uv rebuilds the extension when rust/ changes
#                                   # and cut a PR
#
#     ./borscht/setup.sh --no-dev   # skip the dev group (torch and its several GB
#                                   # of nvidia-* wheels), as CI's GPU-less jobs do
#
# CI (.github/workflows/borscht.yml) runs this script to set up every job, so a
# change here changes CI too. Every step is safe to rerun: each installer is
# skipped when its tool is already present, and uv sync is a no-op when .venv
# already matches uv.lock.
#
# borscht needs only two tools installed by hand, rustup and uv. Everything else
# is pinned in a file that one of those two reads:
#   - python 3.15        .python-version, which uv downloads a managed build of
#   - maturin            [build-system] in pyproject.toml, which uv installs into
#                        an isolated build environment whenever it builds borscht
#   - numpy, torch, ...  uv.lock
#   - cargo, rustc       rust-toolchain.toml, which pins the nightly-featured 1.99.1
#                        and the rust-src, rustc-dev, and llvm-tools components
#                        cuda-oxide's codegen backend needs
#   - cargo-oxide        installed here with cargo, from the same cuda-rust repo
#                        Cargo.toml pulls cuda-device and cuda-host from
#
# The CUDA kernels (cuda-oxide, https://nvidia.github.io/cuda-rust/cuda-oxide/
# getting-started/installation.html) also need system packages, which this
# script apt-installs with sudo when they are missing (debian/ubuntu only):
#   - CUDA Toolkit 13.x  cuda.h and curand.h for cuda-bindings, nvcc and nvJitLink
#                        for PTX. From NVIDIA's apt repo, at the CUDA version the
#                        installed driver reports, since PTX from a newer toolkit
#                        fails to load on an older driver (CUDA error 222). With
#                        no driver at all, 13.0, which compiles kernels but can't
#                        run them.
#   - libclang           bindgen, for the host-side cuda-bindings crate
# The NVIDIA driver itself (580+, for CUDA 13) is not installed here.
# LLVM's llc with the NVPTX backend comes from the llvm-tools component above,
# so a system llvm-21 is optional (cargo oxide prefers llc-23/22/21 when found).
set -euo pipefail

UV_SYNC_ARGS=(--locked)
for arg in "$@"; do
  case "$arg" in
    --no-dev) UV_SYNC_ARGS+=(--no-dev) ;;
    *) echo "setup: unknown argument $arg; usage: setup.sh [--no-dev]" >&2; exit 2 ;;
  esac
done

BORSCHT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$BORSCHT" rev-parse --show-toplevel)"
PATH_CHANGED=false

# rustc links the extension module with the system C toolchain.
if ! command -v cc >/dev/null 2>&1; then
  echo "setup: no C linker (cc) found; install one and rerun:" >&2
  echo "setup:     sudo apt-get install build-essential   # debian/ubuntu" >&2
  echo "setup:     xcode-select --install                 # macos" >&2
  exit 1
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO=sudo; fi

# Installs debian/ubuntu packages, or fails naming them for other systems.
apt_install() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "setup: no apt-get; install the equivalent of these packages and rerun: $*" >&2
    exit 1
  fi
  $SUDO apt-get update -q
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$@"
}

# CUDA Toolkit, located the way cuda-core, cuda-bindings, and cargo-oxide do:
# CUDA_TOOLKIT_PATH, then CUDA_HOME, then the first default install whose cuda.h
# is 13.0 or newer. Sets CUDA_ROOT, empty when none qualifies.
find_cuda() {
  CUDA_ROOT=""
  local dir version
  for dir in ${CUDA_TOOLKIT_PATH:-} ${CUDA_HOME:-} \
    /usr/local/cuda-13.3 /usr/local/cuda-13.2 /usr/local/cuda-13 /usr/local/cuda; do
    [ -f "$dir/include/cuda.h" ] || continue
    version="$(awk '/^#define CUDA_VERSION /{print $3; exit}' "$dir/include/cuda.h")"
    if [ "${version:-0}" -ge 13000 ]; then
      CUDA_ROOT="$dir"
      return
    fi
  done
}
find_cuda
if [ -z "$CUDA_ROOT" ]; then
  # The driver's CUDA version is the newest toolkit whose PTX it can load.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=13.0
    echo "setup: no NVIDIA driver (nvidia-smi) found, so CUDA $CUDA_VER it is:" >&2
    echo "setup: enough to compile the kernels, but running them needs a GPU and driver 580+" >&2
  else
    CUDA_VER="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')"
  fi
  if [ "${CUDA_VER%%.*}" -lt 13 ]; then
    echo "setup: the NVIDIA driver supports up to CUDA $CUDA_VER; cuda-oxide needs 13.0+," >&2
    echo "setup: so upgrade the driver to 580+ and rerun" >&2
    exit 1
  fi
  . /etc/os-release
  case "$(uname -m)" in
    x86_64) CUDA_ARCH=x86_64 ;;
    aarch64) CUDA_ARCH=sbsa ;;
    *) echo "setup: no NVIDIA CUDA apt repo for $(uname -m)" >&2; exit 1 ;;
  esac
  CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/${ID}${VERSION_ID//./}/$CUDA_ARCH"
  echo "setup: installing CUDA Toolkit $CUDA_VER from $CUDA_REPO"
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    keyring="$(mktemp -d)"
    curl -fsSL -o "$keyring/cuda-keyring.deb" "$CUDA_REPO/cuda-keyring_1.1-1_all.deb"
    $SUDO dpkg -i "$keyring/cuda-keyring.deb"
    rm -rf "$keyring"
  fi
  # Just the pieces cuda-oxide uses, not the multi-GB cuda-toolkit metapackage.
  v="${CUDA_VER//./-}"
  apt_install "cuda-nvcc-$v" "cuda-cudart-dev-$v" "cuda-driver-dev-$v" \
    "libcurand-dev-$v" "libnvjitlink-dev-$v"
  # The packages point /usr/local/cuda-13 at the new toolkit through
  # update-alternatives, which is where the cuda-rust build scripts look.
  find_cuda
  if [ -z "$CUDA_ROOT" ]; then
    echo "setup: installed CUDA $CUDA_VER, but no cuda.h 13.0+ turned up in /usr/local/cuda-13;" >&2
    echo "setup: export CUDA_TOOLKIT_PATH=/usr/local/cuda-$CUDA_VER and rerun" >&2
    exit 1
  fi
fi
echo "setup: using CUDA Toolkit at $CUDA_ROOT"

# clang and libclang, for bindgen to generate cuda-bindings from cuda.h: clang-sys
# loads libclang (LIBCLANG_PATH, then the linker cache and llvm dirs), and asks
# clang for the resource dir holding builtin headers like stddef.h.
has_libclang() {
  { [ -n "${LIBCLANG_PATH:-}" ] && compgen -G "$LIBCLANG_PATH/libclang*.so*" >/dev/null; } ||
    ldconfig -p 2>/dev/null | grep -q 'libclang[-.0-9]*\.so' ||
    compgen -G '/usr/lib/llvm-*/lib/libclang*.so*' >/dev/null
}
if ! has_libclang || ! command -v clang >/dev/null 2>&1; then
  # cuda-oxide's docs suggest clang 21; where the distro lacks it (ubuntu 22.04
  # ships 14), the default clang parses cuda.h fine.
  if apt-cache policy clang-21 2>/dev/null | grep -q "Candidate: [0-9]"; then
    apt_install clang-21 libclang-21-dev libclang-common-21-dev
    $SUDO update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100
  else
    apt_install clang libclang-dev
  fi
fi

# Rust, for maturin to compile rust/ into the borscht._borscht extension module.
CARGO_BIN="${CARGO_HOME:-$HOME/.cargo}/bin"
if ! command -v rustup >/dev/null 2>&1; then
  if [ ! -x "$CARGO_BIN/rustup" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  fi
  export PATH="$CARGO_BIN:$PATH"
  PATH_CHANGED=true
fi

# Bare `rustup toolchain install` installs the active toolchain, which inside
# borscht/ means the channel and components rust-toolchain.toml pins. The
# explicit component add covers a toolchain an older rustup installed without
# them: cuda-oxide needs rust-src, rustc-dev, and llvm-tools, and the pre-commit
# hook runs cargo fmt and cargo clippy.
(cd "$BORSCHT" && rustup toolchain install --no-self-update &&
  rustup component add rust-src rustc-dev llvm-tools rustfmt clippy)

# cargo-oxide, the `cargo oxide` subcommand that compiles the #[kernel]s in rust/
# to PTX. Installed from inside borscht/ so it builds with the pinned toolchain.
# Its first run fetches, builds, and caches the rustc codegen backend.
if ! command -v cargo-oxide >/dev/null 2>&1 && [ ! -x "$CARGO_BIN/cargo-oxide" ]; then
  (cd "$BORSCHT" && cargo install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide)
fi
# Builds and caches the codegen backend now rather than on the first build;
# a no-op when the cache already matches the cuda-rust commit Cargo.lock pins.
(cd "$BORSCHT" && cargo oxide setup)

# uv, for python, the virtual environment, and building borscht through maturin.
UV_BIN="${XDG_BIN_HOME:-$HOME/.local/bin}"
if ! command -v uv >/dev/null 2>&1; then
  if [ ! -x "$UV_BIN/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$UV_BIN:$PATH"
  PATH_CHANGED=true
fi

# Creates borscht/.venv with python 3.15 and the dev group (unless --no-dev), then
# builds and installs borscht itself as an editable install. --locked fails rather
# than re-resolving when uv.lock is out of date with pyproject.toml. The dev
# group's torch pulls several GB of nvidia-* CUDA wheels on linux x86_64.
(cd "$BORSCHT" && uv sync "${UV_SYNC_ARGS[@]}")

# pre-commit typechecks, formatchecks, and lints borscht; pre-push runs its tests.
git -C "$ROOT" config core.hooksPath .githooks

# Under GitHub Actions, hand cargo and uv to the job's later steps, which start
# new shells that don't see the PATH exported above.
if $PATH_CHANGED && [ -n "${GITHUB_PATH:-}" ]; then
  echo "$CARGO_BIN" >>"$GITHUB_PATH"
  echo "$UV_BIN" >>"$GITHUB_PATH"
fi

echo
echo "setup: done. Next:"
if $PATH_CHANGED; then
  echo "    open a new shell to put cargo and uv on PATH"
fi
echo "    cd borscht && uv run pytest     # python tests against the built extension"
echo "    cd borscht && cargo oxide test  # rust tests, with their CUDA kernels compiled to PTX"
echo "    cd borscht && cargo oxide build # rust/ with its CUDA kernels compiled to PTX"
echo "    cd borscht && cargo oxide doctor  # check the CUDA toolchain, driver, and GPU"
