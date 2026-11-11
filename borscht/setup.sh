#!/usr/bin/env bash
# Installs everything needed to build, test, and hack on borscht:
#
#     ./borscht/setup.sh            # from anywhere in the repo
#     cd borscht && uv run pytest   # make your edits to python/ or rust/
#                                   # uv rebuilds the extension when rust/ changes
#                                   # and cut a PR
#
# This mirrors the CI build in .github/workflows/borscht.yml, so keep the two in
# sync when you change either. Every step is safe to rerun: each installer is
# skipped when its tool is already present, and uv sync is a no-op when .venv
# already matches uv.lock.
#
# borscht needs only two tools installed by hand, rustup and uv. Everything else
# is pinned in a file that one of those two reads:
#   - python 3.15        .python-version, which uv downloads a managed build of
#   - maturin            [build-system] in pyproject.toml, which uv installs into
#                        an isolated build environment whenever it builds borscht
#   - numpy, torch, ...  uv.lock
#   - cargo, rustc       the active rustup toolchain: the rustup default (stable),
#                        unless a borscht/rust-toolchain.toml pins another
set -euo pipefail

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
# borscht/ means the rustup default, or the channel a rust-toolchain.toml pins.
# The pre-commit hook runs cargo fmt and cargo clippy, which a toolchain
# installed with the minimal profile lacks.
(cd "$BORSCHT" && rustup toolchain install --no-self-update && rustup component add rustfmt clippy)

# uv, for python, the virtual environment, and building borscht through maturin.
UV_BIN="${XDG_BIN_HOME:-$HOME/.local/bin}"
if ! command -v uv >/dev/null 2>&1; then
  if [ ! -x "$UV_BIN/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$UV_BIN:$PATH"
  PATH_CHANGED=true
fi

# Creates borscht/.venv with python 3.15 and the dev group, then builds and
# installs borscht itself as an editable install. --locked fails rather than
# re-resolving when uv.lock is out of date with pyproject.toml, same as CI.
# Unlike CI, this installs the dev group's torch, whose linux x86_64 wheel pulls
# several GB of nvidia-* CUDA dependencies.
(cd "$BORSCHT" && uv sync --locked)

# pre-commit typechecks, formatchecks, and lints borscht; pre-push runs its tests.
git -C "$ROOT" config core.hooksPath .githooks

echo
echo "setup: done. Next:"
if $PATH_CHANGED; then
  echo "    open a new shell to put cargo and uv on PATH"
fi
echo "    cd borscht && uv run pytest     # python tests against the built extension"
echo "    cd borscht && cargo test        # rust tests"
