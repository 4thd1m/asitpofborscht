#!/usr/bin/env bash
# Installs everything needed to build and preview the SITP book locally:
#
#     ./sitp/setup.sh             # from anywhere in the repo
#     cd sitp && mdbook serve     # make your edits to markdown
#                                 # preview with http://localhost:3000/
#                                 # and cut a PR
#
# This mirrors the Netlify build in netlify.toml, which is the source of truth
# for what production uses, so keep the two in sync when you change either.
# Every step is safe to rerun: cargo install skips a crate already installed at
# the same version and source.
#
# asitpofborscht vendors aquascope, an mdbook preprocessor from the Cognitive
# Engineering Lab (see https://willcrichton.net/#cgk:ownership-conceptual-model)
# which conceptually visualizes Rust ownership. Specifically, it vendors
# aquascope as a git submodule in order to pin SITP's mdbook rust toolchain to
# aquascope's via the symlink
# sitp/rust-toolchain.toml -> sitp/vendor/aquascope/rust-toolchain.toml
set -euo pipefail

SITP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SITP" rev-parse --show-toplevel)"
cd "$ROOT"

if ! command -v rustup >/dev/null 2>&1; then
  echo "setup: rustup not found; install it from https://rustup.rs and rerun" >&2
  exit 1
fi

# A no-op if you cloned with --recurse-submodules.
git submodule update --init sitp/vendor/aquascope

TOOLCHAIN=$(grep -m1 '^channel' sitp/rust-toolchain.toml 2>/dev/null | cut -d\" -f2) || TOOLCHAIN=""
if [ -z "$TOOLCHAIN" ]; then
  echo "setup: could not read channel from sitp/rust-toolchain.toml" >&2
  echo "setup: is the sitp/vendor/aquascope submodule initialized?" >&2
  exit 1
fi
echo "setup: sitp toolchain is $TOOLCHAIN"

# Bare `rustup toolchain install` installs the active toolchain, which inside
# sitp/ means the channel and components from rust-toolchain.toml.
(cd sitp && rustup toolchain install)

# The install above ran under sitp/'s override, which does not set a rustup
# default, so every cargo call below names its toolchain explicitly.
rustup toolchain install --no-self-update stable
cargo +stable install mdbook --version 0.5.2 --locked
cargo +stable install mdbook-katex --git https://github.com/lzanini/mdbook-katex
cargo +stable install mdbook-aquascope --locked
cargo +stable install mdbook-quiz --locked
# Only aquascope_front needs the rustc-dev nightly.
cargo "+$TOOLCHAIN" install aquascope_front --git https://github.com/cognitive-engineering-lab/aquascope --tag v0.4.0

# Lean, for the {{#lean}} examples in src/ap.md. Without `lake` on PATH the lean
# preprocessor warns and no-ops, so the appendix renders with its directives
# unsubstituted rather than failing the build.
if ! command -v elan >/dev/null 2>&1 && [ ! -x "$HOME/.elan/bin/elan" ]; then
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
fi

# Retheme matplotlib figures on commit.
git config core.hooksPath .githooks

echo
echo "setup: done. Next:"
if ! command -v lake >/dev/null 2>&1; then
  echo '    export PATH="$HOME/.elan/bin:$PATH"   # or open a new shell'
fi
echo "    cd sitp && mdbook serve               # preview with http://localhost:3000/"
