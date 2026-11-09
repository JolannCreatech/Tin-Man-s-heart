#!/usr/bin/env bash
# Link the already-downloaded Magenta RealTime 2 assets into this repository.
#
# We deliberately use symlinks instead of copying: the model weights are large and
# the mission forbids duplicating them or modifying the external Magenta checkout.
# The project only ever *reads* these files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAGENTA_HOME="${MAGENTA_HOME:-$HOME/Documents/Magenta/magenta-rt-v2}"
ASSETS="$REPO_ROOT/assets/magenta-rt-v2"

if [[ ! -d "$MAGENTA_HOME" ]]; then
  echo "error: Magenta assets not found at $MAGENTA_HOME" >&2
  echo "Set MAGENTA_HOME to the directory containing resources/ and models/." >&2
  exit 1
fi

mkdir -p "$ASSETS"
mkdir -p "$REPO_ROOT/outputs" "$REPO_ROOT/.cache"

link() {
  local src="$1" dst="$2"
  if [[ ! -e "$src" ]]; then
    echo "warning: missing $src (skipping)" >&2
    return 0
  fi
  if [[ -L "$dst" ]]; then
    current="$(readlink "$dst")"
    if [[ "$current" == "$src" ]]; then
      echo "ok: $dst -> $src"
      return 0
    fi
    rm "$dst"
  elif [[ -e "$dst" ]]; then
    echo "warning: $dst exists and is not a symlink; leaving it alone" >&2
    return 0
  fi
  ln -s "$src" "$dst"
  echo "linked: $dst -> $src"
}

link "$MAGENTA_HOME/resources" "$ASSETS/resources"
link "$MAGENTA_HOME/models" "$ASSETS/models"

# Only the small model is complete locally. Warn rather than pretend otherwise.
if [[ ! -f "$ASSETS/models/mrt2_small/mrt2_small.mlxfn" ]]; then
  echo "error: mrt2_small.mlxfn not found through the linked models directory." >&2
  exit 1
fi

echo
echo "Assets ready under $ASSETS"
echo "The project sets MAGENTA_HOME to this directory at runtime; outputs stay in"
echo "$REPO_ROOT/outputs."
