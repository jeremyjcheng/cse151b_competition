#!/usr/bin/env bash
# Remove broken project .venv — use conda env vllm instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  echo "Removing $ROOT/.venv ..."
  rm -rf .venv
  echo "Done. Use: conda activate vllm"
else
  echo "No .venv directory found."
fi
