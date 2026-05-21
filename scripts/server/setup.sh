#!/usr/bin/env bash
# Phase 0: install into conda env vllm (no new .venv).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "Project root: $ROOT"
mkdir -p workspaces logs results data

bash "${ROOT}/scripts/server/remove_venv.sh"
bash "${ROOT}/scripts/server/install_into_vllm.sh"
