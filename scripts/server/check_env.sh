#!/usr/bin/env bash
# Verify conda vllm env before training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ -d "${ROOT}/.venv" ]]; then
  echo "ERROR: .venv still exists — scripts will refuse to run."
  echo "  bash scripts/server/remove_venv.sh"
  exit 1
fi

bash "${SCRIPT_DIR}/test_installs.sh"
