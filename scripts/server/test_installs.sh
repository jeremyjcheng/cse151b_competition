#!/usr/bin/env bash
# Test packages in conda env vllm (or CONDA_ENV_NAME).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

"$PY" "${SCRIPT_DIR}/test_installs.py"
