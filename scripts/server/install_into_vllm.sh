#!/usr/bin/env bash
# Install missing competition deps into existing conda env (default: vllm).
# Does NOT create .venv. Run test_installs.sh after.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Installing into conda env: ${CONDA_ENV_NAME:-vllm}"
echo "Python: $PY"

pip install -U pip wheel setuptools

_req_filtered() {
  grep -v '^#' "$ROOT/requirements.txt" | grep -v '^[[:space:]]*$' \
    | grep -vi 'xformers' | grep -vi '^vllm'
}

echo "=== Ensuring PyTorch ==="
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

echo "=== Project deps (no xformers / no vllm pin) ==="
_req_filtered | pip install -r /dev/stdin

echo "=== peft / datasets (training) ==="
pip install peft datasets accelerate bitsandbytes

echo "=== vLLM (skip if already satisfied) ==="
pip install 'vllm>=0.10.0' 2>/dev/null || echo "vllm install skipped or already OK"

echo ""
bash "${SCRIPT_DIR}/test_installs.sh"
