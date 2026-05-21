#!/usr/bin/env bash
# Install into existing conda env "vllm" and remove broken .venv.
set -euo pipefail

cd "$(dirname "$0")"

bash scripts/server/remove_venv.sh
bash scripts/server/install_into_vllm.sh

echo ""
echo "Next:"
echo "  conda activate vllm"
echo "  bash scripts/server/test_installs.sh"
echo "  bash scripts/server/run_full_pipeline.sh"
