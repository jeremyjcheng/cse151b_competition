#!/usr/bin/env bash
# Safe one-liner: opens a NEW bash subshell with vllm env (never kills your zsh).
# Usage:  bash scripts/server/enter_vllm.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh" || exit 1

echo "Entered vllm env in bash subshell. Type 'exit' to leave."
echo "  diagnose: bash scripts/server/diagnose_vllm.sh"
echo "  sweep:    STAGE2_ROOT=workspaces/stage2_adapt_v2 bash scripts/server/run_stage2_eval_sweep.sh"
exec bash --noprofile --norc -i
