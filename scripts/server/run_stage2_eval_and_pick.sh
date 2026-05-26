#!/usr/bin/env bash
# Phase 2c: checkpoint sweep + write best_adapter.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

STAGE2_ROOT="${STAGE2_ROOT:-workspaces/stage2_adapt_v3}"

STAGE2_ROOT="$STAGE2_ROOT" bash "${SCRIPT_DIR}/run_stage2_eval_sweep.sh"
"$PY" scripts/server/pick_best_checkpoint.py --stage2-root "$STAGE2_ROOT"
