#!/usr/bin/env bash
# Holdout checkpoint sweep using conda vllm python (safe for nohup).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

STAGE2_ROOT="${STAGE2_ROOT:-workspaces/stage2_adapt_v2}"
HOLDOUT="${STAGE2_ROOT}/stage2_holdout.jsonl"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$HOLDOUT" ]]; then
  echo "Missing holdout: $HOLDOUT"
  exit 1
fi

LOG_FILE="${LOG_DIR}/stage2_eval_sweep.log"
echo "Python: $PY"
echo "Log:    $LOG_FILE"

nohup "$PY" scripts/modular_pipeline/eval_runner.py \
  --input "$HOLDOUT" \
  --checkpoint-dir "$STAGE2_ROOT" \
  --split-name val \
  --gpu-id "$GPU_ID" \
  --vllm-quantization none \
  --vllm-load-format auto \
  --eval-report "${STAGE2_ROOT}/holdout_checkpoint_sweep.json" \
  >"$LOG_FILE" 2>&1 &

echo "Started checkpoint sweep (PID $!). Monitor: tail -f $LOG_FILE"
