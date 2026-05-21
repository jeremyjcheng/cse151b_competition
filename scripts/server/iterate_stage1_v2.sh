#!/usr/bin/env bash
# Phase 2/3: stronger Stage 1 (1500 steps) then Stage 2 only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

GPU_ID="${GPU_ID:-0}"
STAGE1_OUT="${STAGE1_OUT:-workspaces/stage1_reasoning_v2}"
LOG="logs/stage1_v2_$(date +%Y%m%d_%H%M).log"

mkdir -p logs workspaces

echo "=== Stage 1 v2 (1500 steps) -> $STAGE1_OUT ==="
nohup "$PY" scripts/modular_pipeline/train_lora.py \
  --stage reasoning \
  --output-dir "$STAGE1_OUT" \
  --include-openmath \
  --include-hendrycks \
  --max-steps 1500 \
  --learning-rate 8e-5 \
  --train-on-full-chat \
  --gpu-id "$GPU_ID" \
  >"$LOG" 2>&1 &

echo $! > logs/stage1_v2.pid
echo "Stage 1 v2 PID=$(cat logs/stage1_v2.pid). Wait for completion, then run:"
echo "  STAGE1_ADAPTER=$STAGE1_OUT/final_adapter bash scripts/server/run_stage2_only.sh"
