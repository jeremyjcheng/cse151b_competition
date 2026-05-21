#!/usr/bin/env bash
# Re-run Stage 2 + eval using an existing Stage 1 adapter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

GPU_ID="${GPU_ID:-0}"
STAGE1_ADAPTER="${STAGE1_ADAPTER:-workspaces/stage1_reasoning/final_adapter}"
LOG="logs/stage2_only_$(date +%Y%m%d_%H%M).log"

if [[ ! -d "$STAGE1_ADAPTER" ]]; then
  echo "Stage 1 adapter not found: $STAGE1_ADAPTER"
  exit 1
fi

mkdir -p logs workspaces

nohup "$PY" scripts/modular_pipeline/run_lora_workspaces.py \
  --adapter-root workspaces \
  --skip-stage1 \
  --stage1-adapter-path "$STAGE1_ADAPTER" \
  --adapt-steps 60 \
  --stage2-learning-rate 1e-5 \
  --stage2-holdout-fraction 0.3 \
  --limit-mcq 50 \
  --limit-free 25 \
  --stage2-final-answer-only \
  --no-stage2-train-on-full-chat \
  --stage2-freeze-reasoning-style \
  --eval-after-stage2 \
  --skip-infer \
  --gpu-id "$GPU_ID" \
  --vllm-quantization none \
  --vllm-load-format auto \
  >"$LOG" 2>&1 &

echo $! > logs/stage2_only.pid
echo "Started Stage 2-only pipeline PID=$(cat logs/stage2_only.pid)"
echo "Monitor: tail -f $LOG"
