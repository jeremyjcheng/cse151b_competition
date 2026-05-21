#!/usr/bin/env bash
# Phase 1: long-running Stage 1 -> Stage 2 -> holdout eval (no private infer)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

GPU_ID="${GPU_ID:-0}"
REASONING_STEPS="${REASONING_STEPS:-1000}"
ADAPT_STEPS="${ADAPT_STEPS:-60}"
LOG="logs/full_pipeline_$(date +%Y%m%d_%H%M).log"

mkdir -p logs workspaces results

bash "${SCRIPT_DIR}/check_env.sh"

echo "Logging to $LOG"
echo "PID will be written to logs/full_pipeline.pid"

nohup "$PY" scripts/modular_pipeline/run_lora_workspaces.py \
  --adapter-root workspaces \
  --include-openmath \
  --include-hendrycks \
  --reasoning-steps "$REASONING_STEPS" \
  --reasoning-learning-rate 8e-5 \
  --adapt-steps "$ADAPT_STEPS" \
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

echo $! > logs/full_pipeline.pid
echo "Started pipeline PID=$(cat logs/full_pipeline.pid)"
echo "Monitor: tail -f $LOG"
