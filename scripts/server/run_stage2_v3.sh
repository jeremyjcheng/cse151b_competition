#!/usr/bin/env bash
# Phase 2: Stage 2 v3 — full public train slice (no 50/25 cap), 200 steps, periodic holdout eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

STAGE1_ADAPTER="${STAGE1_ADAPTER:-workspaces/stage1_reasoning_v2/final_adapter}"
OUT_DIR="${OUT_DIR:-workspaces/stage2_adapt_v3}"
GPU_ID="${GPU_ID:-0}"
LOG="logs/stage2_v3_$(date +%Y%m%d_%H%M).log"

if [[ ! -d "$STAGE1_ADAPTER" ]]; then
  echo "ERROR: Stage 1 adapter not found: $STAGE1_ADAPTER"
  echo "  Run: bash scripts/server/iterate_stage1_v2.sh first"
  exit 1
fi

mkdir -p logs

HELP="$("$PY" scripts/modular_pipeline/train_lora.py --help 2>&1)" || true
EXTRA=()
if echo "$HELP" | grep -q 'load-in-4bit'; then
  EXTRA+=(--no-load-in-4bit)
fi
if echo "$HELP" | grep -q 'gradient-checkpointing'; then
  EXTRA+=(--gradient-checkpointing)
fi
if echo "$HELP" | grep -q 'stage2-mcq-with-reasoning'; then
  EXTRA+=(--stage2-mcq-with-reasoning)
fi
if echo "$HELP" | grep -q 'val-eval-every-steps'; then
  EXTRA+=(--val-eval-every-steps 50)
fi

echo "Stage 1: $STAGE1_ADAPTER"
echo "Output:  $OUT_DIR"
echo "Log:     $LOG"

nohup "$PY" scripts/modular_pipeline/train_lora.py \
  --stage adapt \
  --input public \
  --output-dir "$OUT_DIR" \
  --resume-from-adapter "$STAGE1_ADAPTER" \
  --max-steps 200 \
  --learning-rate 1e-5 \
  --stage2-holdout-fraction 0.3 \
  --stage2-holdout-seed 0 \
  --limit-mcq 0 \
  --limit-free 0 \
  --stage2-final-answer-only \
  --no-stage2-train-on-full-chat \
  --max-seq-length 1024 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --save-every-steps 50 \
  --gpu-id "$GPU_ID" \
  "${EXTRA[@]}" \
  >"$LOG" 2>&1 &

echo $! > logs/stage2_v3.pid
echo "Started Stage 2 v3 PID=$(cat logs/stage2_v3.pid)"
echo "After training:"
echo "  STAGE2_ROOT=$OUT_DIR bash scripts/server/run_stage2_eval_sweep.sh"
echo "  bash scripts/server/pick_best_checkpoint.py --stage2-root $OUT_DIR"
