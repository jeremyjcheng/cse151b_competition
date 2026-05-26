#!/usr/bin/env bash
# Stage 2 training wrapper — defaults to v3 recipe (override OUT_DIR / STAGE1_ADAPTER).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

STAGE1_ADAPTER="${STAGE1_ADAPTER:-workspaces/stage1_reasoning_v2/final_adapter}"
OUT_DIR="${OUT_DIR:-workspaces/stage2_adapt_v3}"
ADAPT_STEPS="${ADAPT_STEPS:-200}"
GPU_ID="${GPU_ID:-0}"
LOG="logs/stage2_train_$(date +%Y%m%d_%H%M).log"

mkdir -p logs

if [[ ! -x "$PY" ]]; then
  echo "ERROR: PY is not executable: $PY" >&2
  exit 1
fi

HELP="$("$PY" scripts/modular_pipeline/train_lora.py --help 2>&1)" || true

EXTRA_ARGS=()
if echo "$HELP" | grep -q 'load-in-4bit'; then
  EXTRA_ARGS+=(--no-load-in-4bit)
fi
if echo "$HELP" | grep -q 'gradient-checkpointing'; then
  EXTRA_ARGS+=(--gradient-checkpointing)
fi
if echo "$HELP" | grep -q 'stage2-mcq-with-reasoning'; then
  EXTRA_ARGS+=(--stage2-mcq-with-reasoning)
fi
if echo "$HELP" | grep -q 'val-eval-every-steps'; then
  EXTRA_ARGS+=(--val-eval-every-steps 50)
fi

echo "Stage 1 adapter: $STAGE1_ADAPTER"
echo "Output: $OUT_DIR"
echo "Steps:  $ADAPT_STEPS"
echo "Log: $LOG"

nohup "$PY" scripts/modular_pipeline/train_lora.py \
  --stage adapt \
  --input public \
  --output-dir "$OUT_DIR" \
  --resume-from-adapter "$STAGE1_ADAPTER" \
  --max-steps "$ADAPT_STEPS" \
  --learning-rate 1e-5 \
  --stage2-holdout-fraction 0.3 \
  --limit-mcq 0 \
  --limit-free 0 \
  --stage2-final-answer-only \
  "${EXTRA_ARGS[@]}" \
  --max-seq-length 1024 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --save-every-steps 50 \
  --gpu-id "$GPU_ID" \
  >"$LOG" 2>&1 &

echo $! > logs/stage2_train.pid
echo "Started PID=$(cat logs/stage2_train.pid)"
echo "Monitor: tail -f $LOG"
