#!/usr/bin/env bash
# Resume Stage 1 from latest checkpoint-step-* (after disconnect/OOM).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

STAGE1_OUT="${STAGE1_OUT:-workspaces/stage1_reasoning_v2}"
GPU_ID="${GPU_ID:-0}"
MAX_STEPS="${MAX_STEPS:-1500}"
LOG="logs/stage1_v2_resume_$(date +%Y%m%d_%H%M).log"

if [[ -d "$STAGE1_OUT/final_adapter" ]]; then
  echo "Stage 1 already finished: $STAGE1_OUT/final_adapter"
  echo "  STAGE1_ADAPTER=$STAGE1_OUT/final_adapter bash scripts/server/run_stage2_v3.sh"
  exit 0
fi

latest=""
best=-1
for d in "$STAGE1_OUT"/checkpoint-step-*; do
  [[ -d "$d" ]] || continue
  step="${d##*checkpoint-step-}"
  if [[ "$step" =~ ^[0-9]+$ ]] && (( step > best )); then
    best=$step
    latest="$d"
  fi
done

if [[ -z "$latest" ]]; then
  echo "No checkpoint found under $STAGE1_OUT — starting fresh."
  exec bash "${SCRIPT_DIR}/iterate_stage1_v2.sh"
fi

remaining=$((MAX_STEPS - best))
if (( remaining <= 0 )); then
  echo "Latest checkpoint is step $best (>= $MAX_STEPS). Copying to final_adapter..."
  mkdir -p "$STAGE1_OUT/final_adapter"
  cp -a "$latest/"* "$STAGE1_OUT/final_adapter/"
  echo "Done: $STAGE1_OUT/final_adapter"
  exit 0
fi

echo "Resuming from: $latest (step $best)"
echo "Training $remaining more steps -> $MAX_STEPS total"
echo "Log: $LOG"

HELP="$("$PY" scripts/modular_pipeline/train_lora.py --help 2>&1)" || true
EXTRA=()
if echo "$HELP" | grep -q 'load-in-4bit'; then EXTRA+=(--no-load-in-4bit); fi
if echo "$HELP" | grep -q 'gradient-checkpointing'; then EXTRA+=(--gradient-checkpointing); fi

nohup "$PY" scripts/modular_pipeline/train_lora.py \
  --stage reasoning \
  --output-dir "$STAGE1_OUT" \
  --resume-from-adapter "$latest" \
  --include-openmath \
  --include-hendrycks \
  --max-steps "$remaining" \
  --learning-rate 8e-5 \
  --train-on-full-chat \
  --save-every-steps 100 \
  --gpu-id "$GPU_ID" \
  "${EXTRA[@]}" \
  >"$LOG" 2>&1 &

echo $! > logs/stage1_v2.pid
echo "Resumed PID=$(cat logs/stage1_v2.pid). Monitor: tail -f $LOG"
