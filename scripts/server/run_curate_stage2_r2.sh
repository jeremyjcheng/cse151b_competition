#!/usr/bin/env bash
# Phase 3c: curated hard-example Stage 2 round 2 (~100 steps).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

STAGE1_ADAPTER="${STAGE1_ADAPTER:-workspaces/stage1_reasoning_v2/final_adapter}"
STAGE2_ROOT="${STAGE2_ROOT:-workspaces/stage2_adapt_v3}"
OUT_DIR="${OUT_DIR:-workspaces/stage2_adapt_v3_r2}"
CURATED="${CURATED:-data/hard_examples_r1.jsonl}"
GPU_ID="${GPU_ID:-0}"
LOG="logs/stage2_r2_$(date +%Y%m%d_%H%M).log"

if [[ ! -f "$CURATED" ]]; then
  echo "Missing curated file: $CURATED"
  echo "  Run: bash scripts/server/run_holdout_infer.sh"
  echo "  Then: $PY scripts/modular_pipeline/curate_data.py \\"
  echo "    --predictions results/holdout_outputs.jsonl \\"
  echo "    --output $CURATED"
  exit 1
fi

mkdir -p logs data

HELP="$("$PY" scripts/modular_pipeline/train_lora.py --help 2>&1)" || true
EXTRA=()
if echo "$HELP" | grep -q 'load-in-4bit'; then EXTRA+=(--no-load-in-4bit); fi
if echo "$HELP" | grep -q 'gradient-checkpointing'; then EXTRA+=(--gradient-checkpointing); fi
if echo "$HELP" | grep -q 'stage2-mcq-with-reasoning'; then EXTRA+=(--stage2-mcq-with-reasoning); fi

nohup "$PY" scripts/modular_pipeline/train_lora.py \
  --stage adapt \
  --input public \
  --output-dir "$OUT_DIR" \
  --resume-from-adapter "$STAGE1_ADAPTER" \
  --curated-input "$CURATED" \
  --max-steps 100 \
  --learning-rate 1e-5 \
  --stage2-holdout-fraction 0.3 \
  --limit-mcq 0 \
  --limit-free 0 \
  --stage2-final-answer-only \
  --max-seq-length 1024 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --gpu-id "$GPU_ID" \
  "${EXTRA[@]}" \
  >"$LOG" 2>&1 &

echo $! > logs/stage2_r2.pid
echo "Started Stage 2 round 2 PID=$(cat logs/stage2_r2.pid)"
echo "Monitor: tail -f $LOG"
