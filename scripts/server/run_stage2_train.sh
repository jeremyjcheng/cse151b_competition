#!/usr/bin/env bash
# Stage 2 training in background — avoids SSH terminal dying on OOM/GPU reset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

STAGE1_ADAPTER="${STAGE1_ADAPTER:-$(bash "${SCRIPT_DIR}/resolve_stage1_adapter.sh")}"
OUT_DIR="${OUT_DIR:-workspaces/stage2_adapt_v2}"
LOG="logs/stage2_train_$(date +%Y%m%d_%H%M).log"

mkdir -p logs

echo "Stage 1 adapter: $STAGE1_ADAPTER"
echo "Output: $OUT_DIR"
echo "Log: $LOG"
echo ""
echo "Tip: run inside tmux —  tmux new -s train"
echo ""

nohup "$PY" scripts/modular_pipeline/train_lora.py \
  --stage adapt \
  --input public \
  --output-dir "$OUT_DIR" \
  --resume-from-adapter "$STAGE1_ADAPTER" \
  --stage2-mcq-with-reasoning \
  --stage2-final-answer-only \
  --no-load-in-4bit \
  --gradient-checkpointing \
  --max-seq-length 1024 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --gpu-id "${GPU_ID:-0}" \
  >"$LOG" 2>&1 &

echo $! > logs/stage2_train.pid
echo "Started PID=$(cat logs/stage2_train.pid)"
echo "Monitor: tail -f $LOG"
echo "GPU:     watch -n2 nvidia-smi"
