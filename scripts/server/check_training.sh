#!/usr/bin/env bash
# Quick status after SSH reconnect.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

echo "=== Stage 1 PID file ==="
if [[ -f logs/stage1_v2.pid ]]; then
  pid="$(cat logs/stage1_v2.pid)"
  echo "PID from file: $pid"
  ps -p "$pid" -o pid,etime,cmd 2>/dev/null || echo "  (not running)"
else
  echo "No logs/stage1_v2.pid"
fi

echo ""
echo "=== train_lora processes ==="
pgrep -af "train_lora.py.*reasoning" || echo "(none)"

echo ""
echo "=== Stage 1 outputs ==="
S1="${STAGE1_OUT:-workspaces/stage1_reasoning_v2}"
if [[ -d "$S1/final_adapter" ]]; then
  echo "DONE: $S1/final_adapter"
fi
ls -d "$S1"/checkpoint-step-* 2>/dev/null | tail -5 || echo "(no checkpoints yet)"

echo ""
echo "=== Latest Stage 1 log (last 15 lines) ==="
latest_log="$(ls -t logs/stage1_v2*.log 2>/dev/null | head -1)"
if [[ -n "$latest_log" ]]; then
  echo "File: $latest_log"
  tail -15 "$latest_log"
else
  echo "(no stage1 log)"
fi
