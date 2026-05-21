#!/usr/bin/env bash
# Phase 1 monitoring: checkpoints, holdout, latest log tail
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=== Latest log (last 30 lines) ==="
LATEST="$(ls -t logs/full_pipeline_*.log 2>/dev/null | head -1 || true)"
if [[ -n "${LATEST:-}" ]]; then
  tail -30 "$LATEST"
else
  echo "No full_pipeline_*.log yet."
fi

echo ""
echo "=== Stage 1 checkpoints ==="
ls -d workspaces/stage1_reasoning/checkpoint-step-* 2>/dev/null || echo "(none yet)"
if [[ -d workspaces/stage1_reasoning/final_adapter ]]; then
  echo "final_adapter: workspaces/stage1_reasoning/final_adapter"
fi

echo ""
echo "=== Stage 2 artifacts ==="
ls -la workspaces/stage2_adapt/ 2>/dev/null || echo "(stage2_adapt not created yet)"
if [[ -f workspaces/stage2_adapt/stage2_holdout.jsonl ]]; then
  echo "holdout lines: $(wc -l < workspaces/stage2_adapt/stage2_holdout.jsonl)"
fi

if [[ -f logs/full_pipeline.pid ]]; then
  PID="$(cat logs/full_pipeline.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    echo ""
    echo "Pipeline still running (PID $PID)"
  else
    echo ""
    echo "Pipeline PID $PID is not running (finished or crashed)."
  fi
fi
