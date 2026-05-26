#!/usr/bin/env bash
# Master script: 30% -> 60% holdout plan (run steps manually between GPU jobs).
set -euo pipefail

cat <<'EOF'
CSE151B accuracy plan (Qwen3-4B) — run from project root inside:
  bash scripts/server/enter_vllm.sh

Phase 0 — baseline ladder (no training):
  bash scripts/server/run_baseline_ladder.sh

Phase 1 — Stage 1 v2 (1500 steps, ~12-24h):
  bash scripts/server/iterate_stage1_v2.sh
  tail -f logs/stage1_v2_*.log

Phase 2 — Stage 2 v3 (full train slice, 200 steps):
  STAGE1_ADAPTER=workspaces/stage1_reasoning_v2/final_adapter \
    bash scripts/server/run_stage2_v3.sh
  STAGE2_ROOT=workspaces/stage2_adapt_v3 bash scripts/server/run_stage2_eval_and_pick.sh

Phase 3 — extraction already in code; optional curation round 2:
  STAGE2_ROOT=workspaces/stage2_adapt_v3 bash scripts/server/run_holdout_infer.sh
  $PY scripts/modular_pipeline/curate_data.py \
    --predictions results/holdout_outputs.jsonl \
    --output data/hard_examples_r1.jsonl
  bash scripts/server/run_curate_stage2_r2.sh

Phase 4 — MCQ self-consistency (if holdout 55-58%):
  Set MCQ_SELF_CONSISTENCY_SAMPLES=3 in scripts/modular_pipeline/setting.py
  bash scripts/server/run_mcq_self_consistency_eval.sh

Phase 5 — private submit:
  STAGE2_ROOT=workspaces/stage2_adapt_v3 bash scripts/server/private_submit.sh
  scp results/private_submission.csv to your laptop and upload to leaderboard

Milestones: M1 >=38% (Stage1 only) | M2 >=48% (Stage2 v3) | M3 >=55% | M4 >=60%
EOF
