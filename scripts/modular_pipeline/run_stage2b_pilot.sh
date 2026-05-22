#!/usr/bin/env bash
set -euo pipefail

# Stage 2B pilot runner (A/B/C) for CSE151B competition repo.
# Run from repo root: bash scripts/modular_pipeline/run_stage2b_pilot.sh

PYTHON_BIN="${PYTHON_BIN:-python}"

COMMON_EVAL_ARGS=(
  --input public
  --limit-mcq 50
  --limit-free 50
  --sample-seed 0
  --vllm-enforce-eager
)

echo "=== A) Evaluate Stage 1 reasoning adapter ==="
"${PYTHON_BIN}" scripts/modular_pipeline/modular_pipeline.py \
  --output-dir results/eval_v3_stage1_reasoning_100 \
  --lora-adapter-path artifacts/lora_clean_v3/stage1_reasoning_new/final_adapter \
  "${COMMON_EVAL_ARGS[@]}"

echo "=== B) Evaluate current Stage 2 cleanup adapter ==="
"${PYTHON_BIN}" scripts/modular_pipeline/modular_pipeline.py \
  --output-dir results/eval_v3_stage2_mcq_cleanup_100 \
  --lora-adapter-path artifacts/lora_clean_v3/stage2_mcq_cleanup/final_adapter \
  "${COMMON_EVAL_ARGS[@]}"

echo "=== C1) Build synthetic MCQ reasoning traces from Stage 1 ==="
"${PYTHON_BIN}" scripts/modular_pipeline/build_synthetic_mcq_reasoning.py \
  --input public \
  --output-path artifacts/lora_clean_v3/synthetic_mcq_reasoning_pilot.jsonl \
  --rejected-output-path artifacts/lora_clean_v3/synthetic_mcq_reasoning_pilot_rejected.jsonl \
  --all-candidates-output-path artifacts/lora_clean_v3/synthetic_mcq_reasoning_pilot_all_candidates.jsonl \
  --train-replay-output-path artifacts/lora_clean_v3/synthetic_mcq_reasoning_pilot_train_replay.jsonl \
  --lora-adapter-path artifacts/lora_clean_v3/stage1_reasoning_new/final_adapter \
  --limit-mcq 50 \
  --sample-seed 0 \
  --num-samples-per-question 4 \
  --temperature 0.7 \
  --top-p 0.9 \
  --mcq-max-new-tokens 2048 \
  --min-raw-tokens 64 \
  --max-raw-tokens 900 \
  --blend-synthetic-ratio 0.8 \
  --blend-frq-ratio 0.2 \
  --frq-replay-path results/eval_v3_stage1_reasoning_100/public_mcq50_free50_seed0_outputs.jsonl \
  --vllm-enforce-eager

echo "=== C2) Train Stage 2B adapter ==="
"${PYTHON_BIN}" scripts/modular_pipeline/train_lora.py \
  --stage mixed_reasoning_mcq \
  --output-dir artifacts/lora_clean_v3/stage2b_mcq_reasoning_distill \
  --resume-from-adapter artifacts/lora_clean_v3/stage1_reasoning_new/final_adapter \
  --include-base-replay \
  --base-replay-path artifacts/lora_clean_v3/synthetic_mcq_reasoning_pilot_train_replay.jsonl \
  --mcq-target-mode full_trace \
  --include-metamathqa \
  --include-numinamath-cot \
  --max-base-replay-examples 400 \
  --max-metamathqa-examples 100 \
  --max-numinamath-cot-examples 100 \
  --target-module-set attention \
  --learning-rate 1e-5 \
  --max-steps 80 \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --max-seq-length 1536 \
  --save-every-steps 50 \
  --sample-seed 0 \
  --print-dataset-samples

echo "=== C3) Evaluate Stage 2B adapter ==="
"${PYTHON_BIN}" scripts/modular_pipeline/modular_pipeline.py \
  --output-dir results/eval_v3_stage2b_mcq_reasoning_distill_100 \
  --lora-adapter-path artifacts/lora_clean_v3/stage2b_mcq_reasoning_distill/final_adapter \
  "${COMMON_EVAL_ARGS[@]}"

echo "=== Compare A/B/C ==="
"${PYTHON_BIN}" scripts/modular_pipeline/compare_runs.py \
  --input public \
  --runs \
    results/eval_v3_stage1_reasoning_100 \
    results/eval_v3_stage2_mcq_cleanup_100 \
    results/eval_v3_stage2b_mcq_reasoning_distill_100

echo "Stage 2B pilot complete."
