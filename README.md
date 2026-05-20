# CSE 151B Competition — Starter Code

Open **`scripts/starter_code_cse151b_comp.ipynb`** for the baseline notebook, or use the modular pipeline under **`scripts/modular_pipeline/`** for LoRA training and vLLM inference.

## Contents

| Path | Description |
|------|-------------|
| `scripts/starter_code_cse151b_comp.ipynb` | Baseline notebook (vLLM INT8) |
| `scripts/modular_pipeline/` | Two-stage LoRA + vLLM inference + evaluation |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers (local only) |
| `data/private.jsonl` | Hidden test set (no answers) |
| `results/` | Output JSONL / eval reports |

## Data splits

| Split | File / alias | Use |
|-------|----------------|-----|
| **Training** | Public train slice after holdout split | Stage-2 LoRA SFT only |
| **Validation** | `stage2_holdout.jsonl` or `--input holdout` | Tune prompts, LoRA, pick checkpoint |
| **Hidden test** | `data/private.jsonl` or `--input private` | Leaderboard submission |

Stage 2 writes `stage2_holdout.jsonl` under the training output directory. **Do not train on holdout items.**

## Quick start: evaluation

```bash
# Score one adapter on holdout (validation)
python scripts/modular_pipeline/eval_runner.py \
  --input workspaces/stage2/stage2_holdout.jsonl \
  --lora-adapter-path workspaces/stage2/final_adapter \
  --split-name val \
  --vllm-quantization none \
  --vllm-load-format auto

# Sweep all checkpoints (leaderboard CSV + JSON report)
python scripts/modular_pipeline/eval_runner.py \
  --input workspaces/stage2/stage2_holdout.jsonl \
  --checkpoint-dir workspaces/stage2 \
  --split-name val \
  --vllm-quantization none \
  --vllm-load-format auto
```

Reports are written under `results/eval_<split>_<timestamp>.json` with validation accuracy, exact-match rates, format failures, extraction vs reasoning failures, and latency (p50/p95, questions/sec).

## LoRA + vLLM stability matrix

`VLLM_ENFORCE_EAGER=False` in `setting.py` enables CUDA graphs (faster inference). LoRA requires a recent vLLM and often **no quantization** during verification:

```bash
pip install 'vllm>=0.10.0'

python scripts/modular_pipeline/verify_lora_vllm.py \
  --lora-adapter-path <adapter>/final_adapter \
  --vllm-quantization none \
  --vllm-load-format auto \
  --no-enforce-eager
```

| Step | Config | Goal |
|------|--------|------|
| A | `enforce_eager=True` | Confirm LoRA applies |
| B | `enforce_eager=False`, no quant | Target: fast + LoRA |
| C | Re-enable `bitsandbytes` | Only if B passes |

Optional: `export VLLM_ATTENTION_BACKEND=FLASH_ATTN` when your vLLM build supports it.

## Curated hard examples

```bash
# After a public run with saved outputs JSONL
python scripts/modular_pipeline/curate_data.py \
  --predictions results/public_outputs.jsonl \
  --output data/hard_examples.jsonl

# Stage-2 train on curated set
python scripts/modular_pipeline/train_lora.py \
  --stage adapt \
  --output-dir workspaces/stage2 \
  --curated-input data/hard_examples.jsonl \
  --val-eval-every-steps 20
```

## End-to-end workflow

```bash
python scripts/modular_pipeline/run_lora_workspaces.py \
  --adapter-root workspaces \
  --include-openmath \
  --vllm-quantization none \
  --vllm-load-format auto
```

This runs Stage 1 → Stage 2 → holdout eval → checkpoint sweep → optional private inference.
