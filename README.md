# CSE 151B Competition Submission

## Overview

This submission uses the base version of **Qwen/Qwen3-4B-Thinking-2507** without fine-tuning, LoRA adapters, or additional training.

| Item                 | Value                                                 |
| -------------------- | ----------------------------------------------------- |
| **Model**            | `Qwen/Qwen3-4B-Thinking-2507`                         |
| **Inference engine** | vLLM with bitsandbytes INT8 quantization              |
| **Entry point**      | `run_inference.py` (defaults to the private test set) |
| **Final output**     | `results/private_submission.csv`                      |

The system improves performance through prompting, reasoning-oriented decoding, MCQ verification, answer extraction, and post-processing while keeping the original model weights unchanged.

All hyperparameters live in `scripts/modular_pipeline/settings.py`.

---

## Datasets

Place competition JSONL files under `data/` (this directory is gitignored; it is not shipped in the repo).

The pipeline splits questions the same way as `runner.py`: a row is **MCQ** if it has a non-empty `options` field; otherwise it is **free-response (FRQ)**.

| File                                           | Total | MCQ | FRQ |
| ---------------------------------------------- | ----: | --: | --: |
| `data/private.jsonl` (default inference input) |   943 | 300 | 643 |
| `data/public.jsonl` (local dev / judger eval)  |  1126 | 375 | 751 |

---

## Method

The pipeline maximizes answer quality with inference-time techniques rather than training:

- Long-form reasoning generation (Qwen thinking mode for MCQ primary pass)
- Multi-stage MCQ extraction and confidence-based verification
- Automatic response post-processing and `\boxed{}` handling
- Incremental checkpointing in `results/*_outputs.jsonl` with resume on rerun

At default settings, generation caps are **12,288** tokens per MCQ item and **16,384** per FRQ item (`MAX_BASE_ENABLED` doubles the 6144 / 8192 baselines). Batch sizes are **12** (MCQ) and **2** (FRQ).

---

## Installation

```bash
pip install -r requirements.txt
```

The model is downloaded from Hugging Face on first run and cached locally.

---

## Running inference

From the repository root:

```bash
python run_inference.py
```

Private test set (default): `data/private.jsonl` → `results/private_submission.csv`.

Public development set (runs judger accuracy when `answer` fields are present):

```bash
python run_inference.py --input public
```

Optional GPU selection:

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py
```

Full CLI (LoRA, limits, vLLM overrides) is available via:

```bash
python scripts/modular_pipeline/modular_pipeline.py --help
```

---

## Outputs

| File                                    | Description                                 |
| --------------------------------------- | ------------------------------------------- |
| `results/private_submission.csv`        | Final competition submission (private run)  |
| `results/private_outputs.jsonl`         | Incremental raw records (resume checkpoint) |
| `results/private_outputs_ordered.jsonl` | Same records, ordered to match input        |

Stem names follow the input file (e.g. `public_submission.csv` for `--input public`).

If a run stops early, rerun the same command; completed question IDs in `*_outputs.jsonl` are skipped.

---

## Runtime

Measured on **NVIDIA A30 (24 GB VRAM)** for an uninterrupted private run (943 questions: 300 MCQ, 643 FRQ).

| Stage                                                            | Approximate time |
| ---------------------------------------------------------------- | ---------------- |
| Model initialization                                             | 3–5 minutes      |
| MCQ (300 items, batch 12; includes verify/finalizer on a subset) | 3–5 hours        |
| FRQ (643 items, batch 2)                                         | 5–8 hours        |
| **Total (uninterrupted)**                                        | **~8–13 hours**  |

Wall-clock time can be much higher if the job is interrupted: the pipeline resumes from `results/private_outputs.jsonl`, but each restart reloads the model.

---

## Configuration

Edit `scripts/modular_pipeline/settings.py`.

| Setting                              | Default (base run)            | Role                                |
| ------------------------------------ | ----------------------------- | ----------------------------------- |
| `MODEL_ID`                           | `Qwen/Qwen3-4B-Thinking-2507` | Hugging Face model id               |
| `MAX_TOKENS_MCQ`                     | 12288                         | MCQ primary generation cap          |
| `MAX_TOKENS_FREE`                    | 16384                         | FRQ generation cap                  |
| `VLLM_MAX_MODEL_LEN`                 | 20480                         | vLLM context length                 |
| `MCQ_BATCH_SIZE` / `FREE_BATCH_SIZE` | 12 / 2                        | Inference batch sizes               |
| `ENABLE_THINKING_MCQ_PRIMARY`        | `True`                        | Thinking mode on MCQ primary pass   |
| `MCQ_VERIFY_ENABLED`                 | `True`                        | Low-confidence MCQ verify/finalizer |

---

## Repository layout

```text
run_inference.py              # competition entry point
scripts/modular_pipeline/
  settings.py                 # all hyperparameters
  run_inference.py            # pipeline wrapper used by run_inference.py
  modular_pipeline.py         # full CLI (delegates to runner.py)
  runner.py                   # batched inference + resume
data/                         # local only (gitignored)
  private.jsonl
  public.jsonl
results/                      # gitignored
judger.py
utils.py
```

---

## Public evaluation

Same pipeline and settings; judger runs automatically when answers are available:

```bash
python run_inference.py --input public
```

Equivalent full CLI:

```bash
python scripts/modular_pipeline/modular_pipeline.py --input public
```

---

## Summary

This submission uses unmodified **Qwen/Qwen3-4B-Thinking-2507** weights and relies on inference-time prompting, decoding, MCQ verification, and post-processing. No fine-tuning or LoRA adapters are used for the base submission path.
