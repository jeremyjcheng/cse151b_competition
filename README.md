# CSE 151B Competition

This submission uses the **base model only** — no fine-tuning or LoRA adapters.

**Model:** `[Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)`  
**Inference:** vLLM with bitsandbytes INT8 quantization  
**Entry point:** `run_inference()` in `run_inference.py`

## Approach

We run the stock Qwen3-4B-Thinking model with tuned decoding hyperparameters (long reasoning traces, MCQ verify/finalizer, repetition controls, thinking-mode prompts). All settings are in `scripts/modular_pipeline/settings.py`. The pipeline applies post-processing automatically and writes the final submission CSV.

---

## Hardware and runtime

| Item                                                | Value                                                                                                        |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **GPU used**                                        | **NVIDIA A30** (24 GB VRAM)                                                                                  |
| **Private set size**                                | 943 questions (`data/private.jsonl`)                                                                         |
| **Wall-clock time (this submission)**               | **Up to ~20 hours** — run overnight with server disconnects; exact idle time between resumes was not tracked |
| **Active GPU generation (estimate, uninterrupted)** | **~10–14 hours** on A30 with current `settings.py`                                                           |

Our highest run began running overnight on a remote server that **disconnected repeatedly**. Each time, we rebooted the machine and reran the same command; the pipeline **resumed** from `results/private_outputs.jsonl` and skipped questions already completed. Total calendar time reached **~20 hours**, but that includes unknown downtime between sessions (reboot, reconnect, model reload), not 20 hours of continuous generation.

Timing breakdown for an **uninterrupted** run (approximate, with current token caps):

- **Model load + vLLM init:** ~3–5 minutes per session (first run also downloads weights; see below)
- **MCQ (~375 items, batch 12):** ~4–6 hours — long thinking traces + optional verify/finalizer pass
- **Free-form (~568 items, batch 2):** ~5–8 hours — up to 16k new tokens per item with sampling

To reproduce: run `python run_inference.py` and rerun the same command after any crash or disconnect until all 943 ids are present. Times will differ on other GPUs (e.g. faster on A100/H100, slower on T4). GPU utilization often sits around 30–60% during autoregressive decoding with small batch sizes; that is normal.

---

## Model weights setup

**No manual download or local weight directory is required.** vLLM loads `[Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)` from the Hugging Face Hub on first run and caches weights under `~/.cache/huggingface/hub/`.

---

## Reproduce results: `run_inference()`

Install dependencies once: `pip install -r requirements.txt`

Hyperparameters come from `scripts/modular_pipeline/settings.py`. **Input defaults to private**; override only if needed.

### Shell

```bash
python run_inference.py                  # private (default)
python run_inference.py --input public   # public dev set + judger eval
```

Optional GPU selection:

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py
```

### Python

```python
from run_inference import run_inference

submission_csv = run_inference()
print(submission_csv)  # -> results/private_submission.csv
```

### Outputs

| File                                    | Description                                   |
| --------------------------------------- | --------------------------------------------- |
| `results/private_submission.csv`        | **Final submission** (sorted, verified)       |
| `results/private_outputs.jsonl`         | Incremental raw inference records (resumable) |
| `results/private_outputs_ordered.jsonl` | Same records, ordered by input file           |

---

## Configuration

Edit `scripts/modular_pipeline/settings.py` to change behavior. Key knobs:

| Setting                              | Default (this submission)     | Role                                |
| ------------------------------------ | ----------------------------- | ----------------------------------- |
| `MODEL_ID`                           | `Qwen/Qwen3-4B-Thinking-2507` | Base model on Hugging Face Hub      |
| `MAX_TOKENS_MCQ`                     | 12288                         | MCQ generation cap                  |
| `MAX_TOKENS_FREE`                    | 16384                         | Free-form generation cap            |
| `VLLM_MAX_MODEL_LEN`                 | 20480                         | Context window                      |
| `MCQ_BATCH_SIZE` / `FREE_BATCH_SIZE` | 12 / 2                        | Inference batch sizes               |
| `ENABLE_THINKING_MCQ_PRIMARY`        | `True`                        | Qwen thinking mode for MCQ          |
| `MCQ_VERIFY_ENABLED`                 | `True`                        | Low-confidence MCQ verify/finalizer |

---

## Repository layout

| Path                                   | Description                            |
| -------------------------------------- | -------------------------------------- |
| `run_inference.py`                     | Competition entry point                |
| `scripts/modular_pipeline/`            | Inference pipeline                     |
| `scripts/modular_pipeline/settings.py` | All hyperparameters                    |
| `data/private.jsonl`                   | Private test set (local only)          |
| `data/public.jsonl`                    | Public dev set with answers (optional) |
| `judger.py`, `utils.py`                | Optional public-set scoring            |
| `results/`                             | Runtime outputs                        |

## Optional: evaluate on public data

```bash
python scripts/modular_pipeline/modular_pipeline.py --input public
```

Uses the same base-model pipeline and reports accuracy via `judger.py` (local dev only).
