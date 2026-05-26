"""CLI parsing and run configuration helpers."""

import argparse
import json
import re
from pathlib import Path

from lora_vllm_utils import validate_lora_adapter_dir
from settings import (
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LEN,
    MAX_STEPS,
    SAVE_EVERY_STEPS,
    TRAIN_BATCH_SIZE,
    VLLM_ENFORCE_EAGER,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular batched inference for Qwen3-4B-Thinking.",
    )
    parser.add_argument(
        "--input",
        default="private",
        help=(
            "'private' (default), 'public', or a path to a .jsonl file. "
            "'private' targets the leaderboard test set; 'public' enables "
            "judger-based local accuracy reporting."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Default: <project root>/results",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip judger evaluation even if the input has 'answer' fields.",
    )
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="CUDA_VISIBLE_DEVICES value passed through to the pipeline.",
    )
    parser.add_argument(
        "--lora-adapter-path",
        default=None,
        help=(
            "Optional local path to a trained LoRA adapter directory. "
            "If provided, adapter weights are loaded via vLLM LoRA (enable_lora + LoRARequest)."
        ),
    )
    parser.add_argument(
        "--vllm-quantization",
        default=None,
        help=(
            "Override vLLM quantization (default from settings: bitsandbytes). "
            "Use 'none' to disable quantization for LoRA debugging."
        ),
    )
    parser.add_argument(
        "--vllm-load-format",
        default=None,
        help=(
            "Override vLLM load_format (default from settings: bitsandbytes). "
            "Use 'auto' with --vllm-quantization none for full-precision LoRA experiments."
        ),
    )
    parser.add_argument(
        "--limit-mcq",
        type=int,
        default=None,
        help=(
            "Cap the number of MCQ items processed (random subset, fixed by "
            "--sample-seed). Default: no cap."
        ),
    )
    parser.add_argument(
        "--limit-free",
        type=int,
        default=None,
        help=(
            "Cap the number of free-form items processed (random subset, fixed "
            "by --sample-seed). Default: no cap."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for the random subset selection used by --limit-mcq / --limit-free.",
    )
    parser.add_argument(
        "--save-raw-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled (default), persist full raw model generations in JSONL "
            "outputs under the `raw` field. Disable with --no-save-raw-output."
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override vLLM enforce_eager (CUDA graphs disabled when True). "
            f"Default from settings: {VLLM_ENFORCE_EAGER}."
        ),
    )
    args = parser.parse_args()
    if args.enforce_eager is None:
        args.enforce_eager = VLLM_ENFORCE_EAGER
    return args


def parse_eval_args() -> argparse.Namespace:
    """CLI for eval_runner.py (inference + metrics JSON + checkpoint sweep)."""
    parser = argparse.ArgumentParser(
        description="Run inference and write evaluation metrics with latency.",
    )
    parser.add_argument(
        "--input",
        default="public",
        help="'public', 'private', 'holdout' (alias for stage2_holdout.jsonl path), or .jsonl path.",
    )
    parser.add_argument(
        "--split-name",
        default=None,
        choices=("train", "val", "test"),
        help="Metadata label for eval reports: train, val, or test.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for eval JSON/CSV outputs. Default: <project>/results",
    )
    parser.add_argument(
        "--eval-report",
        default=None,
        help="Path for combined eval JSON report (default: results/eval_<split>_<ts>.json).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "If set, evaluate every checkpoint-step-* and final_adapter under this "
            "directory and write a leaderboard CSV."
        ),
    )
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Load base model in 4-bit via bitsandbytes. Default False (bfloat16) — "
            "safer on clusters missing libnvJitLink.so.13."
        ),
    )
    parser.add_argument(
        "--load-in-8bit",
        dest="load_in_8bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load base model in 8-bit (less VRAM than bf16; may work when 4-bit fails).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable gradient checkpointing to reduce VRAM (default: on).",
    )
    parser.add_argument(
        "--use-bnb-optimizer",
        dest="use_bnb_optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use bitsandbytes PagedAdamW8bit (default: torch AdamW on LoRA only).",
    )
    parser.add_argument("--lora-adapter-path", default=None)
    parser.add_argument("--vllm-quantization", default=None)
    parser.add_argument("--vllm-load-format", default=None)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Override vLLM enforce_eager. Default from settings: {VLLM_ENFORCE_EAGER}.",
    )
    parser.add_argument("--limit-mcq", type=int, default=None)
    parser.add_argument("--limit-free", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--save-raw-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include raw generations in per-item records (eval_runner only).",
    )
    args = parser.parse_args()
    if args.enforce_eager is None:
        args.enforce_eager = VLLM_ENFORCE_EAGER
    return args


def discover_adapter_checkpoints(adapter_root: Path) -> list[Path]:
    """Return sorted checkpoint-step-* dirs plus final_adapter when present."""
    adapter_root = adapter_root.resolve()
    if not adapter_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {adapter_root}")

    step_dirs: list[tuple[int, Path]] = []
    for path in adapter_root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-step-(\d+)", path.name)
        if match:
            try:
                validate_lora_adapter_dir(path)
            except FileNotFoundError:
                continue
            step_dirs.append((int(match.group(1)), path))

    ordered = [p for _, p in sorted(step_dirs, key=lambda x: x[0])]
    final = adapter_root / "final_adapter"
    if final.is_dir():
        try:
            validate_lora_adapter_dir(final)
            ordered.append(final)
        except FileNotFoundError:
            pass
    return ordered


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning with a custom PyTorch loop.",
    )
    parser.add_argument(
        "--stage",
        choices=("reasoning", "adapt"),
        default="adapt",
        help=(
            "Training stage. `reasoning` trains on public reasoning datasets; "
            "`adapt` lightly adapts formatting on competition data."
        ),
    )
    parser.add_argument(
        "--input",
        default="public",
        help=(
            "'public' (default), 'private', or a path to a .jsonl file with "
            "optional `answer` fields used as supervision targets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where LoRA adapter checkpoints and final adapter are saved.",
    )
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="CUDA_VISIBLE_DEVICES value passed through to training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAIN_BATCH_SIZE,
        help=f"Per-step micro-batch size. Default: {TRAIN_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=GRAD_ACCUM_STEPS,
        help=f"Gradient accumulation steps. Default: {GRAD_ACCUM_STEPS}.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Optimizer learning rate. Default: {LEARNING_RATE}.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help=f"Optimizer weight decay. Default: {WEIGHT_DECAY}.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum optimizer steps. Default: {MAX_STEPS}.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=WARMUP_RATIO,
        help=f"Warmup ratio for scheduler. Default: {WARMUP_RATIO}.",
    )
    parser.add_argument(
        "--max-seq-length",
        "--max-seq-len",
        dest="max_seq_len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Training sequence length cap. Default: {MAX_SEQ_LEN}.",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=SAVE_EVERY_STEPS,
        help=f"Checkpoint save interval. Default: {SAVE_EVERY_STEPS}.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=LORA_R,
        help=f"LoRA rank. Default: {LORA_R}.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=LORA_ALPHA,
        help=f"LoRA alpha. Default: {LORA_ALPHA}.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=LORA_DROPOUT,
        help=f"LoRA dropout. Default: {LORA_DROPOUT}.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=list(LORA_TARGET_MODULES),
        help=(
            "One or more module names to target with LoRA adapters. "
            f"Default: {' '.join(LORA_TARGET_MODULES)}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for data order and reproducibility.",
    )
    parser.add_argument(
        "--limit-mcq",
        type=int,
        default=None,
        help="Optional cap for MCQ examples during training.",
    )
    parser.add_argument(
        "--limit-free",
        type=int,
        default=None,
        help="Optional cap for free-form examples during training.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for subset selection when limits are enabled.",
    )
    parser.add_argument(
        "--include-openmath",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include unsloth/OpenMathReasoning-mini in stage `reasoning` "
            "(opt-in)."
        ),
    )
    parser.add_argument(
        "--include-hendrycks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include EleutherAI/hendrycks_math in stage `reasoning` (opt-in).",
    )
    parser.add_argument(
        "--max-openmath-examples",
        type=int,
        default=None,
        help="Optional cap for OpenMath examples in stage `reasoning`.",
    )
    parser.add_argument(
        "--max-hendrycks-examples",
        type=int,
        default=None,
        help="Optional cap for Hendrycks examples in stage `reasoning`.",
    )
    parser.add_argument(
        "--hendrycks-configs",
        nargs="+",
        default=[
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ],
        help=(
            "Hendrycks subject configs to load when --include-hendrycks is set."
        ),
    )
    parser.add_argument(
        "--train-on-full-chat",
        "--stage2-train-on-full-chat",
        dest="train_on_full_chat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, train on all assistant tokens in the completion. "
            "Keep disabled for conservative Stage-2 adaptation to avoid style copying."
        ),
    )
    parser.add_argument(
        "--stage2-final-answer-only",
        dest="stage2_final_answer_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stage-2 only: for free-form, supervise final boxed answers only. "
            "MCQ still uses reasoning scaffolds when --stage2-mcq-with-reasoning is on."
        ),
    )
    parser.add_argument(
        "--stage2-mcq-with-reasoning",
        dest="stage2_mcq_with_reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stage-2 only: train MCQ on brief reasoning + \\boxed{letter} instead of "
            "bare \\boxed{A}. Default: True (see setting.STAGE2_MCQ_WITH_REASONING)."
        ),
    )
    parser.add_argument(
        "--stage2-freeze-reasoning-style",
        dest="stage2_freeze_reasoning_style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stage-2 only: keep adaptation focused on answer format and avoid re-teaching "
            "reasoning style from small public data."
        ),
    )
    parser.add_argument(
        "--stage2-holdout-fraction",
        type=float,
        default=None,
        help=(
            "Stage-2 only: fraction of supervised public items reserved for eval (not trained on). "
            "Use this so local public scoring reflects generalization, not memorization."
        ),
    )
    parser.add_argument(
        "--stage2-holdout-seed",
        type=int,
        default=0,
        help="Seed for Stage-2 train/holdout split.",
    )
    parser.add_argument(
        "--resume-from-adapter",
        default=None,
        help=(
            "Optional adapter directory to resume from before current stage "
            "training (typically Stage 2 adaption from Stage 1 adapter)."
        ),
    )
    parser.add_argument(
        "--save-final-merged",
        action="store_true",
        help="If set, also save a merged full model checkpoint (large).",
    )
    parser.add_argument(
        "--curated-input",
        default=None,
        help="Optional JSONL of curated hard examples (overrides default adapt sampling).",
    )
    parser.add_argument(
        "--val-eval-every-steps",
        type=int,
        default=0,
        help=(
            "If > 0, run lightweight holdout eval every N optimizer steps "
            "(requires stage2_holdout.jsonl in output-dir)."
        ),
    )
    parser.add_argument(
        "--val-eval-max-items",
        type=int,
        default=30,
        help="Max holdout items for periodic validation during training.",
    )
    return parser.parse_args()


def resolve_input_path(arg_value: str, root: Path) -> Path:
    if arg_value == "private":
        return root / "data" / "private.jsonl"
    if arg_value == "public":
        return root / "data" / "public.jsonl"
    if arg_value == "holdout":
        holdout = root / "workspaces" / "stage2" / "stage2_holdout.jsonl"
        if holdout.is_file():
            return holdout
        raise FileNotFoundError(
            f"Holdout not found at {holdout}. Train Stage 2 first or pass a .jsonl path."
        )

    path_value = Path(arg_value)
    return path_value if path_value.is_absolute() else (root / path_value)


def split_train_holdout(
    data: list[dict],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Deterministically split supervised items into train vs holdout eval sets."""
    if not data:
        return [], []
    if holdout_fraction <= 0:
        return data, []
    if holdout_fraction >= 1:
        return [], list(data)

    import random as _random

    rng = _random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_holdout = max(1, int(round(len(data) * holdout_fraction)))
    n_holdout = min(n_holdout, len(data) - 1)

    holdout_set = set(indices[:n_holdout])
    train = [data[i] for i in range(len(data)) if i not in holdout_set]
    holdout = [data[i] for i in range(len(data)) if i in holdout_set]
    return train, holdout


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def normalize_train_limit(value: int | None) -> int | None:
    """Map CLI 0 to no cap (None) for Stage-2 training and eval subsets."""
    if value is None or value <= 0:
        return None
    return value


def apply_subset_caps(
    data: list[dict],
    *,
    limit_mcq: int | None,
    limit_free: int | None,
    seed: int,
) -> list[dict]:
    """Return a deterministic per-type-capped subset of `data`."""
    limit_mcq = normalize_train_limit(limit_mcq)
    limit_free = normalize_train_limit(limit_free)
    if limit_mcq is None and limit_free is None:
        return data

    import random as _random

    rng = _random.Random(seed)

    mcq_indices = [i for i, item in enumerate(data) if item.get("options")]
    free_indices = [i for i, item in enumerate(data) if not item.get("options")]

    if limit_mcq is not None and limit_mcq < len(mcq_indices):
        mcq_pick = sorted(rng.sample(mcq_indices, limit_mcq))
    else:
        mcq_pick = mcq_indices

    if limit_free is not None and limit_free < len(free_indices):
        free_pick = sorted(rng.sample(free_indices, limit_free))
    else:
        free_pick = free_indices

    keep = sorted(set(mcq_pick) | set(free_pick))
    selected = [data[i] for i in keep]

    print(
        f"Subset: {len(mcq_pick)}/{len(mcq_indices)} MCQ + "
        f"{len(free_pick)}/{len(free_indices)} free-form "
        f"(seed={seed}) -> {len(selected)} items"
    )
    return selected


def build_run_stem(input_stem: str, args: argparse.Namespace) -> str:
    """Append a deterministic suffix when subset caps are active."""
    parts: list[str] = []
    mcq_cap = normalize_train_limit(getattr(args, "limit_mcq", None))
    free_cap = normalize_train_limit(getattr(args, "limit_free", None))
    if mcq_cap is not None:
        parts.append(f"mcq{mcq_cap}")
    if free_cap is not None:
        parts.append(f"free{free_cap}")
    if not parts:
        return input_stem
    parts.append(f"seed{args.sample_seed}")
    return f"{input_stem}_{'_'.join(parts)}"
