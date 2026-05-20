"""CLI parsing and run configuration helpers."""

import argparse
import json
from pathlib import Path

from settings import (
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_TOKENS_FREE,
    MAX_TOKENS_MCQ,
    MAX_TOKENS_MCQ_FINAL,
    MAX_SEQ_LEN,
    MAX_STEPS,
    SAVE_EVERY_STEPS,
    TRAIN_BATCH_SIZE,
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
        "--no-bitsandbytes",
        action="store_true",
        help=(
            "Disable vLLM bitsandbytes quantization (sets quantization=none, "
            "load_format=auto). Use when LoRA + bitsandbytes hangs on first generate."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default=None,
        help=(
            "Load base weights in full precision (implies --no-bitsandbytes). "
            "Use bfloat16 for LoRA debugging on Ampere+ GPUs."
        ),
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help=(
            "Set vLLM enforce_eager=True (skip CUDA graphs; can avoid hangs during "
            "first LoRA generate / torch.compile)."
        ),
    )
    parser.add_argument(
        "--inference-backend",
        choices=("vllm", "peft"),
        default="vllm",
        help=(
            "Inference engine. Default vllm. Use peft for Transformers+PEFT LoRA "
            "when vLLM LoRA is unstable (requires --lora-adapter-path)."
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
        "--mcq-max-new-tokens",
        type=int,
        default=MAX_TOKENS_MCQ,
        help=f"Max new tokens for MCQ primary generation. Default: {MAX_TOKENS_MCQ}.",
    )
    parser.add_argument(
        "--mcq-final-max-new-tokens",
        type=int,
        default=MAX_TOKENS_MCQ_FINAL,
        help=(
            "Max new tokens for MCQ finalizer generation. "
            f"Default: {MAX_TOKENS_MCQ_FINAL}."
        ),
    )
    parser.add_argument(
        "--free-max-new-tokens",
        type=int,
        default=MAX_TOKENS_FREE,
        help=f"Max new tokens for free-form generation. Default: {MAX_TOKENS_FREE}.",
    )
    args = parser.parse_args()
    return apply_vllm_cli_overrides(args)


def apply_vllm_cli_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Apply convenience flags that override vLLM quantization/load_format."""
    if getattr(args, "no_bitsandbytes", False) or getattr(args, "dtype", None):
        args.vllm_quantization = "none"
        args.vllm_load_format = "auto"
        label = "no-bitsandbytes" if args.no_bitsandbytes else f"dtype={args.dtype}"
        print(f"vLLM: using full-precision load ({label})")
    return args


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning with a custom PyTorch loop.",
    )
    parser.add_argument(
        "--stage",
        choices=("reasoning", "adapt", "mcq", "mixed_reasoning_mcq"),
        default="adapt",
        help=(
            "Training stage. `reasoning` trains on public reasoning datasets; "
            "`adapt` lightly adapts formatting on competition data; "
            "`mcq` trains MCQ-focused adapter data; "
            "`mixed_reasoning_mcq` mixes FRQ reasoning + MCQ data."
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
        "--include-math-mc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include XiangPan/math-mc in MCQ-capable stages (opt-in).",
    )
    parser.add_argument(
        "--include-compmath-mcq",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include biancaraimondi/CompMath-MCQ in MCQ-capable stages (opt-in).",
    )
    parser.add_argument(
        "--max-math-mc-examples",
        type=int,
        default=None,
        help="Optional cap for math-mc examples.",
    )
    parser.add_argument(
        "--max-compmath-mcq-examples",
        type=int,
        default=None,
        help="Optional cap for CompMath-MCQ examples.",
    )
    parser.add_argument(
        "--mcq-example-weight",
        type=float,
        default=1.0,
        help=(
            "Relative weight for MCQ examples in mixed training. "
            "1.0 keeps natural counts; >1 upsamples MCQ; <1 downsamples MCQ."
        ),
    )
    parser.add_argument(
        "--print-dataset-samples",
        action="store_true",
        help="Print 3-5 formatted samples from each enabled dataset before training.",
    )
    parser.add_argument(
        "--include-base-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include base replay examples from a prior output JSONL.",
    )
    parser.add_argument(
        "--base-replay-path",
        default=None,
        help="Path to base model output JSONL used for replay filtering.",
    )
    parser.add_argument(
        "--max-base-replay-examples",
        type=int,
        default=None,
        help="Optional cap for accepted base replay examples.",
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
            "Stage-2 only: supervise final boxed answers instead of long reasoning traces. "
            "Recommended to reduce overfitting to public labels."
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
        "--target-module-set",
        choices=("full", "attention"),
        default=None,
        help=(
            "Convenience preset for --lora-target-modules. "
            "'full' = q/k/v/o_proj + gate/up/down_proj (current default). "
            "'attention' = q/k/v/o_proj only (recommended for Stage-2 "
            "format adapters to avoid MLP-driven behaviour drift)."
        ),
    )

    args = parser.parse_args()

    if args.target_module_set == "attention":
        args.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif args.target_module_set == "full":
        args.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return args


def resolve_input_path(arg_value: str, root: Path) -> Path:
    if arg_value == "private":
        return root / "data" / "private.jsonl"
    if arg_value == "public":
        return root / "data" / "public.jsonl"

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


def apply_subset_caps(
    data: list[dict],
    *,
    limit_mcq: int | None,
    limit_free: int | None,
    seed: int,
) -> list[dict]:
    """Return a deterministic per-type-capped subset of `data`."""
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
    if args.limit_mcq is not None:
        parts.append(f"mcq{args.limit_mcq}")
    if args.limit_free is not None:
        parts.append(f"free{args.limit_free}")
    if not parts:
        return input_stem
    parts.append(f"seed{args.sample_seed}")
    return f"{input_stem}_{'_'.join(parts)}"
