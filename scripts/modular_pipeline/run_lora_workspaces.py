"""Run Stage 1 reasoning + Stage 2 adaptation + inference."""

import argparse
import subprocess
import sys
from pathlib import Path

from settings import (
    STAGE2_DEFAULT_HOLDOUT_FRACTION,
    STAGE2_TRAIN_LIMIT_FREE,
    STAGE2_TRAIN_LIMIT_MCQ,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end two-stage LoRA workflow with optional inference.",
    )
    parser.add_argument(
        "--adapter-root",
        required=True,
        help="Root directory where stage outputs are written.",
    )
    parser.add_argument(
        "--adapt-input",
        default="public",
        help=(
            "Competition split/path used for Stage 2 adaptation (default: public). "
            "Public is your dev set — use holdout + limits so you do not memorize it."
        ),
    )
    parser.add_argument(
        "--infer-input",
        default="private",
        help="Inference split/path passed to modular_pipeline.py (default: private).",
    )
    parser.add_argument(
        "--inference-output-dir",
        default=None,
        help="Optional inference output directory. Default is project `results/`.",
    )
    parser.add_argument("--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--reasoning-steps",
        type=int,
        default=1000,
        help="Stage 1 (reasoning) optimizer steps.",
    )
    parser.add_argument(
        "--adapt-steps",
        "--stage2-max-steps",
        dest="stage2_max_steps",
        type=int,
        default=60,
        help="Stage 2 (adaptation) optimizer steps. Keep low to avoid overfitting public labels.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Train micro-batch size.")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=16,
        help="Train gradient accumulation steps.",
    )
    parser.add_argument(
        "--reasoning-learning-rate",
        type=float,
        default=8e-5,
        help="Stage 1 learning rate.",
    )
    parser.add_argument(
        "--adapt-learning-rate",
        "--stage2-learning-rate",
        dest="stage2_learning_rate",
        type=float,
        default=1e-5,
        help="Stage 2 learning rate. Conservative default to preserve Stage-1 reasoning.",
    )
    parser.add_argument(
        "--stage2-train-on-full-chat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, Stage 2 trains on full assistant traces. "
            "Keep disabled to reduce memorization of public answer style."
        ),
    )
    parser.add_argument(
        "--stage2-final-answer-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled (default), Stage 2 supervises only final boxed answers for safer "
            "format adaptation."
        ),
    )
    parser.add_argument(
        "--stage2-freeze-reasoning-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled (default), Stage 2 prompts emphasize preserving Stage-1 reasoning style."
        ),
    )
    parser.add_argument(
        "--stage2-sanity-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a small base-vs-stage1-vs-stage2 sanity comparison after Stage 2.",
    )
    parser.add_argument(
        "--stage2-holdout-fraction",
        type=float,
        default=STAGE2_DEFAULT_HOLDOUT_FRACTION,
        help=(
            "Fraction of supervised public items reserved for eval only (default: 0.25). "
            "Stage 2 never trains on holdout items."
        ),
    )
    parser.add_argument(
        "--stage2-holdout-seed",
        type=int,
        default=0,
        help="Seed for Stage-2 train/holdout split.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Sequence length passed to train_lora.py.",
    )
    parser.add_argument(
        "--include-openmath",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include OpenMath in Stage 1 reasoning (opt-in).",
    )
    parser.add_argument(
        "--include-hendrycks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Hendrycks MATH in Stage 1 reasoning (opt-in).",
    )
    parser.add_argument(
        "--max-openmath-examples",
        type=int,
        default=None,
        help="Optional OpenMath cap for Stage 1.",
    )
    parser.add_argument(
        "--max-hendrycks-examples",
        type=int,
        default=None,
        help="Optional Hendrycks cap for Stage 1.",
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
        help="Hendrycks configs for Stage 1.",
    )
    parser.add_argument(
        "--limit-mcq",
        type=int,
        default=STAGE2_TRAIN_LIMIT_MCQ,
        help=(
            "MCQ cap for Stage-2 training on public data (default: "
            f"{STAGE2_TRAIN_LIMIT_MCQ}). Use 0 or a large value to disable cap."
        ),
    )
    parser.add_argument(
        "--limit-free",
        type=int,
        default=STAGE2_TRAIN_LIMIT_FREE,
        help=(
            "Free-form cap for Stage-2 training on public data (default: "
            f"{STAGE2_TRAIN_LIMIT_FREE}). Use 0 or a large value to disable cap."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Subset seed when using --limit-mcq/--limit-free.",
    )
    parser.add_argument(
        "--eval-after-stage2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run eval_runner.py on stage2_holdout.jsonl after Stage 2 training.",
    )
    parser.add_argument(
        "--verify-lora-before-infer",
        action="store_true",
        help=(
            "Run verify_lora_vllm.py on the inference adapter before modular_pipeline.py."
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override vLLM enforce_eager for verify/inference/eval stages.",
    )
    parser.add_argument(
        "--vllm-quantization",
        default=None,
        help="Forwarded to inference / LoRA verify (use 'none' to disable BnB).",
    )
    parser.add_argument(
        "--vllm-load-format",
        default=None,
        help="Forwarded to inference / LoRA verify (e.g. auto with --vllm-quantization none).",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Only run training; skip the inference stage.",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 and reuse an existing stage-1 adapter.",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip Stage 2 and infer with stage-1 adapter.",
    )
    parser.add_argument(
        "--stage1-adapter-path",
        default=None,
        help="Optional pre-existing Stage 1 adapter path when --skip-stage1 is set.",
    )
    return parser.parse_args()


def _append_optional(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _append_enforce_eager(cmd: list[str], args: argparse.Namespace) -> None:
    if args.enforce_eager is not None:
        cmd.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")


def _stage2_train_limit(value: int | None) -> int | None:
    """Treat 0 as 'no cap' for Stage-2 subset limits."""
    if value is None or value <= 0:
        return None
    return value


def _eval_input_for_stage2(stage2_root: Path, fallback: str) -> str:
    holdout_path = stage2_root / "stage2_holdout.jsonl"
    if holdout_path.exists():
        print(
            f"Using Stage-2 holdout for local checks: {holdout_path} "
            "(items never seen during Stage-2 training)."
        )
        return str(holdout_path)
    print(
        "Warning: stage2_holdout.jsonl not found; falling back to full public for local checks. "
        "Prefer holdout scoring to avoid optimistic dev scores."
    )
    return fallback


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent

    adapter_root = Path(args.adapter_root).resolve()
    adapter_root.mkdir(parents=True, exist_ok=True)
    stage1_root = adapter_root / "stage1_reasoning"
    stage2_root = adapter_root / "stage2_adapt"
    stage1_adapter_path = stage1_root / "final_adapter"
    stage2_adapter_path = stage2_root / "final_adapter"

    if not args.skip_stage1:
        if not (args.include_openmath or args.include_hendrycks):
            raise SystemExit(
                "Stage 1 requires at least one dataset. Pass --include-openmath and/or --include-hendrycks."
            )

        stage1_cmd = [
            sys.executable,
            str(here / "train_lora.py"),
            "--stage",
            "reasoning",
            "--output-dir",
            str(stage1_root),
            "--gpu-id",
            args.gpu_id,
            "--max-steps",
            str(args.reasoning_steps),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum-steps",
            str(args.grad_accum_steps),
            "--learning-rate",
            str(args.reasoning_learning_rate),
            "--sample-seed",
            str(args.sample_seed),
            "--max-seq-length",
            str(args.max_seq_length),
            "--train-on-full-chat",
            "--hendrycks-configs",
            *args.hendrycks_configs,
        ]
        if args.include_openmath:
            stage1_cmd.append("--include-openmath")
        else:
            stage1_cmd.append("--no-include-openmath")
        if args.include_hendrycks:
            stage1_cmd.append("--include-hendrycks")
        else:
            stage1_cmd.append("--no-include-hendrycks")
        _append_optional(stage1_cmd, "--max-openmath-examples", args.max_openmath_examples)
        _append_optional(stage1_cmd, "--max-hendrycks-examples", args.max_hendrycks_examples)

        print("Running Stage 1 reasoning command:")
        print(" ".join(stage1_cmd))
        subprocess.run(stage1_cmd, check=True)
    elif args.stage1_adapter_path:
        stage1_adapter_path = Path(args.stage1_adapter_path).resolve()
    else:
        raise SystemExit("--skip-stage1 requires --stage1-adapter-path")

    if not args.skip_stage2:
        stage2_cmd = [
            sys.executable,
            str(here / "train_lora.py"),
            "--stage",
            "adapt",
            "--input",
            args.adapt_input,
            "--output-dir",
            str(stage2_root),
            "--gpu-id",
            args.gpu_id,
            "--max-steps",
            str(args.stage2_max_steps),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum-steps",
            str(args.grad_accum_steps),
            "--learning-rate",
            str(args.stage2_learning_rate),
            "--sample-seed",
            str(args.sample_seed),
            "--max-seq-length",
            str(args.max_seq_length),
            "--resume-from-adapter",
            str(stage1_adapter_path),
        ]
        if args.stage2_train_on_full_chat:
            stage2_cmd.append("--stage2-train-on-full-chat")
        else:
            stage2_cmd.append("--no-stage2-train-on-full-chat")
        if args.stage2_final_answer_only:
            stage2_cmd.append("--stage2-final-answer-only")
        else:
            stage2_cmd.append("--no-stage2-final-answer-only")
        if args.stage2_freeze_reasoning_style:
            stage2_cmd.append("--stage2-freeze-reasoning-style")
        else:
            stage2_cmd.append("--no-stage2-freeze-reasoning-style")
        stage2_cmd.extend(
            [
                "--stage2-holdout-fraction",
                str(args.stage2_holdout_fraction),
                "--stage2-holdout-seed",
                str(args.stage2_holdout_seed),
            ]
        )
        _append_optional(stage2_cmd, "--limit-mcq", _stage2_train_limit(args.limit_mcq))
        _append_optional(stage2_cmd, "--limit-free", _stage2_train_limit(args.limit_free))
        print("Running Stage 2 adaptation command:")
        print(" ".join(stage2_cmd))
        subprocess.run(stage2_cmd, check=True)

        if args.eval_after_stage2:
            holdout_path = stage2_root / "stage2_holdout.jsonl"
            if holdout_path.is_file() and stage2_adapter_path.is_dir():
                eval_cmd = [
                    sys.executable,
                    str(here / "eval_runner.py"),
                    "--input",
                    str(holdout_path),
                    "--lora-adapter-path",
                    str(stage2_adapter_path),
                    "--split-name",
                    "val",
                    "--gpu-id",
                    args.gpu_id,
                    "--eval-report",
                    str(stage2_root / "holdout_eval_final.json"),
                ]
                _append_optional(eval_cmd, "--vllm-quantization", args.vllm_quantization)
                _append_optional(eval_cmd, "--vllm-load-format", args.vllm_load_format)
                _append_enforce_eager(eval_cmd, args)
                print("Running post-Stage-2 holdout eval:")
                print(" ".join(eval_cmd))
                subprocess.run(eval_cmd, check=False)

                sweep_cmd = [
                    sys.executable,
                    str(here / "eval_runner.py"),
                    "--input",
                    str(holdout_path),
                    "--checkpoint-dir",
                    str(stage2_root),
                    "--split-name",
                    "val",
                    "--gpu-id",
                    args.gpu_id,
                    "--eval-report",
                    str(stage2_root / "holdout_checkpoint_sweep.json"),
                ]
                _append_optional(sweep_cmd, "--vllm-quantization", args.vllm_quantization)
                _append_optional(sweep_cmd, "--vllm-load-format", args.vllm_load_format)
                _append_enforce_eager(sweep_cmd, args)
                print("Running checkpoint sweep on holdout:")
                print(" ".join(sweep_cmd))
                subprocess.run(sweep_cmd, check=False)
            else:
                print("Skipping post-Stage-2 eval: holdout or final_adapter missing.")

        if args.stage2_sanity_check:
            sanity_input = _eval_input_for_stage2(stage2_root, "public")
            sanity_cmd = [
                sys.executable,
                str(here / "sanity_check_stage_adapters.py"),
                "--stage1-adapter-path",
                str(stage1_adapter_path),
                "--stage2-adapter-path",
                str(stage2_adapter_path),
                "--gpu-id",
                args.gpu_id,
                "--input",
                sanity_input,
            ]
            _append_optional(sanity_cmd, "--vllm-quantization", args.vllm_quantization)
            _append_optional(sanity_cmd, "--vllm-load-format", args.vllm_load_format)
            _append_enforce_eager(sanity_cmd, args)
            print("Running Stage adapter sanity check:")
            print(" ".join(sanity_cmd))
            subprocess.run(sanity_cmd, check=True)

    if args.skip_infer:
        print("Training stages finished. Skipping inference stage by request.")
        return

    infer_adapter = stage2_adapter_path if not args.skip_stage2 else stage1_adapter_path

    if args.verify_lora_before_infer:
        verify_input = (
            _eval_input_for_stage2(stage2_root, "public")
            if not args.skip_stage2
            else "public"
        )
        verify_cmd = [
            sys.executable,
            str(here / "verify_lora_vllm.py"),
            "--lora-adapter-path",
            str(infer_adapter),
            "--gpu-id",
            args.gpu_id,
            "--input",
            verify_input,
        ]
        _append_optional(verify_cmd, "--vllm-quantization", args.vllm_quantization)
        _append_optional(verify_cmd, "--vllm-load-format", args.vllm_load_format)
        _append_enforce_eager(verify_cmd, args)
        print("Running LoRA verification:")
        print(" ".join(verify_cmd))
        subprocess.run(verify_cmd, check=True)

    infer_cmd = [
        sys.executable,
        str(here / "modular_pipeline.py"),
        "--input",
        args.infer_input,
        "--gpu-id",
        args.gpu_id,
        "--lora-adapter-path",
        str(infer_adapter),
    ]
    _append_optional(infer_cmd, "--output-dir", args.inference_output_dir)
    _append_optional(infer_cmd, "--vllm-quantization", args.vllm_quantization)
    _append_optional(infer_cmd, "--vllm-load-format", args.vllm_load_format)
    _append_enforce_eager(infer_cmd, args)

    print("Running adapter-backed inference command:")
    print(" ".join(infer_cmd))
    subprocess.run(infer_cmd, check=True)


if __name__ == "__main__":
    main()
