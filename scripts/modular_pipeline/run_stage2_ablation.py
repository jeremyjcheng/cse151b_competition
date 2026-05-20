#!/usr/bin/env python3
"""Stage 2 (adapt) hyperparameter grid for comparing lightweight formatting tuning vs reasoning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run Stage 2 LoRA adaptation ablations on a fixed Stage 1 adapter. "
            "Each combo writes under --adapter-root/stage2_ablate_<suffix>/final_adapter."
        ),
    )
    p.add_argument(
        "--adapter-root",
        required=True,
        help="Root directory for ablation outputs (created if missing).",
    )
    p.add_argument(
        "--stage1-adapter-path",
        required=True,
        help="Path to Stage 1 final_adapter directory.",
    )
    p.add_argument(
        "--adapt-input",
        default="public",
        help="Same as train_lora.py --input (default: public).",
    )
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--limit-mcq", type=int, default=None)
    p.add_argument("--limit-free", type=int, default=None)
    p.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[2e-5, 3e-5, 5e-5],
        help="Stage 2 learning rates to sweep.",
    )
    p.add_argument(
        "--max-steps-list",
        nargs="+",
        type=int,
        default=[50, 75, 100, 150],
        help="Stage 2 optimizer steps to sweep.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not train.",
    )
    p.add_argument(
        "--eval-after-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run eval_runner.py on each run's stage2_holdout.jsonl after training.",
    )
    p.add_argument(
        "--holdout-path",
        default=None,
        help="Override holdout JSONL path (default: <run>/stage2_holdout.jsonl).",
    )
    p.add_argument("--vllm-quantization", default=None)
    p.add_argument("--vllm-load-format", default=None)
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    train_lora = here / "train_lora.py"

    adapter_root = Path(args.adapter_root).resolve()
    adapter_root.mkdir(parents=True, exist_ok=True)

    stage1 = Path(args.stage1_adapter_path).resolve()
    if not stage1.is_dir():
        raise SystemExit(f"Stage 1 adapter not found: {stage1}")

    grid: list[dict] = []
    for lr in args.learning_rates:
        for steps in args.max_steps_list:
            for train_full in (False, True):
                lr_tag = f"{lr:.0e}".replace("e-0", "e-").replace("+", "")
                suffix = f"lr{lr_tag}_steps{steps}_fullchat{int(train_full)}"
                out_dir = adapter_root / f"stage2_ablate_{suffix}"
                cmd = [
                    sys.executable,
                    str(train_lora),
                    "--stage",
                    "adapt",
                    "--input",
                    args.adapt_input,
                    "--output-dir",
                    str(out_dir),
                    "--gpu-id",
                    args.gpu_id,
                    "--resume-from-adapter",
                    str(stage1),
                    "--max-steps",
                    str(steps),
                    "--learning-rate",
                    str(lr),
                    "--batch-size",
                    str(args.batch_size),
                    "--grad-accum-steps",
                    str(args.grad_accum_steps),
                    "--max-seq-length",
                    str(args.max_seq_length),
                    "--sample-seed",
                    str(args.sample_seed),
                ]
                if train_full:
                    cmd.append("--train-on-full-chat")
                if args.limit_mcq is not None:
                    cmd.extend(["--limit-mcq", str(args.limit_mcq)])
                if args.limit_free is not None:
                    cmd.extend(["--limit-free", str(args.limit_free)])

                grid.append(
                    {
                        "suffix": suffix,
                        "output_dir": str(out_dir),
                        "learning_rate": lr,
                        "max_steps": steps,
                        "train_on_full_chat": train_full,
                        "command": cmd,
                    }
                )

    manifest_path = adapter_root / "stage2_ablation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"Wrote {len(grid)} runs to manifest: {manifest_path}")

    for entry in grid:
        print("\n" + "=" * 60)
        print(entry["suffix"])
        print(" ".join(entry["command"]))
        if args.dry_run:
            continue
        subprocess.run(entry["command"], check=True)

        if args.eval_after_train:
            out_dir = Path(entry["output_dir"])
            holdout = (
                Path(args.holdout_path).resolve()
                if args.holdout_path
                else out_dir / "stage2_holdout.jsonl"
            )
            adapter = out_dir / "final_adapter"
            if holdout.is_file() and adapter.is_dir():
                eval_cmd = [
                    sys.executable,
                    str(here / "eval_runner.py"),
                    "--input",
                    str(holdout),
                    "--lora-adapter-path",
                    str(adapter),
                    "--split-name",
                    "val",
                    "--gpu-id",
                    args.gpu_id,
                    "--eval-report",
                    str(out_dir / "holdout_eval.json"),
                ]
                if args.vllm_quantization:
                    eval_cmd.extend(["--vllm-quantization", args.vllm_quantization])
                if args.vllm_load_format:
                    eval_cmd.extend(["--vllm-load-format", args.vllm_load_format])
                if args.enforce_eager is not None:
                    flag = "--enforce-eager" if args.enforce_eager else "--no-enforce-eager"
                    eval_cmd.append(flag)
                print("Running holdout eval:", " ".join(eval_cmd))
                subprocess.run(eval_cmd, check=False)
            else:
                print(f"Skipping eval for {entry['suffix']}: missing holdout or adapter.")

    print("\nDone. Compare adapters under:", adapter_root)
    print(
        "Checkpoint sweep: python eval_runner.py --input <holdout> "
        "--checkpoint-dir <adapter_root>"
    )


if __name__ == "__main__":
    main()
