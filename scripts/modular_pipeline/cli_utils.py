"""CLI parsing and run configuration helpers."""

import argparse
from pathlib import Path


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
    return parser.parse_args()


def resolve_input_path(arg_value: str, root: Path) -> Path:
    if arg_value == "private":
        return root / "data" / "private.jsonl"
    if arg_value == "public":
        return root / "data" / "public.jsonl"

    path_value = Path(arg_value)
    return path_value if path_value.is_absolute() else (root / path_value)


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
