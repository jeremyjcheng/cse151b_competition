"""Single competition entry point: base-model inference → submission CSV.

Default: ``data/private.jsonl`` → ``results/private_submission.csv``.
Override with ``--input public`` or a path to another ``.jsonl`` file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cli_utils import resolve_input_path
from runner import _discover_project_root, run_pipeline
from settings import MODEL_ID


def _ensure_project_root_on_path(root: Path) -> None:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def run_inference(input: str = "private", *, no_eval: bool | None = None) -> Path:
    """Run base-model inference; return the submission CSV path.

    ``input`` defaults to ``"private"`` (``data/private.jsonl``). Pass ``"public"``
    or a path to another JSONL file to override. Hyperparameters come from
    ``settings.py``.
    """
    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    _ensure_project_root_on_path(root)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    input_path = resolve_input_path(input, root)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    is_private = input_path.stem == "private"
    if no_eval is None:
        no_eval = is_private

    out_dir = root / "results"
    run_stem = "private" if is_private else input_path.stem

    print(f"Model: {MODEL_ID} (base only)")
    print(f"Input: {input_path.resolve()}")
    print(f"Output directory: {out_dir.resolve()}")
    print("Hyperparameters: settings.py")

    return run_pipeline(
        input_path=input_path,
        output_dir=out_dir,
        gpu_id=os.environ["CUDA_VISIBLE_DEVICES"],
        lora_adapter_path=None,
        inference_backend="vllm",
        no_eval=no_eval,
        run_stem=run_stem,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Base-model inference. Defaults to the private test set.",
    )
    parser.add_argument(
        "--input",
        default="private",
        help="'private' (default), 'public', or a path to a .jsonl file.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip judger evaluation even if the input has answer fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    submission_path = run_inference(
        input=args.input,
        no_eval=True if args.no_eval else None,
    )
    print(f"Done. Submission: {submission_path.resolve()}")


if __name__ == "__main__":
    main()
