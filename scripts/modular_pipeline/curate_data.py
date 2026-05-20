#!/usr/bin/env python3
"""Build curated hard-example training JSONL from model predictions on public data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cli_utils import resolve_input_path, write_jsonl
from evaluation import compute_evaluation_metrics


def _discover_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start.parent


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_records(path: Path) -> dict:
    records: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("id") is not None:
                records[rec["id"]] = rec
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curate hard/edge examples for Stage-2 training.")
    p.add_argument("--input", default="public", help="Supervised dataset (default: public).")
    p.add_argument(
        "--predictions",
        required=True,
        help="JSONL of model outputs (id, response, optional meta/raw) from a prior run.",
    )
    p.add_argument(
        "--output",
        default="data/hard_examples.jsonl",
        help="Output curated JSONL path.",
    )
    p.add_argument(
        "--max-mcq",
        type=int,
        default=50,
        help="Max incorrect MCQ examples to include.",
    )
    p.add_argument(
        "--max-free",
        type=int,
        default=25,
        help="Max incorrect free-form examples to include.",
    )
    p.add_argument(
        "--include-format-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include items with extraction/format failures even if judger might pass.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    input_path = resolve_input_path(args.input, root)
    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = root / pred_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    data = [item for item in _load_jsonl(input_path) if item.get("answer") is not None]
    records_by_id = _load_records(pred_path)
    metrics = compute_evaluation_metrics(data, records_by_id, verbose=True)

    from evaluation import _load_project_judger, _safe_auto_judge, is_format_valid

    Judger = _load_project_judger()
    if Judger is None:
        raise SystemExit("Judger required for curation.")
    judger = Judger(strict_extract=False)

    hard_mcq: list[dict] = []
    hard_free: list[dict] = []

    for item in data:
        rec = records_by_id.get(item.get("id"))
        if rec is None:
            continue
        pred = rec.get("response", "")
        meta = rec.get("meta") or {}
        is_mcq = bool(item.get("options"))

        answer = item.get("answer")
        gold = answer if isinstance(answer, list) else [answer]
        options_per_slot = [item.get("options", [])] * len(gold)
        labels = [chr(65 + i) for i in range(len(item.get("options", [])))] if is_mcq else []
        format_ok = is_format_valid(pred, is_mcq=is_mcq, labels=labels)
        correct = _safe_auto_judge(judger, pred=pred, gold=gold, options_per_slot=options_per_slot)

        include = not correct
        if args.include_format_failures and (meta.get("malformed_output") or not format_ok):
            include = True

        if not include:
            continue

        if is_mcq and len(hard_mcq) < args.max_mcq:
            hard_mcq.append(item)
        elif not is_mcq and len(hard_free) < args.max_free:
            hard_free.append(item)

    curated = hard_mcq + hard_free
    write_jsonl(output_path, curated)
    print(f"Wrote {len(curated)} curated examples to {output_path.resolve()}")
    print(
        f"  incorrect/filtered: {len(hard_mcq)} MCQ, {len(hard_free)} free-form "
        f"(metrics overall acc={metrics.get('validation_accuracy_pct', 0):.2f}%)"
    )


if __name__ == "__main__":
    main()
