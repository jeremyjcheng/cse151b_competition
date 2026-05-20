"""Curate `data/public.jsonl` into a clean LoRA training file.

Filters out items the trainer cannot supervise cleanly:

- Items without a gold `answer` field.
- MCQ items whose `answer` cannot be resolved to a valid option letter.
- Items whose canonical target would have zero or multiple final
  ``\\boxed{...}`` spans (training on noisy targets is a primary cause of
  the repeated-box failure mode observed in the current adapter).

Optional filters:

- ``--only-base-fails``: keep only items where a prior base run got the
  answer wrong per the project judger. Useful to focus the adapter on
  the cases where it has a real chance of adding value, and to keep it
  away from items base already solves.
- ``--mcq-only`` / ``--free-only``: restrict the output by item type so
  separate MCQ and free-form adapters can be trained.

The script prints kept / dropped counts by reason and writes the cleaned
JSONL to ``data/public_clean_for_lora.jsonl`` (override with
``--output-path``). The schema of the output JSONL matches
``data/public.jsonl`` so it can be fed directly to ``train_lora.py``
via ``--input <output_path>``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from text_processing import (
    ensure_boxed,
    extract_all_boxed,
    extract_boxed,
    extract_valid_letter,
)


# The next four helpers are local copies of small utilities from
# ``train_lora.py``. We duplicate them here so this script can run on a CPU
# box without pulling in torch / transformers via the trainer module. The
# behaviour must stay in sync with their counterparts in ``train_lora.py``.

def _discover_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    return start.parent


def _strip_boxed(text: str) -> str:
    return re.sub(r"\\boxed\s*\{[^{}]*\}", "", text)


def _enforce_single_final_boxed(response: str, fallback_answer: str = "") -> str:
    cleaned = str(response or "").strip()
    boxed_values = extract_all_boxed(cleaned)
    final_boxed = boxed_values[-1] if boxed_values else ""
    if not final_boxed:
        final_boxed = str(fallback_answer).strip()

    body = _strip_boxed(cleaned).strip()
    if final_boxed:
        if body:
            return f"{body}\n\n\\boxed{{{final_boxed}}}"
        return f"\\boxed{{{final_boxed}}}"

    ensured = ensure_boxed(cleaned)
    value = extract_boxed(ensured).strip()
    body = _strip_boxed(ensured).strip()
    if body:
        return f"{body}\n\n\\boxed{{{value}}}"
    return f"\\boxed{{{value}}}"


def _normalize_mcq_answer(item: dict) -> str:
    options = item.get("options") or []
    labels = [chr(65 + i) for i in range(len(options))]
    if not labels:
        return ""

    answer_text = str(item.get("answer", "")).strip()
    if not answer_text:
        return ""

    letter = extract_valid_letter(answer_text, labels)
    if letter:
        return letter

    answer_lower = answer_text.lower()
    for lbl, opt in zip(labels, options):
        if answer_lower == str(opt).strip().lower():
            return lbl

    if len(answer_text) == 1 and answer_text.upper() in labels:
        return answer_text.upper()
    return ""


def _normalize_free_answer(item: dict) -> str:
    answer_value = item.get("answer")
    if answer_value is None:
        return ""
    if isinstance(answer_value, (list, tuple)):
        answer_text = ", ".join(str(v).strip() for v in answer_value)
    else:
        answer_text = str(answer_value).strip()
    if not answer_text:
        return ""
    return _enforce_single_final_boxed("", fallback_answer=answer_text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter public.jsonl into a clean LoRA training file. "
            "Run before retraining the adapter."
        ),
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Path to source JSONL. Default: <project_root>/data/public.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Destination JSONL. Default: "
            "<project_root>/data/public_clean_for_lora.jsonl."
        ),
    )
    parser.add_argument(
        "--only-base-fails",
        default=None,
        help=(
            "Optional path to a prior base outputs JSONL "
            "(e.g. results/private_base/public_outputs_ordered.jsonl). When "
            "set, only items whose `response` is judged wrong are kept. "
            "Requires answers in the source JSONL and the project judger."
        ),
    )
    parser.add_argument(
        "--mcq-only",
        action="store_true",
        help="Keep only MCQ items in the output.",
    )
    parser.add_argument(
        "--free-only",
        action="store_true",
        help="Keep only free-form items in the output.",
    )
    parser.add_argument(
        "--require-base-fails-judger-timeout",
        type=float,
        default=2.0,
        help="Per-item judger timeout when --only-base-fails is set.",
    )
    parser.add_argument(
        "--output-mcq-path",
        default=None,
        help=(
            "Also write MCQ-only cleaned items to this path. "
            "Compatible with --mcq-only/--free-only (writes the MCQ subset "
            "of whatever passes the main filter)."
        ),
    )
    parser.add_argument(
        "--output-free-path",
        default=None,
        help=(
            "Also write free-only cleaned items to this path. "
            "Compatible with --mcq-only/--free-only (writes the free subset "
            "of whatever passes the main filter)."
        ),
    )
    args = parser.parse_args()
    if args.mcq_only and args.free_only:
        parser.error("--mcq-only and --free-only are mutually exclusive.")
    return args


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _has_clean_target(item: dict) -> tuple[bool, str]:
    """Return (ok, reason) for whether the trainer can build a clean target.

    Mirrors the logic of train_lora._build_adapt_examples without
    instantiating the dataset.
    """
    is_mcq = bool(item.get("options"))
    answer = item.get("answer")
    if answer is None:
        return False, "no_answer"

    if is_mcq:
        letter = _normalize_mcq_answer(item)
        if not letter:
            return False, "mcq_answer_not_resolvable"
        target = f"\\boxed{{{letter}}}"
    else:
        target = _normalize_free_answer(item)
        if not target:
            return False, "free_answer_empty"

    target_boxes = extract_all_boxed(target)
    if len(target_boxes) == 0:
        return False, "target_has_no_box"
    if len(target_boxes) > 1:
        return False, "target_has_multiple_boxes"
    if not target_boxes[0].strip():
        return False, "target_box_empty"

    # Final sanity: confirm _enforce_single_final_boxed agrees.
    enforced = _enforce_single_final_boxed("", fallback_answer=target_boxes[0])
    if len(extract_all_boxed(enforced)) != 1:
        return False, "enforce_produced_unexpected_boxes"

    return True, ""


def _build_base_fail_index(
    base_outputs_path: Path,
    items_by_id: dict[Any, dict],
    *,
    timeout_s: float,
) -> set:
    """Return ids whose prior base response was judged wrong vs gold."""
    from evaluation import _load_project_judger, _safe_auto_judge

    JudgerCls = _load_project_judger()
    if JudgerCls is None:
        raise SystemExit(
            "Could not load project judger; --only-base-fails requires it."
        )
    judger = JudgerCls(strict_extract=False)

    fails: set = set()
    with open(base_outputs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("id")
            item = items_by_id.get(rid)
            if item is None:
                continue
            gold = item.get("answer")
            if gold is None:
                continue
            gold_list = gold if isinstance(gold, list) else [gold]
            options_per_slot = [item.get("options", []) or []] * len(gold_list)
            pred = rec.get("response", "")
            ok = _safe_auto_judge(
                judger,
                pred=pred,
                gold=gold_list,
                options_per_slot=options_per_slot,
                timeout_s=timeout_s,
            )
            if not ok:
                fails.add(rid)
    return fails


def main() -> None:
    args = _parse_args()

    here = Path(__file__).resolve().parent
    root = _discover_project_root(here)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    input_path = Path(args.input_path) if args.input_path else (root / "data" / "public.jsonl")
    output_path = (
        Path(args.output_path)
        if args.output_path
        else (root / "data" / "public_clean_for_lora.jsonl")
    )

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    items = _load_jsonl(input_path)
    print(f"Loaded {len(items)} items from {input_path}")

    base_fail_ids: set | None = None
    if args.only_base_fails:
        base_outputs_path = Path(args.only_base_fails)
        if not base_outputs_path.exists():
            raise SystemExit(f"base-outputs file not found: {base_outputs_path}")
        items_by_id = {it.get("id"): it for it in items}
        base_fail_ids = _build_base_fail_index(
            base_outputs_path,
            items_by_id,
            timeout_s=args.require_base_fails_judger_timeout,
        )
        print(
            f"Loaded {len(base_fail_ids)} base-fail item ids from "
            f"{base_outputs_path}"
        )

    kept: list[dict] = []
    drop_reasons: dict[str, int] = {}
    skipped_by_type = 0
    skipped_base_passed = 0

    for item in items:
        is_mcq = bool(item.get("options"))
        if args.mcq_only and not is_mcq:
            skipped_by_type += 1
            continue
        if args.free_only and is_mcq:
            skipped_by_type += 1
            continue

        if base_fail_ids is not None and item.get("id") not in base_fail_ids:
            skipped_base_passed += 1
            continue

        ok, reason = _has_clean_target(item)
        if not ok:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            continue

        # Additional MCQ sanity: gold letter must be within the option count.
        if is_mcq:
            options = item.get("options") or []
            labels = [chr(65 + i) for i in range(len(options))]
            answer_text = str(item.get("answer", "")).strip()
            if not extract_valid_letter(answer_text, labels):
                if not _normalize_mcq_answer(item):
                    drop_reasons["mcq_gold_letter_out_of_range"] = (
                        drop_reasons.get("mcq_gold_letter_out_of_range", 0) + 1
                    )
                    continue

        kept.append(item)

    _write_jsonl(output_path, kept)

    mcq_kept = [it for it in kept if it.get("options")]
    free_kept = [it for it in kept if not it.get("options")]

    if args.output_mcq_path:
        mcq_path = Path(args.output_mcq_path)
        _write_jsonl(mcq_path, mcq_kept)
    if args.output_free_path:
        free_path = Path(args.output_free_path)
        _write_jsonl(free_path, free_kept)

    n_total = len(items)
    n_kept = len(kept)
    n_dropped = n_total - n_kept - skipped_by_type - skipped_base_passed

    print("=" * 60)
    print(f"Source:      {input_path} (n={n_total})")
    print(f"Output:      {output_path} (n={n_kept})")
    print(f"  MCQ:       {len(mcq_kept)}")
    print(f"  Free-form: {len(free_kept)}")
    if args.output_mcq_path:
        print(f"MCQ split:   {args.output_mcq_path} (n={len(mcq_kept)})")
    if args.output_free_path:
        print(f"Free split:  {args.output_free_path} (n={len(free_kept)})")
    print(f"Type filter: skipped {skipped_by_type} items by --mcq-only/--free-only")
    if base_fail_ids is not None:
        print(
            f"Base filter: skipped {skipped_base_passed} items base already solved"
        )
    print(f"Quality:     dropped {n_dropped} items by clean-target filters:")
    for reason, count in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  - {reason}: {count}")
    print("=" * 60)
    print(
        f"Done. Feed this into train_lora.py with `--input {output_path.name}` "
        "after copying it under data/, or with a full path."
    )


if __name__ == "__main__":
    main()
