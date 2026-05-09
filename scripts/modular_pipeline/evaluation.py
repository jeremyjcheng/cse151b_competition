"""Judger-based evaluation helpers."""

import importlib.util
import signal
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from text_processing import extract_all_boxed, extract_valid_letter


class _JudgeTimeout(Exception):
    """Raised when a single judger call exceeds the time budget."""


def _alarm_handler(signum, frame):
    del signum, frame
    raise _JudgeTimeout()


def _safe_auto_judge(judger, pred: str, gold: list, options_per_slot: list, timeout_s: float = 2.0) -> bool:
    """Run `judger.auto_judge` with a per-item timeout and safe fallback."""
    if "\\boxed" not in pred:
        # Most malformed outputs without a final boxed answer are low quality and
        # can trigger expensive parsing paths in symbolic equivalence checks.
        return False

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        return bool(judger.auto_judge(pred=pred, gold=gold, options=options_per_slot))
    except (_JudgeTimeout, Exception):
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _load_project_judger() -> Optional[type]:
    """Load `Judger` from this project's `judger.py` explicitly by path."""
    judger_path = Path(__file__).resolve().parents[2] / "judger.py"
    if not judger_path.exists():
        print(f"Could not locate project judger at: {judger_path}")
        return None

    try:
        spec = importlib.util.spec_from_file_location("project_judger", judger_path)
        if spec is None or spec.loader is None:
            print(f"Could not create import spec for: {judger_path}")
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "Judger", None)
    except Exception as exc:
        print(f"Could not import project judger from {judger_path}: {exc}")
        return None


def evaluate_with_judger(data: list[dict], records_by_id: dict) -> None:
    """Score predictions against gold answers using judger.Judger.auto_judge."""
    Judger = _load_project_judger()
    if Judger is None:
        print("Could not import Judger for evaluation.")
        return

    try:
        judger = Judger(strict_extract=False)
    except Exception as exc:
        print(f"Could not initialize Judger for evaluation: {exc}")
        return

    mcq_total = mcq_correct = 0
    free_total = free_correct = 0
    malformed_missing_box = 0
    malformed_multi_box = 0
    malformed_invalid_mcq = 0
    format_valid_total = 0

    for item in tqdm(data, desc="Scoring with Judger"):
        answer = item.get("answer")
        if answer is None:
            continue

        rec = records_by_id.get(item.get("id"))
        if rec is None:
            continue

        pred = rec.get("response", "")
        boxed_values = extract_all_boxed(pred)
        if not boxed_values:
            malformed_missing_box += 1
        if len(boxed_values) > 1:
            malformed_multi_box += 1

        gold = answer if isinstance(answer, list) else [answer]
        options_per_slot = [item.get("options", [])] * len(gold)

        is_mcq = bool(item.get("options"))
        labels = [chr(65 + i) for i in range(len(item.get("options", [])))] if is_mcq else []
        if is_mcq and boxed_values:
            if labels and not extract_valid_letter(pred, labels):
                malformed_invalid_mcq += 1

        ok = _safe_auto_judge(
            judger,
            pred=pred,
            gold=gold,
            options_per_slot=options_per_slot,
        )

        if is_mcq:
            mcq_total += 1
            mcq_correct += int(ok)
        else:
            free_total += 1
            free_correct += int(ok)

        format_valid_total += int(
            bool(boxed_values)
            and len(boxed_values) == 1
            and (not is_mcq or not labels or bool(extract_valid_letter(pred, labels)))
        )

    overall_total = mcq_total + free_total
    overall_correct = mcq_correct + free_correct

    def acc(correct: int, total: int) -> float:
        return (correct / total * 100.0) if total else 0.0

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {mcq_correct:4d} / {mcq_total:4d}  ({acc(mcq_correct, mcq_total):.2f}%)")
    print(f"  Free-form  : {free_correct:4d} / {free_total:4d}  ({acc(free_correct, free_total):.2f}%)")
    print(f"  Overall    : {overall_correct:4d} / {overall_total:4d}  ({acc(overall_correct, overall_total):.2f}%)")
    print(
        f"  Format OK  : {format_valid_total:4d} / {overall_total:4d}  "
        f"({acc(format_valid_total, overall_total):.2f}%)"
    )
    print(
        f"  Malformed  : missing_box={malformed_missing_box}, "
        f"multiple_box={malformed_multi_box}, invalid_mcq_letter={malformed_invalid_mcq}"
    )
    print("=" * 50)
