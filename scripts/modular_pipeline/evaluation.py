"""Judger-based evaluation helpers."""

import signal

from tqdm import tqdm


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


def evaluate_with_judger(data: list[dict], records_by_id: dict) -> None:
    """Score predictions against gold answers using judger.Judger.auto_judge."""
    try:
        from judger import Judger
    except Exception as exc:
        print(f"Could not import Judger for evaluation: {exc}")
        return

    try:
        judger = Judger(strict_extract=False)
    except Exception as exc:
        print(f"Could not initialize Judger for evaluation: {exc}")
        return

    mcq_total = mcq_correct = 0
    free_total = free_correct = 0

    for item in tqdm(data, desc="Scoring with Judger"):
        answer = item.get("answer")
        if answer is None:
            continue

        rec = records_by_id.get(item.get("id"))
        if rec is None:
            continue

        pred = rec.get("response", "")
        gold = answer if isinstance(answer, list) else [answer]
        options_per_slot = [item.get("options", [])] * len(gold)

        ok = _safe_auto_judge(
            judger,
            pred=pred,
            gold=gold,
            options_per_slot=options_per_slot,
        )

        if item.get("options"):
            mcq_total += 1
            mcq_correct += int(ok)
        else:
            free_total += 1
            free_correct += int(ok)

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
    print("=" * 50)
