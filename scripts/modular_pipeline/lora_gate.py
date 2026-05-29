"""Base-vs-LoRA routing logic.

Two strictly separated APIs:

- ``score_diagnostic_winner`` is GOLD-AWARE and PUBLIC-ONLY. Returns
  ``truth_winner`` using the project's judger. It has no
  ``deploy_response`` field and never reaches the submission CSV. Use it
  to understand whether LoRA *intrinsically* helps.

- ``decide_deployable_winner`` is GOLD-BLIND. Same logic on public and
  private. Returns the ``deploy_winner`` and ``deploy_response`` that
  the submission CSV uses. MCQ safety (re-extract first valid letter,
  fall back to base if none) is part of this function.

These are kept separate so a strong LoRA the deploy gate cannot
reliably select cannot accidentally be shipped.
"""

from __future__ import annotations

from typing import Any, Callable

from text_processing import (
    extract_first_valid_letter,
    mcq_canonical_response,
)


def score_diagnostic_winner(
    base_response: str,
    lora_response: str,
    *,
    gold: Any,
    options: list[str],
    judger: Any,
    safe_auto_judge: Callable[..., bool],
) -> dict[str, Any]:
    """Compute the gold-aware diagnostic winner. Public-only.

    Parameters
    ----------
    base_response, lora_response : str
        Canonical responses (already canonicalised to a single
        ``\\boxed{...}``).
    gold : list | str
        Gold answer(s). A scalar value is wrapped into a single-slot list
        to match how ``evaluation.py`` calls the judger.
    options : list[str]
        Options for the item; empty list for free-form items.
    judger : object
        Loaded ``Judger`` instance (e.g. ``Judger(strict_extract=False)``).
    safe_auto_judge : callable
        Wrapper around ``judger.auto_judge`` with a per-item timeout
        (typically ``evaluation._safe_auto_judge``).

    Returns
    -------
    dict
        ``{truth_winner, base_correct, lora_correct, reason}``.
        ``truth_winner`` is one of ``"base"``, ``"lora"``, ``"tie_correct"``,
        ``"tie_wrong"``.
    """
    if isinstance(gold, list):
        gold_list = list(gold)
    else:
        gold_list = [gold]

    options_per_slot = [list(options or [])] * len(gold_list)

    base_correct = bool(
        safe_auto_judge(
            judger,
            pred=base_response,
            gold=gold_list,
            options_per_slot=options_per_slot,
        )
    )
    lora_correct = bool(
        safe_auto_judge(
            judger,
            pred=lora_response,
            gold=gold_list,
            options_per_slot=options_per_slot,
        )
    )

    if base_correct and lora_correct:
        truth_winner = "tie_correct"
        reason = "both_correct"
    elif base_correct and not lora_correct:
        truth_winner = "base"
        reason = "base_correct_only"
    elif lora_correct and not base_correct:
        truth_winner = "lora"
        reason = "lora_correct_only"
    else:
        truth_winner = "tie_wrong"
        reason = "both_wrong"

    return {
        "truth_winner": truth_winner,
        "base_correct": base_correct,
        "lora_correct": lora_correct,
        "reason": reason,
    }


def decide_deployable_winner(
    base_response: str,
    lora_response: str,
    lora_raw: str,
    *,
    base_score: dict,
    lora_score: dict,
    is_mcq: bool,
    labels: list[str],
    strict: bool = False,
) -> dict[str, Any]:
    """Gold-blind deploy decision. Identical on public and private.

    Returns
    -------
    dict
        ``{deploy_winner, deploy_response, reason, deploy_meta}``.
        ``deploy_winner`` is one of ``"base"``, ``"lora"``. We never emit
        ``"tie"`` or ``"neither"`` so the submission CSV is always
        populated.
    """
    base_pass = bool(base_score.get("mandatory_pass"))
    lora_pass = bool(lora_score.get("mandatory_pass"))
    base_conf = int(base_score.get("confidence_score", 0))
    lora_conf = int(lora_score.get("confidence_score", 0))
    confidence_delta = lora_conf - base_conf

    if lora_pass and not base_pass:
        winner = "lora"
        reason = "base_failed_mandatory"
    elif base_pass and not lora_pass:
        winner = "base"
        reason = "lora_failed_mandatory"
    elif not base_pass and not lora_pass:
        winner = "base"
        reason = "both_failed_mandatory"
    else:
        if strict:
            if confidence_delta >= 2:
                winner = "lora"
                reason = "strict_confidence_delta_ge_2"
            else:
                winner = "base"
                reason = "strict_confidence_delta_lt_2"
        else:
            if lora_conf > base_conf:
                winner = "lora"
                reason = "higher_lora_confidence"
            else:
                winner = "base"
                reason = "tie_or_base_higher_confidence"

    deploy_response = lora_response if winner == "lora" else base_response

    downgraded = False
    downgrade_reason = ""

    if winner == "lora" and is_mcq:
        letter = extract_first_valid_letter(lora_raw, labels)
        if letter:
            deploy_response = mcq_canonical_response(letter)
        else:
            winner = "base"
            deploy_response = base_response
            downgraded = True
            downgrade_reason = "lora_no_valid_mcq_letter"
            reason = downgrade_reason

    deploy_meta = {
        "base_mandatory_pass": base_pass,
        "lora_mandatory_pass": lora_pass,
        "base_confidence": base_conf,
        "lora_confidence": lora_conf,
        "confidence_delta": confidence_delta,
        "strict": bool(strict),
        "downgraded_lora_to_base": bool(downgraded),
        "downgrade_reason": downgrade_reason,
    }

    return {
        "deploy_winner": winner,
        "deploy_response": deploy_response,
        "reason": reason,
        "deploy_meta": deploy_meta,
    }
