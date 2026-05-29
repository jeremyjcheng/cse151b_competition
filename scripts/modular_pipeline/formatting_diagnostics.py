"""Per-output formatting diagnostics for the base-vs-LoRA gate.

Pure functions. No vLLM, no IO, no torch dependency. The compare runner
calls `score_output` on each side's raw text and canonical response and
passes the resulting dicts to `lora_gate.decide_deployable_winner`.

Outputs:

- ``mandatory_pass`` (bool): True iff none of the hard-fail rules tripped.
  A side with ``mandatory_pass=False`` is hard-rejected by the deploy gate.
- ``confidence_score`` (int): higher = more likely-good output. The deploy
  gate picks the higher score when both sides pass mandatory.

Hard fails (mandatory_pass=False):
  - empty_boxed: any ``\\boxed{}`` with empty inner value.
  - invalid_mcq_letter: MCQ response whose boxed inner does not resolve to
    a valid option letter.
  - hit_max_no_clean_box: generation hit the max token budget and there is
    no clean single final box in the response.
  - final_box_malformed: response has no final ``\\boxed{...}`` at all
    (free-form failure mode).
  - repeated_boxed_answers (>= 3 duplicates): training-echo failure mode
    where the model emits ``\\boxed{X}`` many times.
"""

from __future__ import annotations

import re
from typing import Any

from text_processing import (
    _mcq_letter_from_boxed_inner,
    extract_valid_letter,
    iter_boxed_spans,
)


_REPEAT_PHRASE_RE = re.compile(
    r"\bis\s+the\s+(?:correct|final|right)?\s*answer\b",
    re.IGNORECASE,
)


def _is_valid_mcq_letter(inner: str, labels: list[str]) -> bool:
    if not labels or not inner:
        return False
    valid_set_upper = {str(x).strip().upper() for x in labels}
    return bool(_mcq_letter_from_boxed_inner(inner.strip(), valid_set_upper))


def score_output(
    raw: str,
    response: str,
    *,
    is_mcq: bool,
    labels: list[str] | None = None,
    max_new_tokens: int = 0,
    n_tokens: int = 0,
    pre_trunc_n_tokens: int | None = None,
    generation_hit_max: bool | None = None,
) -> dict[str, Any]:
    """Score a single model output for the deploy gate.

    Parameters
    ----------
    raw : str
        Raw model output for diagnostics. Should be the post-truncation raw
        the pipeline saved (i.e. ``solve_*_batch`` -> record["raw"]).
    response : str
        Canonical response string actually saved (the single ``\\boxed{...}``
        emitted by ``mcq_canonical_response`` / ``canonicalize_free_response``).
    is_mcq : bool
        Whether the item is multiple choice.
    labels : list[str] | None
        Valid option letters (``["A","B","C",...]``) for MCQ items.
    max_new_tokens : int
        Token budget the model was given. Used together with
        ``pre_trunc_n_tokens`` / ``n_tokens`` as a fallback to detect
        ``generation_hit_max`` when the pipeline did not report it.
    n_tokens, pre_trunc_n_tokens : int | None
        From the pipeline meta. ``pre_trunc_n_tokens`` is preferred since
        ``n_tokens`` is post-truncation length.
    generation_hit_max : bool | None
        Pipeline-reported flag; when set this overrides any inference from
        token counts.
    """
    labels = list(labels or [])
    raw_text = str(raw or "")
    response_text = str(response or "")

    raw_spans = iter_boxed_spans(raw_text)
    response_spans = iter_boxed_spans(response_text)
    raw_inners = [inner.strip() for _s, _e, inner in raw_spans]
    response_inners = [inner.strip() for _s, _e, inner in response_spans]

    boxed_count_raw = len(raw_inners)
    boxed_count_response = len(response_inners)

    has_empty_box = any(not inner for inner in raw_inners) or any(
        not inner for inner in response_inners
    )

    has_exactly_one_final_box = (
        boxed_count_response == 1 and bool(response_inners[0])
    )

    mcq_boxed_letter_valid = False
    if is_mcq and labels and response_inners:
        mcq_boxed_letter_valid = _is_valid_mcq_letter(response_inners[-1], labels)
        # If the canonical response did not encode a valid letter, fall back
        # to checking the raw text for any valid letter (so we can still tell
        # downstream "LoRA produced a valid letter somewhere" vs "no letter").
        if not mcq_boxed_letter_valid and raw_text:
            mcq_boxed_letter_valid = bool(extract_valid_letter(raw_text, labels))

    counts: dict[str, int] = {}
    for inner in raw_inners:
        if not inner:
            continue
        counts[inner] = counts.get(inner, 0) + 1
    repeated_boxed_answers = sum(c - 1 for c in counts.values() if c >= 2)

    repeated_phrase_after_box = 0
    if raw_spans:
        # Count "is the correct answer" style phrases that appear AFTER the
        # first boxed span. This catches the LoRA failure mode where the
        # model emits a final answer and then keeps re-emitting it:
        #   \boxed{A} is the correct answer. \boxed{A} is the correct answer.
        first_box_end = raw_spans[0][1]
        tail = raw_text[first_box_end:]
        repeated_phrase_after_box = len(_REPEAT_PHRASE_RE.findall(tail))

    if generation_hit_max is not None:
        hit_max = bool(generation_hit_max)
    elif max_new_tokens > 0:
        effective_tokens = pre_trunc_n_tokens if pre_trunc_n_tokens is not None else n_tokens
        hit_max = bool(effective_tokens and effective_tokens >= max_new_tokens)
    else:
        hit_max = False

    hit_max_without_clean_box = hit_max and not has_exactly_one_final_box

    if is_mcq and labels:
        final_box_malformed = boxed_count_response == 0 or not mcq_boxed_letter_valid
    else:
        final_box_malformed = boxed_count_response == 0

    output_len_chars = len(raw_text)

    confidence = 0
    if has_exactly_one_final_box:
        confidence += 2
    if is_mcq and mcq_boxed_letter_valid:
        confidence += 2
    if not has_empty_box:
        confidence += 1
    if repeated_boxed_answers == 0:
        confidence += 1
    if repeated_phrase_after_box == 0:
        confidence += 1
    if not hit_max:
        confidence += 1
    if boxed_count_response == 1:
        confidence += 1

    mandatory_fail_reasons: list[str] = []
    if has_empty_box:
        mandatory_fail_reasons.append("empty_boxed")
    if is_mcq and labels and not mcq_boxed_letter_valid:
        mandatory_fail_reasons.append("invalid_mcq_letter")
    if hit_max_without_clean_box:
        mandatory_fail_reasons.append("hit_max_no_clean_box")
    if final_box_malformed and not (is_mcq and labels):
        if boxed_count_response == 0:
            mandatory_fail_reasons.append("final_box_malformed")
    if repeated_boxed_answers >= 3:
        mandatory_fail_reasons.append("repeated_boxed_answers")

    mandatory_pass = len(mandatory_fail_reasons) == 0

    return {
        "boxed_count_raw": int(boxed_count_raw),
        "boxed_count_response": int(boxed_count_response),
        "has_exactly_one_final_box": bool(has_exactly_one_final_box),
        "mcq_boxed_letter_valid": bool(mcq_boxed_letter_valid),
        "has_empty_box": bool(has_empty_box),
        "repeated_boxed_answers": int(repeated_boxed_answers),
        "repeated_phrase_after_box": int(repeated_phrase_after_box),
        "hit_max_without_clean_box": bool(hit_max_without_clean_box),
        "final_box_malformed": bool(final_box_malformed),
        "output_len_chars": int(output_len_chars),
        "confidence_score": int(confidence),
        "mandatory_pass": bool(mandatory_pass),
        "mandatory_fail_reasons": mandatory_fail_reasons,
    }
