"""Answer extraction and normalization utilities."""

import re


def _find_boxed_spans(text: str) -> list[tuple[int, int]]:
    """Return [start, end) spans for every brace-balanced \\boxed{...}."""
    spans: list[tuple[int, int]] = []
    i = 0
    while True:
        start = text.find("\\boxed{", i)
        if start < 0:
            break
        brace_start = start + len("\\boxed{")
        depth = 1
        j = brace_start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth == 0:
            spans.append((start, j))
            i = j
        else:
            break
    return spans


def iter_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """Brace-balanced \\boxed{{...}} spans: (start, end, inner_text)."""
    return _find_boxed_with_values(text)


def _find_boxed_with_values(text: str) -> list[tuple[int, int, str]]:
    """Return [start, end, inner_value] for every complete \\boxed{...}."""
    out: list[tuple[int, int, str]] = []
    i = 0
    while True:
        start = text.find("\\boxed{", i)
        if start < 0:
            break
        brace_start = start + len("\\boxed{")
        depth = 1
        j = brace_start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth == 0:
            out.append((start, j, text[brace_start : j - 1].strip()))
            i = j
        else:
            break
    return out


def has_complete_boxed(text: str) -> bool:
    return bool(_find_boxed_spans(text))


def extract_all_boxed(text: str) -> list[str]:
    """Brace-balanced extraction of every \\boxed{...} occurrence."""
    out: list[str] = []
    i = 0
    while True:
        idx = text.find("\\boxed{", i)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        j = brace_start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth == 0:
            out.append(text[brace_start : j - 1].strip())
            i = j
        else:
            break
    return out


def extract_boxed(text: str) -> str:
    matches = extract_all_boxed(text)
    return matches[-1] if matches else ""


def visible_answer_after_think_tags(text: str) -> str:
    """Return text after the last explicit think/reasoning closing delimiter, if any.

    Some templates wrap chain-of-thought; the final \\boxed{{...}} is often after
    the closing tag, so phrase regexes work better on that slice.
    """
    if not text:
        return text
    markers = (
        "</think>",
        "</redacted_reasoning>",
    )
    best_pos = -1
    best_len = 0
    for m in markers:
        pos = text.rfind(m)
        if pos > best_pos:
            best_pos = pos
            best_len = len(m)
        elif pos == best_pos and pos >= 0 and len(m) > best_len:
            best_len = len(m)
    if best_pos < 0:
        return text
    return text[best_pos + best_len :].strip()


def clean_special_tokens(text: str) -> str:
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    return text.strip()


_LATEX_SINGLE_LETTER_WRAP = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit|textbf|textit|emph|mbox)\s*\{\s*([A-Za-z])\s*\}",
    re.IGNORECASE,
)


def _mcq_letter_from_boxed_inner(inner: str, valid_set_upper: set[str]) -> str:
    """Map \\boxed{{inner}} to one option letter (handles \\text{{J}}, nested wrappers)."""
    if not inner or not valid_set_upper:
        return ""

    s = inner.strip().strip(".$)'\"")
    for _ in range(8):
        if len(s) == 1 and s.upper() in valid_set_upper:
            return s.upper()
        m = _LATEX_SINGLE_LETTER_WRAP.search(s)
        if not m:
            break
        s = m.group(1).strip().strip(".$)'\"")
    if len(s) == 1 and s.upper() in valid_set_upper:
        return s.upper()
    return ""


def extract_valid_letter(text: str, labels: list[str]) -> str:
    valid_set_upper = {str(x).strip().upper() for x in labels}
    if not valid_set_upper:
        return ""

    # Last *valid* letter among all \\boxed{{...}}, not only the last box (avoids \\boxed{{}} tail).
    for _start, _end, inner in reversed(iter_boxed_spans(text)):
        cand = _mcq_letter_from_boxed_inner(inner, valid_set_upper)
        if cand:
            return cand

    upper = text.upper()

    patterns = [
        r"\\BOXED\{\s*([A-Z])\s*\}",
        r"OPTION\s+([A-Z])",
        r"CHOICE\s+([A-Z])",
        r"CORRECT\s+ANSWER\s+IS\s+([A-Z])",
        r"CORRECT\s+CHOICE\s+IS\s+([A-Z])",
        r"THE\s+ANSWER\s+IS\s+([A-Z])",
        r"ANSWER\s+IS\s+(?:OPTION\s+)?([A-Z])",
        r"CHOICE\s+IS\s+([A-Z])",
        r"OPTION\s+IS\s+([A-Z])",
        r"FINAL\s+ANSWER\s+IS\s+([A-Z])",
        r"CORRESPONDS\s+TO\s+OPTION\s+([A-Z])",
        r"MATCH(?:ES)?\s+OPTION\s+([A-Z])",
        r"(?:SELECT|PICK|CHOOSE)\s+(?:OPTION\s+)?([A-Z])\b",
        r"\bTHEREFORE[,:]?\s+(?:OPTION\s+)?([A-Z])\b",
        r"\bHENCE[,:]?\s+(?:OPTION\s+)?([A-Z])\b",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, upper)
        for match in reversed(matches):
            if match in valid_set_upper:
                return match

    return ""


def extract_mcq_letter_from_option_phrases(text: str, labels: list[str]) -> str:
    """Extract a valid MCQ letter from explicit option/answer phrase patterns.

    Targets recovery cases where output was truncated but still includes clues
    like:
      - "option B is ..."
      - "option C matches ..."
      - "the answer is D"
      - "choice A"
      - "A is correct"
    """
    valid_set_upper = {str(x).strip().upper() for x in labels}
    if not text or not valid_set_upper:
        return ""

    upper = text.upper()
    patterns = [
        r"\bOPTION\s+([A-Z])\s+IS\b",
        r"\bOPTION\s+([A-Z])\s+MATCH(?:ES|ED|ING)?\b",
        r"\bOPTION\s*[:=-]?\s*([A-Z])\b",
        r"\bTHE\s+ANSWER\s+IS\s+([A-Z])\b",
        r"\bANSWER\s+IS\s+([A-Z])\b",
        r"\bFINAL\s+ANSWER\s*(?:IS|=|:)\s*([A-Z])\b",
        r"\bCHOICE\s+([A-Z])\b",
        r"\bCHOOSE\s+([A-Z])\b",
        r"\bPICK\s+([A-Z])\b",
        r"\b([A-Z])\s+IS\s+CORRECT\b",
        r"\b([A-Z])\s+IS\s+THE\s+CORRECT\s+(?:ANSWER|CHOICE|OPTION)\b",
    ]

    best_pos = -1
    best_letter = ""
    for pattern in patterns:
        for m in re.finditer(pattern, upper):
            cand = m.group(1).strip().upper()
            if cand in valid_set_upper and m.start() >= best_pos:
                best_pos = m.start()
                best_letter = cand

    return best_letter


def extract_first_valid_letter(text: str, labels: list[str]) -> str:
    """Extract the first valid MCQ letter, prioritizing boxed options."""
    valid_set_upper = {str(x).strip().upper() for x in labels}
    if not valid_set_upper:
        return ""

    for _, _, boxed_value in _find_boxed_with_values(text):
        cand = _mcq_letter_from_boxed_inner(boxed_value, valid_set_upper)
        if cand:
            return cand

    upper = text.upper()

    patterns = [
        r"\\BOXED\{\s*([A-Z])\s*\}",
        r"OPTION\s+([A-Z])",
        r"CHOICE\s+([A-Z])",
        r"CORRECT\s+ANSWER\s+IS\s+([A-Z])",
        r"CORRECT\s+CHOICE\s+IS\s+([A-Z])",
        r"THE\s+ANSWER\s+IS\s+([A-Z])",
        r"ANSWER\s+IS\s+(?:OPTION\s+)?([A-Z])",
        r"CHOICE\s+IS\s+([A-Z])",
        r"OPTION\s+IS\s+([A-Z])",
        r"FINAL\s+ANSWER\s+IS\s+([A-Z])",
        r"CORRESPONDS\s+TO\s+OPTION\s+([A-Z])",
        r"MATCH(?:ES)?\s+OPTION\s+([A-Z])",
        r"(?:SELECT|PICK|CHOOSE)\s+(?:OPTION\s+)?([A-Z])\b",
        r"\bTHEREFORE[,:]?\s+(?:OPTION\s+)?([A-Z])\b",
        r"\bHENCE[,:]?\s+(?:OPTION\s+)?([A-Z])\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, upper):
            if match in valid_set_upper:
                return match
    return ""


def extract_tail_mcq_letter(text: str, labels: list[str]) -> str:
    """Extract a likely final MCQ letter from answer-style phrases near the end."""
    valid_set_upper = {str(x).strip().upper() for x in labels}
    if not valid_set_upper:
        return ""

    tail = text[-2000:]
    upper = tail.upper()

    patterns = [
        r"(?:FINAL\s+ANSWER|ANSWER|ANS)\s*(?:IS|=|:)?\s*[\(\[]?\s*([A-Z])\s*[\)\]]?",
        r"(?:OPTION|CHOICE)\s*(?:IS|=|:)?\s*[\(\[]?\s*([A-Z])\s*[\)\]]?",
        r"(?:I\s+CHOOSE|MY\s+CHOICE\s+IS)\s*[\(\[]?\s*([A-Z])\s*[\)\]]?",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, upper, flags=re.IGNORECASE)
        for match in reversed(matches):
            ch = str(match).strip().upper()
            if ch in valid_set_upper:
                return ch

    # Last resort in tail: standalone single-letter answer line.
    line_matches = re.findall(r"^\s*[\(\[]?\s*([A-Z])\s*[\)\]]?\s*$", upper, flags=re.MULTILINE)
    for match in reversed(line_matches):
        ch = str(match).strip().upper()
        if ch in valid_set_upper:
            return ch

    return ""


_LATEX_BLOCK_PATTERNS = [
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\$(.+?)\$", re.DOTALL),
]


def _last_latex_block(text: str) -> str:
    for pat in _LATEX_BLOCK_PATTERNS:
        matches = pat.findall(text)
        if matches:
            return matches[-1].strip()
    return ""


def _last_answer_phrase(text: str) -> str:
    for pat in (
        r"(?:final\s+)?answer\s+is[:\s]+([^\n\.]+)",
        r"(?:therefore|thus|so|hence)[,]?\s+([^\n\.]+)",
        r"=\s*([^\n=]+?)\s*$",
    ):
        matches = re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            value = matches[-1].strip().strip(".,:; \t")
            if value:
                return value

    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/-?\d+(?:\.\d+)?)?", text)
    return nums[-1] if nums else ""


def _count_ans_slots(question: str | None) -> int:
    if not question:
        return 0
    return str(question).count("[ANS]")


def _split_comma_values(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip().strip(".,:; \t") for p in str(text).split(",")]
    return [p for p in parts if p]


def _normalize_multi_answer_candidate(candidate: str) -> str:
    """Normalize phrase candidate before comma-splitting.

    Handles wrappers like ``\\[ ... \\]`` and prefers inner ``\\boxed{...}``
    content when present so we avoid emitting nested boxed responses.
    """
    s = str(candidate or "").strip().strip(".,:; \t")
    if not s:
        return ""

    # Strip common display-math wrappers.
    s = re.sub(r"^\\\[\s*", "", s)
    s = re.sub(r"\s*\\\]$", "", s)
    s = re.sub(r"^\$\$(.*)\$\$$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"^\$(.*)\$$", r"\1", s, flags=re.DOTALL)
    s = s.strip().strip(".,:; \t")

    # If candidate includes boxed math, prefer the last boxed inner content.
    boxed = _find_boxed_with_values(s)
    if boxed:
        s = boxed[-1][2].strip()

    return s.strip().strip(".,:; \t")


def _extract_multi_answer_values_from_phrase(text: str, expected_slots: int) -> list[str]:
    if expected_slots < 2 or not text:
        return []

    patterns = (
        r"(?:final\s+)?answers?\s*(?:are|is)[:\s]+([^\n]+)",
        r"(?:therefore|thus|hence|so)[,]?\s+([^\n]+)",
    )
    candidate = ""
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            candidate = str(matches[-1]).strip()
            break

    if not candidate:
        return []

    values = _split_comma_values(_normalize_multi_answer_candidate(candidate))
    if len(values) != expected_slots:
        return []
    return values


def canonicalize_mcq_response(response: str, letter: str) -> str:
    """Keep rationale text while enforcing exactly one final \\boxed{X}."""
    normalized_letter = str(letter).strip().upper()
    if not normalized_letter:
        return response

    spans = _find_boxed_spans(response)
    if not spans:
        body = response.rstrip()
    else:
        chunks: list[str] = []
        cursor = 0
        for start, end in spans:
            chunks.append(response[cursor:start])
            cursor = end
        chunks.append(response[cursor:])
        body = "".join(chunks).rstrip()

    if body:
        return body + f"\n\n\\boxed{{{normalized_letter}}}"
    return f"\\boxed{{{normalized_letter}}}"


def ensure_boxed(response: str) -> str:
    """Guarantee the response contains a final \\boxed{...} for the judger."""
    if extract_all_boxed(response):
        return response

    visible = response
    think_end = visible.rfind("</think>")
    if think_end >= 0:
        visible = visible[think_end + len("</think>") :].strip()
    if not visible:
        visible = response

    fallback = _last_latex_block(visible) or _last_answer_phrase(visible)
    if fallback:
        return response.rstrip() + f"\n\n\\boxed{{{fallback}}}"
    return response.rstrip() + "\n\n\\boxed{}"


def truncate_after_first_boxed(response: str) -> str:
    """Drop trailing text after first complete \\boxed{...} span."""
    spans = _find_boxed_spans(response)
    if not spans:
        return response
    first_end = spans[0][1]
    return response[:first_end].rstrip()


def truncate_after_last_boxed(response: str) -> str:
    """Drop trailing text after the last complete \\boxed{...} span."""
    spans = _find_boxed_spans(response)
    if not spans:
        return response
    last_end = spans[-1][1]
    return response[:last_end].rstrip()


def truncate_after_first_valid_mcq_box(text: str, labels: list[str]) -> str:
    """Trim ``text`` to end at the first ``\\boxed{...}`` whose inner value
    resolves to a valid option letter.

    Useful for the LoRA gate's MCQ safety rule: when a broken adapter emits
    ``\\boxed{A} is the correct answer. \\boxed{A} is the correct...`` we
    want diagnostics to look at the clean prefix up to the first valid
    boxed letter, not the polluted tail.

    Returns the original text unchanged if no boxed span maps to a valid
    option letter.
    """
    valid_set_upper = {str(x).strip().upper() for x in (labels or [])}
    if not valid_set_upper:
        return text
    for _start, end, inner in iter_boxed_spans(text):
        letter = _mcq_letter_from_boxed_inner(inner, valid_set_upper)
        if letter:
            return text[:end].rstrip()
    return text


_FINAL_ANSWER_CUE_PATTERN = re.compile(
    r"(?:final\s+answer|therefore|thus|hence)\b",
    re.IGNORECASE,
)


def canonicalize_free_response(response: str, question: str | None = None) -> str:
    """Return exactly one boxed free-response answer using cue-aware selection."""
    out, _meta = canonicalize_free_response_with_meta(response, question=question)
    return out


def canonicalize_free_response_with_meta(response: str, question: str | None = None) -> tuple[str, dict]:
    """Return canonical single \\boxed{...} plus extractor diagnostics for FRQ."""
    ensured = ensure_boxed(response)
    boxed = _find_boxed_with_values(ensured)
    boxed_inners = [b[2].strip() for b in boxed]
    expected_ans_slots = _count_ans_slots(question)
    meta: dict = {
        "extractor_path": "free",
        "boxed_count_in_raw": len(boxed),
        "boxed_candidates": boxed_inners,
        "selected_boxed_index": None,
        "expected_ans_slots": expected_ans_slots,
        "extracted_values": [],
        "phrase_override": False,
        "cue_matched": False,
        "fallback_used": False,
        "malformed_output": False,
        "malformed_reason": "",
    }

    if not boxed:
        meta["fallback_used"] = True
        meta["malformed_output"] = True
        meta["malformed_reason"] = "no_boxed_after_ensure"
        return ensure_boxed("\\boxed{}"), meta

    if expected_ans_slots >= 2:
        visible = visible_answer_after_think_tags(response)
        phrase_values = _extract_multi_answer_values_from_phrase(visible, expected_ans_slots)
        if phrase_values:
            meta["extractor_path"] = "free_multi_ans_phrase"
            meta["extracted_values"] = phrase_values
            meta["phrase_override"] = True
            return f"\\boxed{{{', '.join(phrase_values)}}}", meta

        boxed_values = _split_comma_values(boxed[-1][2].strip())
        if len(boxed_values) == expected_ans_slots:
            meta["extractor_path"] = "free_multi_ans_last_box"
            meta["selected_boxed_index"] = len(boxed) - 1
            meta["extracted_values"] = boxed_values
            return f"\\boxed{{{', '.join(boxed_values)}}}", meta

    cue_pattern = _FINAL_ANSWER_CUE_PATTERN
    selected_value = boxed[-1][2].strip()
    selected_index = len(boxed) - 1
    meta["selected_boxed_index"] = selected_index

    for actual_idx in range(len(boxed) - 1, -1, -1):
        start, _end, value = boxed[actual_idx]
        context_start = max(0, start - 180)
        context = ensured[context_start:start]
        if cue_pattern.search(context):
            selected_value = value.strip()
            selected_index = actual_idx
            meta["cue_matched"] = True
            meta["selected_boxed_index"] = selected_index
            break

    meta["extractor_path"] = "free_cue_last" if meta["cue_matched"] else "free_last_boxed"
    return f"\\boxed{{{selected_value}}}", meta


def mcq_canonical_response(letter: str) -> str:
    """Single MCQ submission line: exactly \\boxed{X}."""
    ch = str(letter).strip().upper()
    return f"\\boxed{{{ch}}}" if ch else "\\boxed{}"
