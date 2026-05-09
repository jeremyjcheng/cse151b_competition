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


def clean_special_tokens(text: str) -> str:
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    return text.strip()


def extract_valid_letter(text: str, labels: list[str]) -> str:
    valid_set = set(labels)
    upper = text.upper()

    boxed = extract_boxed(upper).strip().upper()
    if boxed in valid_set:
        return boxed

    patterns = [
        r"\\BOXED\{\s*([A-Z])\s*\}",
        r"OPTION\s+([A-Z])",
        r"CHOICE\s+([A-Z])",
        r"ANSWER\s+IS\s+([A-Z])",
        r"FINAL\s+ANSWER\s+IS\s+([A-Z])",
        r"CORRESPONDS\s+TO\s+OPTION\s+([A-Z])",
        r"MATCH(?:ES)?\s+OPTION\s+([A-Z])",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, upper)
        for match in reversed(matches):
            if match in valid_set:
                return match

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


def canonicalize_free_response(response: str) -> str:
    """Preserve rationale but force exactly one final boxed answer."""
    ensured = ensure_boxed(response)
    final_value = extract_boxed(ensured).strip()
    spans = _find_boxed_spans(ensured)
    if spans:
        chunks: list[str] = []
        cursor = 0
        for start, end in spans:
            chunks.append(ensured[cursor:start])
            cursor = end
        chunks.append(ensured[cursor:])
        body = "".join(chunks).rstrip()
    else:
        body = ensured.rstrip()
    if body:
        return body + f"\n\n\\boxed{{{final_value}}}"
    return f"\\boxed{{{final_value}}}"
