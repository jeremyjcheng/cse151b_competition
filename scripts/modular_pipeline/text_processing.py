"""Answer extraction and normalization utilities."""

import re


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
