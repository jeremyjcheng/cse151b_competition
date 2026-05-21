"""Prompt construction helpers."""



def build_mcq_user(question: str, options: list[str]) -> str:
    """Inference MCQ prompt aligned with adapt-training format."""
    return build_adapt_train_mcq_user(question, options, with_reasoning=True)


def count_ans_slots(question: str) -> int:
    return question.count("[ANS]")


def build_free_user(question: str) -> str:
    n_slots = count_ans_slots(question)
    if n_slots >= 2:
        return (
            f"{question}\n\n"
            f"This question has {n_slots} answer slots. "
            f"Output exactly {n_slots} values, in order and comma-separated, "
            f"inside one final \\boxed{{...}}."
        )
    return question


def build_reasoning_train_user(problem: str) -> str:
    return (
        f"{problem}\n\n"
        "Show concise step-by-step reasoning, then provide exactly one final "
        "\\boxed{...} answer."
    )


def build_adapt_train_mcq_user(question: str, options: list[str], *, with_reasoning: bool = True) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    valid_letters = ", ".join(labels)
    opts_text = "\n".join(f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options))
    if with_reasoning:
        reasoning_line = (
            "Show brief step-by-step reasoning (setup, key calculation, compare to options), "
            "then end with exactly one final boxed letter like \\boxed{A}. "
        )
    else:
        reasoning_line = (
            "Output exactly one final boxed letter like \\boxed{A}. "
        )
    return (
        f"Q: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"Valid choices: [{valid_letters}].\n"
        f"{reasoning_line}"
        "Do not output full option text and do not output multiple boxed answers."
    )


def build_mcq_reasoning_target(letter: str) -> str:
    """Stage-2 MCQ label: scaffolded reasoning + single \\boxed{letter} (not letter alone)."""
    ch = str(letter).strip().upper()
    return (
        "Work through the problem step by step.\n"
        "Compare your result to each option and eliminate choices that do not match.\n"
        f"Therefore the correct choice is \\boxed{{{ch}}}."
    )


def build_adapt_train_free_user(question: str) -> str:
    n_slots = count_ans_slots(question)
    if n_slots >= 2:
        return (
            f"{question}\n\n"
            f"This question has {n_slots} answer slots. "
            f"Output exactly {n_slots} values, in order and comma-separated, "
            f"inside one final \\boxed{{...}}."
        )
    return (
        f"{question}\n\n"
        "Output exactly one final \\boxed{...} answer and do not include more than one box."
    )
