"""Prompt construction helpers."""

from settings import MCQ_FEWSHOT


def build_mcq_user(question: str, options: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    valid_letters = ", ".join(labels)
    opts_text = "\n".join(f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options))

    return (
        f"{MCQ_FEWSHOT}"
        f"Q: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"Valid choices: [{valid_letters}].\n"
        "Choose the single best option and output exactly one final line as \\boxed{X}, "
        "where X is one valid choice letter. If uncertain, output your best guess from "
        "valid choices. Do not include explanation text, and do not output multiple boxed answers."
    )


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


def build_adapt_train_mcq_user(question: str, options: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    valid_letters = ", ".join(labels)
    opts_text = "\n".join(f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options))
    return (
        f"Q: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"Valid choices: [{valid_letters}].\n"
        "Output exactly one final boxed letter like \\boxed{A}. "
        "Do not output option text and do not output multiple boxed answers."
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
