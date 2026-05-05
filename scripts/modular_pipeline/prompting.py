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
        "Compute the answer first, compare it to the options, "
        "then output the final answer as \\boxed{X}."
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
