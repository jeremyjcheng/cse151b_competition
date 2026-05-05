"""Configuration constants for the modular pipeline."""

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# MCQ settings
MAX_TOKENS_MCQ = 1024
THINK_BUDGET_MCQ = 512
MAX_TOKENS_MCQ_FINAL = 256
TEMP_MCQ = 0.0
TOP_P_MCQ = 1.0
TOP_K_MCQ = 0
REP_PEN_MCQ = 1.10

# Free-form settings
MAX_TOKENS_FREE = 768
TEMP_FREE = 0.1
TOP_P_FREE = 0.9
TOP_K_FREE = 10
REP_PEN_FREE = 1.10
THINK_BUDGET_FREE = 384

MCQ_BATCH_SIZE = 8
FREE_BATCH_SIZE = 2

SYSTEM_PROMPT_MCQ = (
    "Solve the multiple-choice math problem. "
    "Compute the result first. "
    "Compare your result to each option. "
    "Choose the closest matching option. "
    "Output exactly one letter from the listed valid choices; never output "
    "the option text itself. "
    "End with exactly one final answer in the form \\boxed{X}."
)

SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician solving a timed exam. "
    "Solve step by step, but keep the solution concise. "
    "End with exactly one final \\boxed{...}. Do not box intermediate sub-answers. "
    "If the question contains multiple [ANS] slots, output exactly that many "
    "values, in order and comma-separated, inside that single \\boxed{...}, "
    "e.g. \\boxed{3, 7}. "
    "Stop once the final boxed answer is written."
)

MCQ_FEWSHOT = (
    "Example:\n"
    "Q: What is 2+3?\n"
    "A. 4\nB. 5\nC. 6\nD. 7\n\n"
    "Compute: 2+3=5.\n"
    "Compare: option B is 5.\n"
    "\\boxed{B}\n\n"
)
