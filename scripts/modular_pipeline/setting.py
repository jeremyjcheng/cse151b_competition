"""Configuration constants for the modular pipeline."""

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# LoRA defaults
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Training defaults
TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 8e-5
MAX_STEPS = 1000
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 2048
SAVE_EVERY_STEPS = 100

# Stage-specific train defaults
REASONING_DEFAULT_LEARNING_RATE = 8e-5
REASONING_DEFAULT_MAX_STEPS = 1000
ADAPT_DEFAULT_LEARNING_RATE = 5e-5
ADAPT_DEFAULT_MAX_STEPS = 150

# MCQ settings
MAX_TOKENS_MCQ = 1024
THINK_BUDGET_MCQ = 512
MAX_TOKENS_MCQ_FINAL = 96
TEMP_MCQ = 0.0
TOP_P_MCQ = 1.0
TOP_K_MCQ = 0
REP_PEN_MCQ = 1.18
REP_PEN_MCQ_FINAL = 1.05

# Generation: stop after boxed (final-answer-aware; see model_pipeline.SmartBoxedStopProcessor)
MIN_TOKENS_BEFORE_BOXED_STOP = 48
POST_BOX_PATIENCE_TOKENS_FREE = 64
POST_BOX_PATIENCE_TOKENS_MCQ = 0
FINAL_ANSWER_CUE_WINDOW_CHARS = 180

# Optional n-gram blocking (0 = disabled for HF generate)
NO_REPEAT_NGRAM_SIZE_MCQ = 4
NO_REPEAT_NGRAM_SIZE_MCQ_FINAL = 3
NO_REPEAT_NGRAM_SIZE_FREE = 4

# Free-form settings
MAX_TOKENS_FREE = 2048
THINK_BUDGET_FREE = 1024
TEMP_FREE = 0.1
TOP_P_FREE = 0.9
TOP_K_FREE = 10
REP_PEN_FREE = 1.18

MCQ_BATCH_SIZE = 8
FREE_BATCH_SIZE = 2

SYSTEM_PROMPT_MCQ = (
    "Solve the multiple-choice math problem. "
    "Reason briefly and efficiently. "
    "Compute the result first. "
    "Compare your result to the options. "
    "Choose the closest valid option. "
    "Output exactly one final answer in the form \\boxed{X}. "
    "Do not output multiple boxed answers. "
    "Do not repeat yourself. "
    "Stop immediately after writing the final boxed answer."
)

SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician solving a timed exam. "
    "Reason briefly and efficiently. "
    "Avoid repeating the same step. "
    "End with exactly one final \\boxed{...}. "
    "Do not box intermediate answers. "
    "If multiple [ANS] slots exist, output exactly that many values "
    "inside a single \\boxed{...}, comma-separated. "
    "Stop immediately after the final boxed answer."
)

MCQ_FEWSHOT = (
    "Example:\n"
    "Q: What is 2+3?\n"
    "A. 4\n"
    "B. 5\n"
    "C. 6\n"
    "D. 7\n\n"
    "2+3=5, so the answer is \\boxed{B}.\n\n"
)