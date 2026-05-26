"""Configuration constants for the modular pipeline."""

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# ============================================================
# vLLM engine defaults
# ============================================================

VLLM_GPU_MEMORY_UTILIZATION = 0.70
VLLM_MAX_MODEL_LEN = 4096
VLLM_MAX_NUM_SEQS = 4
VLLM_MAX_NUM_BATCHED_TOKENS = 4096
VLLM_QUANTIZATION = "bitsandbytes"
VLLM_LOAD_FORMAT = "bitsandbytes"

# Avoid CUDA graph capture OOM during sanity checks / small inference runs.
VLLM_ENFORCE_EAGER = False

# vLLM LoRA: max_loras caps concurrent adapters; max_lora_rank is set after LORA_R below.
VLLM_MAX_LORAS = 1

# Qwen3 dense LoRA in vLLM improved across 0.10+; upgrade if logs show Transformers fallback.
VLLM_MIN_VERSION = "0.10.0"

# ============================================================
# LoRA defaults
# ============================================================

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

# Match training rank unless adapter_config.json specifies a higher rank.
VLLM_MAX_LORA_RANK = LORA_R


# ============================================================
# Training defaults
# ============================================================

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

# Stage-2 adaptation should remain conservative:
# keep updates small so the model learns output format without overriding Stage-1 reasoning.
ADAPT_DEFAULT_LEARNING_RATE = 1e-5
ADAPT_DEFAULT_MAX_STEPS = 60

# Reserve this fraction of public supervised items for eval only (never used in Stage-2 training).
# Public is your pre-submit dev set — do not train on all of it if you score on it before private.
STAGE2_DEFAULT_HOLDOUT_FRACTION = 0.3

# Cap Stage-2 training size on public data (in addition to holdout split).
# 0 = no cap (use full train slice after holdout). Old defaults 50/25 severely limited SFT.
STAGE2_TRAIN_LIMIT_MCQ = 0
STAGE2_TRAIN_LIMIT_FREE = 0

# Stage-2 MCQ: supervise brief reasoning + final letter (not bare \\boxed{A} only).
STAGE2_MCQ_WITH_REASONING = True

# Training load: False = bfloat16 full weights (avoids bitsandbytes 4-bit / libnvJitLink issues).
TRAIN_LOAD_IN_4BIT = False

# Stage-2 sequences are shorter; lower cap reduces OOM risk during bf16 training.
ADAPT_DEFAULT_MAX_SEQ_LEN = 1024


# ============================================================
# MCQ generation settings
# ============================================================

# Shorter MCQ budget: deterministic \\boxed{X} answers (less truncation risk, faster).
MAX_TOKENS_MCQ = 1024

# Ignored by vLLM backend, kept for compatibility with older HF path.
THINK_BUDGET_MCQ = 0

# Finalizer only emits one box; a bit of headroom avoids cut-off \\boxed{}.
MAX_TOKENS_MCQ_FINAL = 512

# Greedy MCQ decoding.
TEMP_MCQ = 0.0
TOP_P_MCQ = 1.0
TOP_K_MCQ = 0

# Keep repetition penalties neutral for MCQ. Higher values were likely hurting output.
REP_PEN_MCQ = 1.0
REP_PEN_MCQ_FINAL = 1.0


# ============================================================
# Stop / truncation settings
# ============================================================

MIN_TOKENS_BEFORE_BOXED_STOP = 64

POST_BOX_PATIENCE_TOKENS_FREE = 256
POST_BOX_PATIENCE_TOKENS_MCQ = 0

FINAL_ANSWER_CUE_WINDOW_CHARS = 600


# ============================================================
# Optional n-gram blocking
# ============================================================

# Disabled because it can interfere with math reasoning and option comparison.
NO_REPEAT_NGRAM_SIZE_MCQ = 0
NO_REPEAT_NGRAM_SIZE_MCQ_FINAL = 0
NO_REPEAT_NGRAM_SIZE_FREE = 0


# ============================================================
# Free-form generation settings
# ============================================================

MAX_TOKENS_FREE = 8192

# Ignored by vLLM backend, kept for compatibility with older HF path.
THINK_BUDGET_FREE = 0

# Keep free-form mostly stable because it was much better than MCQ.
TEMP_FREE = 0.1
TOP_P_FREE = 0.9
TOP_K_FREE = 10
REP_PEN_FREE = 1.05


# ============================================================
# Batch sizes
# ============================================================

MCQ_BATCH_SIZE = 12
FREE_BATCH_SIZE = 2

# MCQ self-consistency: 0 = disabled; 3 = majority vote over 3 samples (Phase 4).
MCQ_SELF_CONSISTENCY_SAMPLES = 0
MCQ_SELF_CONSISTENCY_TEMP = 0.3

# ============================================================
# Thinking mode toggles
# ============================================================

ENABLE_THINKING_MCQ_PRIMARY = False
ENABLE_THINKING_MCQ_FINAL = False
ENABLE_THINKING_FREE = True

# ============================================================
# Prompts
# ============================================================

SYSTEM_PROMPT_MCQ = (
    "You solve multiple-choice math problems. "
    "Reason step by step: set up the problem, eliminate wrong options when possible, "
    "then state the final choice. "
    "The last line must be exactly one letter from the listed choices as \\boxed{X}. "
    "Do not box numbers or full option text. "
    "After the final \\boxed{...}, stop."
)

SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician solving a timed exam. "
    "Reason briefly and efficiently. "
    "Avoid repeating the same step. "
    "End with exactly one final \\boxed{...}. "
    "Do not box intermediate answers. "
    "If multiple [ANS] slots exist, output exactly that many values "
    "inside a single \\boxed{...}, comma-separated. "
    "After the final boxed answer, stop."
)

MCQ_FEWSHOT = ""