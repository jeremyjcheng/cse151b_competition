#!/usr/bin/env python
# coding: utf-8

# # CSE 151B Competition — Starter Notebook
# 
# Welcome to the **CSE 151B Spring 2026 Math Reasoning Competition**!  
# This notebook walks you through the full pipeline end-to-end:
# 
# 1. Setting up the Python environment with `uv`
# 2. Loading the competition dataset
# 3. Running inference with **Qwen3-4B-Thinking** via vLLM (INT8 quantized)
# 4. Scoring responses against ground-truth answers
# 5. Saving results to JSONL for submission
# 
# The public dataset (`public.jsonl`) contains questions **with** answers so you can measure accuracy locally.  
# The private test set used for the leaderboard does **not** include answers — for that, skip evaluation and submit the raw responses.

# ## 1. Environment Setup
# 
# We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible package management.
# 
# The steps below:
# 1. Install `uv` into `~/.local/bin`
# 2. Create a virtual environment at `.venv/`
# 3. Install all required packages (This might take a while)
# 
# > **After running this cell, restart the kernel** so that the newly installed packages (especially `vllm` and `transformers`) are picked up by the current Python session.

# ### Comment Out the cell below after first installation.

# In[1]:


# # Install uv
# !wget -qO- https://astral.sh/uv/install.sh | sh

# # Create a virtual environment
# !uv venv .venv --seed

# # Install dependencies — this is fast thanks to uv's parallel resolver
# !.venv/bin/python -m pip install sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter

# # Install Jupyter Kernel
# !.venv/bin/python -m ipykernel install --user --name cse151b --display-name "Python (cse151b)"

# print("Done. Restart the kernel before proceeding.")
# print("Selection process: on top right, click on current kernel '(ususally named python)' -> 'select another kernel' -> 'Jupyter Kernel' -> 'Python (cse151b)'.")


# ### Run the cell below every time to activate the installed environment. 

# In[2]:


# activate venv after installation. This needs to be run everytime.
#!source ./.venv/bin/activate


# ## 2. Imports & Configuration
# 
# All key settings are collected in one place.  
# - `DATA_PATH` — public dataset with ground-truth answers (use this to measure accuracy)
# - `OUTPUT_PATH` — where per-question results will be written
# - `GPU_ID` — which GPU to use (update if your machine has a different device index)
# - `MAX_TOKENS` — maximum tokens the model may generate per response

# In[3]:


import json, os, re, sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results/improved_results.jsonl"
SAVE_EVAL   = True  # set False for private test set

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# Token budgets per difficulty tier
DIFFICULTY_BUDGETS = {
    "easy":   {"max_tokens": 2048,  "think_cap": 1500},
    "medium": {"max_tokens": 8192,  "think_cap": 4000},
    "hard":   {"max_tokens": 32768, "think_cap": 8000},
}

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
print("Imports done.")



# import json
# import os

# # ── Configuration ─────────────────────────────────────────────────────────────
# MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
# GPU_ID      = "0"#"1"                    # CUDA_VISIBLE_DEVICES
# DATA_PATH   = "data/public.jsonl"
# OUTPUT_PATH = "results/starter_results.jsonl"
# MAX_TOKENS  = 32768

# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# print("1")
# import re
# print("2")
# import sys
# from pathlib import Path
# print("3")
# from typing import Optional

# from transformers import AutoTokenizer
# print("A")
# from vllm import LLM, SamplingParams
# print("B")
# from tqdm import tqdm


# ## 3. Load the Dataset
# 
# The dataset is stored as newline-delimited JSON (`.jsonl`). Each line is one question with the following fields:
# 
# | Field | Description |
# |---|---|
# | `id` | Unique question identifier |
# | `question` | Problem statement |
# | `options` | List of answer choices — present for **MCQ**, absent for **free-form** |
# | `answer` | Ground-truth answer (letter for MCQ, value/list for free-form) |

# In[4]:


data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

# Preview one MCQ and one free-form item
mcq_sample  = next(d for d in data if d.get("options"))
free_sample = next(d for d in data if not d.get("options"))

print("\n── MCQ sample ──")
print(json.dumps(mcq_sample, indent=2))
print("\n── Free-form sample ──")
print(json.dumps(free_sample, indent=2))


# ## 4. Prompt Construction
# 
# We use two system prompts depending on the question type:
# 
# - **MCQ** — the model must select the best answer letter and wrap it in `\boxed{}`
# - **Free-form** — the model solves step-by-step and puts the final answer in `\boxed{}`
# 
# `build_prompt()` returns the appropriate `(system, user)` pair for each item.

# In[5]:


# ── System prompts ─────────────────────────────────────────────

SYSTEM_MCQ = (
    "/no_think\n"
    "You are an expert mathematician. Select the single best answer letter.\n"
    "Rules:\n"
    "1. Work through the problem once. Do NOT re-check or second-guess.\n"
    "2. If your result does not match any option, pick the closest one and commit.\n"
    "3. Output your final answer as: FINAL: <letter>  (e.g. FINAL: B)\n"
    "4. Also put it in \\boxed{} (e.g. \\boxed{B}).\n"
    "Do NOT loop or reconsider."
)

SYSTEM_FREE_EASY = (
    "/no_think\n"
    "You are an expert mathematician. Solve this problem concisely.\n"
    "Give a direct, single-pass solution. Put the final answer in \\boxed{}.\n"
    "Do NOT over-explain. Do NOT verify with a second method."
)

SYSTEM_FREE_MEDIUM = (
    "You are an expert mathematician. Solve step-by-step in a single pass.\n"
    "Put your final answer in \\boxed{}. Stop immediately after \\boxed{}.\n"
    "Do NOT re-derive or double-check once you have a confident answer."
)

SYSTEM_FREE_HARD = (
    "You are an expert mathematician tackling a difficult problem.\n"
    "Reason carefully step-by-step. Put your final answer in \\boxed{}.\n"
    "Stop immediately once you write \\boxed{your answer}."
)

SYSTEM_MCQ_FALLBACK = (
    "/no_think\n"
    "Output ONLY a single letter (A, B, C, D, etc.) — nothing else."
)


def classify_difficulty(item: dict) -> str:
    """Keyword heuristic — replace with a real classifier if you train one."""
    q = item.get("question", "").lower()
    hard_kw = ["contour", "residue", "manifold", "eigenvalue", "differential equation",
                "laplace transform", "fourier", "complex analysis", "linear algebra",
                "number theory", "combinatorics", "proof"]
    easy_kw = ["simplify", "evaluate", "compute", "find the value", "what is",
                "percentage", "ratio", "average", "mean"]
    if any(k in q for k in hard_kw):
        return "hard"
    if any(k in q for k in easy_kw) and len(q) < 300:
        return "easy"
    return "medium"


def get_system_prompt(item: dict) -> str:
    if item.get("options"):
        return SYSTEM_MCQ
    diff = classify_difficulty(item)
    return {"easy": SYSTEM_FREE_EASY, "medium": SYSTEM_FREE_MEDIUM, "hard": SYSTEM_FREE_HARD}[diff]


def build_user_prompt(item: dict) -> str:
    question = item["question"]
    options  = item.get("options")
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return f"{question}\n\nOptions:\n{opts_text}"
    return question


# Verify
for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
    diff = "MCQ" if item.get("options") else classify_difficulty(item)
    print(f"── {label} ({diff}) system prompt preview ──")
    print(get_system_prompt(item)[:120], "...\n")



# SYSTEM_PROMPT_MATH = (
#     "You are an expert mathematician. Solve the problem step-by-step. "
#     "Put your final answer inside \\boxed{}. "
#     "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
#     "e.g. \\boxed{3, 7}."
# )

# SYSTEM_PROMPT_MCQ = (
#     "You are an expert mathematician. "
#     "Read the problem and the answer choices below, then select the single best answer. "
#     "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
# )


# def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
#     """Return (system_prompt, user_prompt) for a question."""
#     if options:
#         labels    = [chr(65 + i) for i in range(len(options))]
#         opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
#         return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
#     return SYSTEM_PROMPT_MATH, question


# # Verify with samples
# for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
#     sys_p, usr_p = build_prompt(item["question"], item.get("options"))
#     print(f"── {label} user prompt (first 200 chars) ──")
#     print(usr_p[:200], "...\n")


# ## 5. Load Model with vLLM (for general case, vLLM is faster)
# 
# We load **Qwen3-4B-Thinking-2507** with **INT8 quantization** via BitsAndBytes.  
# Setting `load_format="bitsandbytes"` tells vLLM to apply on-the-fly INT8 weight quantization, roughly halving GPU memory usage compared to BF16.
# 
# Key parameters:
# - `gpu_memory_utilization` — fraction of GPU VRAM reserved for the model and KV cache
# - `max_model_len` — maximum sequence length (prompt + generation)
# - `max_num_seqs` — maximum number of sequences processed in parallel

# In[6]:


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MODEL_ID,
    quantization=None,
    gpu_memory_utilization=0.85,
    max_model_len=32768,
    trust_remote_code=True,
    max_num_seqs=32,
    max_num_batched_tokens=32768,
)
print("Model loaded.")



# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# tokenizer.pad_token = tokenizer.eos_token


# llm = LLM(
#     model=MODEL_ID,
#     quantization=None,
#     gpu_memory_utilization=0.5,
#     max_model_len=4096,   # 👈 IMPORTANT: reduce
#     trust_remote_code=True,
#     max_num_seqs=16,      # 👈 IMPORTANT: reduce
#     max_num_batched_tokens=8192  # 👈 SAFE
# )


# # llm = LLM(
# #     model=MODEL_ID,
# #     quantization=None,   
# #     load_format="auto",
# #     gpu_memory_utilization=0.50,
# #     max_model_len=16384,
# #     trust_remote_code=True,
# #     max_num_seqs=256,
# #     max_num_batched_tokens=32768,
# # )
# # llm = LLM(
# #     model=MODEL_ID,
# #     quantization="bitsandbytes",
# #     load_format="bitsandbytes",
# #     enable_prefix_caching=False,
# #     gpu_memory_utilization=0.50,
# #     max_model_len=16384,
# #     trust_remote_code=True,
# #     max_num_seqs=256,
# #     max_num_batched_tokens=10000,#32768,
# # )

# sampling_params = SamplingParams(
#     max_tokens=MAX_TOKENS,
#     temperature=0.6,
#     top_p=0.95,
#     top_k=20,
#     min_p=0.0,
#     presence_penalty=0.0,
#     repetition_penalty=1.0,
# )

# print("Model loaded.")


# ## 5. Load Model with Transformers (alternative to vLLM for DataHub)
# 
# We load **Qwen3-4B-Thinking-2507** with **INT4 quantization** via BitsAndBytes.  
# 
# Key parameters:
# - `load_in_4bit` — quantization strategy of INT4

# In[7]:


import subprocess
subprocess.run([sys.executable, "-m", "pip", "show", "transformers"])


# In[8]:


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

# llm = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     trust_remote_code=True,
#     #quantization_config=bnb_config,
#     device_map="auto",
#     torch_dtype=torch.float16,
# )

# llm = llm.to("cuda")


# ## 6. Generate Responses
# 
# We format every question into a chat-template prompt, then call `llm.generate()` in one batched pass.  
# vLLM handles batching and scheduling internally — no manual batching needed.

# ### Generate with vLLM

# In[9]:


#new
def build_prompt_text(item: dict, system_override: str = None) -> str:
    system = system_override if system_override else get_system_prompt(item)
    user   = build_user_prompt(item)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_letter(text: str) -> str:
    """Check FINAL: X first, then \\boxed{X}, then last capital letter."""
    m = re.search(r"FINAL:\s*([A-Za-z])", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

print("Helpers defined.")


# In[10]:


#new
def run_mcq_fallback(item: dict) -> str:
    """
    If the first MCQ pass produces no parseable letter, run a minimal
    second prompt that asks for just the letter.
    """
    fallback_text = build_prompt_text(item, system_override=SYSTEM_MCQ_FALLBACK)
    fallback_text += f"\n\nJust the letter of the correct answer:"
    fallback_params = SamplingParams(max_tokens=10, temperature=0.0)
    out = llm.generate([fallback_text], fallback_params)
    return out[0].outputs[0].text.strip()

print("Fallback defined.")


# In[11]:


import time

def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

# Split by type
mcq_items  = [(i, d) for i, d in enumerate(data) if d.get("options")]
free_items = [(i, d) for i, d in enumerate(data) if not d.get("options")]
responses  = [""] * len(data)

run_start = time.time()

# ── MCQ batch ─────────────────────────────────────────────────
print(f"[1/4] MCQ batch — {len(mcq_items)} questions (greedy, max 2048 tokens)...")
t0 = time.time()
mcq_prompts = [build_prompt_text(d) for _, d in mcq_items]
mcq_params  = SamplingParams(max_tokens=2048, temperature=0.0, repetition_penalty=1.05)
mcq_outputs = llm.generate(mcq_prompts, mcq_params)

fallback_count = 0
total_tokens   = 0
for (orig_idx, item), out in zip(mcq_items, mcq_outputs):
    resp = out.outputs[0].text.strip()
    total_tokens += len(out.outputs[0].token_ids)
    if not extract_letter(resp):
        print(f"  [fallback] id={item.get('id')}")
        resp = run_mcq_fallback(item)
        fallback_count += 1
    responses[orig_idx] = resp

elapsed = time.time() - t0
print(f"  ✓ done in {fmt_time(elapsed)} | "
      f"avg {elapsed/len(mcq_items):.1f}s/q | "
      f"avg {total_tokens//len(mcq_items)} tokens/q | "
      f"{fallback_count} fallbacks")
print(f"  elapsed total: {fmt_time(time.time() - run_start)}\n")

# ── Free-form batches ──────────────────────────────────────────
sys_prompt_map = {
    "easy":   SYSTEM_FREE_EASY,
    "medium": SYSTEM_FREE_MEDIUM,
    "hard":   SYSTEM_FREE_HARD,
}
batch_labels = {"easy": "2/4", "medium": "3/4", "hard": "4/4"}

for diff in ["easy", "medium", "hard"]:
    tier_items = [(i, d) for i, d in free_items if classify_difficulty(d) == diff]
    if not tier_items:
        print(f"[{batch_labels[diff]}] free-form [{diff}] — 0 questions, skipping\n")
        continue

    budget = DIFFICULTY_BUDGETS[diff]
    print(f"[{batch_labels[diff]}] free-form [{diff}] — {len(tier_items)} questions "
          f"(max {budget['max_tokens']} tokens)...")
    t0 = time.time()

    prompts = [build_prompt_text(d, system_override=sys_prompt_map[diff])
               for _, d in tier_items]
    params  = SamplingParams(
        max_tokens=budget["max_tokens"],
        temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
    )
    outputs = llm.generate(prompts, params)

    total_tokens = 0
    maxed_out    = 0
    for (orig_idx, item), out in zip(tier_items, outputs):
        toks = len(out.outputs[0].token_ids)
        total_tokens += toks
        if toks >= budget["max_tokens"] - 10:   # hit the ceiling
            maxed_out += 1
        responses[orig_idx] = out.outputs[0].text.strip()

    elapsed = time.time() - t0
    avg_tok = total_tokens // len(tier_items)
    print(f"  ✓ done in {fmt_time(elapsed)} | "
          f"avg {elapsed/len(tier_items):.1f}s/q | "
          f"avg {avg_tok} tokens/q | "
          f"{maxed_out} hit token ceiling")
    print(f"  elapsed total: {fmt_time(time.time() - run_start)}\n")

total_elapsed = time.time() - run_start
print(f"Inference complete — {len(responses)} responses in {fmt_time(total_elapsed)}")

# Preview 3
for i in range(min(3, len(data))):
    print(f"\n── Response {i} (id={data[i].get('id')}) ──")
    print(responses[i][:300], "..." if len(responses[i]) > 300 else "")


# ### Generate with Transformers (for Datahub)

# In[ ]:


# from transformers import TextStreamer

# # Build prompts for first 5 entries
# prompts = []
# for item in data[:5]:
#     system, user = build_prompt(item["question"], item.get("options"))
#     prompt_text = tokenizer.apply_chat_template(
#         [
#             {"role": "system", "content": system},
#             {"role": "user", "content": user},
#         ],
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     prompts.append(prompt_text)

# responses = []

# for i, prompt in enumerate(prompts):
#     print(f"\n── Generating Response {i} (id={data[i].get('id')}) ──")

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512,
#     ).to(llm.device)

#     streamer = TextStreamer(
#         tokenizer,
#         skip_prompt=True,
#         skip_special_tokens=True,
#     )

#     with torch.no_grad():
#         output_ids = llm.generate(
#             **inputs,
#             max_new_tokens=MAX_TOKENS,
#             temperature=0.6,
#             top_p=0.95,
#             top_k=20,
#             repetition_penalty=1.0,
#             do_sample=True,
#             streamer=streamer,
#         )

#     # Decode only new tokens
#     new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
#     response = tokenizer.decode(
#         new_tokens,
#         skip_special_tokens=True,
#     ).strip()

#     responses.append(response)

#     print(f"\n── Finished Response {i} ──")


# ## 7. Score Responses
# 
# Scoring differs by question type:
# 
# - **MCQ**: extract the predicted letter from `\boxed{}` and compare to the gold letter (exact match).
# - **Free-form**: use `Judger.auto_judge()` which handles symbolic and numeric equivalence.
# 
# Each result record contains `{id, is_mcq, gold, response, correct}`.

# In[ ]:


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()

sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

results = []
for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]

    if is_mcq:
        correct = score_mcq(response, str(gold))
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            correct = judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        except Exception:
            correct = False

    results.append({
        "id":       item.get("id"),
        "is_mcq":   is_mcq,
        "gold":     gold,
        "response": response,
        "correct":  correct,
    })

print(f"Scoring complete. {len(results)} results.")

# def extract_letter(text: str) -> str:
#     m = re.search(r"\\boxed\{([A-Za-z])\}", text)
#     if m:
#         return m.group(1).upper()
#     matches = re.findall(r"\b([A-Z])\b", text.upper())
#     return matches[-1] if matches else ""


# def score_mcq(response: str, gold_letter: str) -> bool:
#     return extract_letter(response) == gold_letter.strip().upper()


# # Load Judger for free-form scoring
# sys.path.insert(0, ".")
# from judger import Judger
# judger = Judger(strict_extract=False)

# results = []
# for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
#     is_mcq = bool(item.get("options"))
#     gold   = item["answer"]

#     if is_mcq:
#         correct = score_mcq(response, str(gold))
#     else:
#         gold_list = gold if isinstance(gold, list) else [gold]
#         try:
#             correct = judger.auto_judge(
#                 pred=response,
#                 gold=gold_list,
#                 options=[[]] * len(gold_list),
#             )
#         except Exception:
#             correct = False

#     results.append({
#         "id":       item.get("id"),
#         "is_mcq":   is_mcq,
#         "gold":     gold,
#         "response": response,
#         "correct":  correct,
#     })

# print(f"Scoring complete. {len(results)} results.")


# ## 8. Summary
# 
# Print accuracy broken down by question type.

# In[ ]:


mcq_res  = [r for r in results if r["is_mcq"]]
free_res = [r for r in results if not r["is_mcq"]]

def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
print("=" * 50)


# ## 9. Save Results
# 
# Results are written as newline-delimited JSON.
# 
# **With evaluation** (public set — you have ground-truth):  
# Each line: `{id, is_mcq, gold, response, correct}`
# 
# **Without evaluation** (private test set — no ground-truth available):  
# Each line: `{id, is_mcq, response}` — omit `gold` and `correct`.
# 
# Toggle `SAVE_EVAL` below accordingly.

# In[ ]:


SAVE_EVAL = True   # Set to False when running on the private test set

out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    for r in results:
        if SAVE_EVAL:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                      "response": r["response"], "correct": r["correct"]}
        else:
            record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
        f.write(json.dumps(record) + "\n")

print(f"Saved {len(results)} records to {out_path}")


# ## Next Steps
# 
# This notebook gives you a working baseline. Here are directions to improve your score:
# 
# - **Prompt engineering** — try different system prompts or few-shot examples inside the user turn
# - **Sampling parameters** — adjust `temperature`, `top_p`, or use majority voting across multiple samples
# - **Fine-tuning** — the competition allows model fine-tuning; see the course resources for guidance
# 
# Good luck!

# In[ ]:




