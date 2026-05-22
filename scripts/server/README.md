# Server run scripts (conda env `vllm`)

Uses your existing **conda environment `vllm`** — does **not** create `.venv`.

## One-time setup

```bash
cd /path/to/cse151b_competition-1
git pull origin LoRA

# Remove broken .venv and install missing deps into conda vllm
bash install_gpu.sh

# Or step by step:
bash scripts/server/remove_venv.sh
conda activate vllm
bash scripts/server/install_into_vllm.sh
```

## Activate vllm first (required)

```bash
# Option A — activate in current shell (bash or zsh)
source scripts/server/activate_vllm.sh

# Option B — safe if sourcing still closes your terminal: new bash subshell
bash scripts/server/enter_vllm.sh

# Diagnose — always bash, never source
bash scripts/server/diagnose_vllm.sh
```

**zsh users:** old scripts used `set -e` + `exit` when sourced, which **closes the whole terminal**. Pull latest `LoRA` branch. If unsure, use `bash scripts/server/enter_vllm.sh` instead of `source`.

Do NOT use `python` from `/opt/conda/bin` (base) or `~/.local` pip installs.

### `libcudart.so.13` / vLLM import fails

Symptoms: `ImportError: libcudart.so.13`, vllm loaded from `~/.local/lib/python3.13/...`.

```bash
cd ~/private/cse151b_competition/cse151b_competition
source scripts/server/activate_vllm.sh   # sets PYTHONNOUSERSITE=1 + CUDA libs
bash scripts/server/diagnose_vllm.sh
bash scripts/server/test_installs.sh
```

If diagnose still fails on a GPU node, load your cluster CUDA module (example):

```bash
module avail cuda 2>/dev/null | head
module load cuda/13.0   # name varies by site
source scripts/server/activate_vllm.sh
bash scripts/server/diagnose_vllm.sh
```

Then re-run eval:

```bash
STAGE2_ROOT=workspaces/stage2_adapt_v2 bash scripts/server/run_stage2_eval_sweep.sh
```

## Test installs (run anytime)

```bash
source scripts/server/activate_vllm.sh
bash scripts/server/test_installs.sh
```

Must end with `PASSED: environment ready for Stage 1/2 pipeline.`

## Training

```bash
source scripts/server/activate_vllm.sh
bash scripts/server/run_full_pipeline.sh
bash scripts/server/monitor.sh
```

Pipeline Python comes from conda `vllm` only. `scripts/modular_pipeline/*.py` do not reference `.venv`. Old logs showing `.venv/bin/python` mean an outdated script or a leftover `.venv` folder.

## Different conda env name

```bash
CONDA_ENV_NAME=myenv bash scripts/server/test_installs.sh
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CONDA_ENV_NAME` | `vllm` | Conda env to activate |
| `GPU_ID` | `0` | CUDA device |
| `REASONING_STEPS` | `1000` | Stage 1 steps |
| `ADAPT_STEPS` | `60` | Stage 2 steps |
