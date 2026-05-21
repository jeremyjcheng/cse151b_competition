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
source scripts/server/activate_vllm.sh
# or: conda activate vllm   (from project root, after remove_venv)
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
