# Server run scripts (Stage 1 + Stage 2 plan)

Run these on a **Linux machine with an NVIDIA GPU**. Copy `data/public.jsonl` and `data/private.jsonl` into the repo before training.

## Quick start

```bash
cd /path/to/cse151b_competition-1
git checkout LoRA && git pull

bash scripts/server/setup.sh
source .venv/bin/activate

# tmux recommended
tmux new -s lora
bash scripts/server/run_full_pipeline.sh
# Ctrl-b d to detach

bash scripts/server/monitor.sh
```

## After training

```bash
bash scripts/server/eval_checkpoints.sh
# Optional stronger Stage 1:
# bash scripts/server/iterate_stage1_v2.sh
# STAGE1_ADAPTER=workspaces/stage1_reasoning_v2/final_adapter bash scripts/server/run_stage2_only.sh

bash scripts/server/private_submit.sh
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPU_ID` | `0` | `CUDA_VISIBLE_DEVICES` |
| `STAGE2_ROOT` | `workspaces/stage2_adapt` | Stage 2 output dir |
| `STAGE1_ADAPTER` | `workspaces/stage1_reasoning/final_adapter` | For stage2-only rerun |
