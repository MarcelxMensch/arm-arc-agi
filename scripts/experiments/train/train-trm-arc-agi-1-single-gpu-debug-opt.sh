#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S
#SBATCH --job-name=train-trm-arc-agi-1-single-gpu-debug-opt
#SBATCH --output=logs/train/train-trm-arc-agi-1-single-gpu-debug-opt.out
#SBATCH --error=logs/train/train-trm-arc-agi-1-single-gpu-debug-opt.err

# Debug variant: logs all params/grads via wandb.watch and logs metrics every step.
# global_batch_size=720 with 3 gradient accumulation steps (micro_batch=240 per step).

# --- Define this experiment (used by set-up-experiment.sh to create the experiment folder) ---
export EXPERIMENT_NAME="train-trm-arc-agi-1-single-gpu-debug-opt"
export EXPERIMENT_CATEGORY="training"
export EXPERIMENT_DESCRIPTION="Training the TRM on ARC-AGI-1 (1 GPU, debug logging, optimized for speed)."

# --- Shared scaffold: create the experiment output folder ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
# shellcheck source=scripts/experiments/set-up-experiment.sh
source "$ROOT/scripts/experiments/set-up-experiment.sh"

if [ -z "${EXPERIMENT_DIR:-}" ]; then
  echo "Scaffold failed." >&2
  exit 1
fi

cd "$EXPERIMENT_ROOT" || exit 1
mkdir -p "$ROOT/logs/train"

# Checkpoint path relative to project root: experiments/<name>/checkpoints
CHECKPOINT_PATH="experiments/${EXPERIMENT_NAME}/checkpoints"

# micro_batch=220 on L40S, 3 accum steps -> effective 660
# NOTE: DATALOADER_NUM_WORKERS > 1 crashes on L40S partition
export DATALOADER_NUM_WORKERS=1
PYTHONPATH="$EXPERIMENT_ROOT" uv run --no-project python pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  arch.num_heads=16 \
  global_batch_size=660 \
  gradient_accumulation_steps=3 \
  +run_name=train-arc-agi-1-single-gpu-debug-opt \
  +checkpoint_path="$CHECKPOINT_PATH" \
  ema=True \
  debug=True \
  log_interval=1 \
  skip_eval=True \
  checkpoint_interval=5000
