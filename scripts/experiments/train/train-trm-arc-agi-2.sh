#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=L40S
#SBATCH --job-name=train-trm-arc-agi-2
#SBATCH --output=logs/train/train-trm-arc-agi-2.out
#SBATCH --error=logs/train/train-trm-arc-agi-2.err

# Train TRM on ARC-AGI with 4 GPUs. Checkpoints saved under experiments/train-trm-arc-agi/checkpoints/
# DataLoader uses num_workers=0 by default to avoid shared-memory Bus errors on multi-GPU nodes.
# To use workers (faster loading, needs more /dev/shm): export DATALOADER_NUM_WORKERS=1 before sbatch.

# --- Define this experiment (used by set-up-experiment.sh to create the experiment folder) ---
export EXPERIMENT_NAME="train-trm-arc-agi-2"
export EXPERIMENT_CATEGORY="training"
export EXPERIMENT_DESCRIPTION="Training the TRM on ARC-AGI-2 (4 GPUs)."

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

PYTHONPATH="$EXPERIMENT_ROOT" uv run --no-project torchrun \
  --nproc-per-node 4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/arc2concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  global_batch_size=768 \
  gradient_accumulation_steps=1 \
  +run_name=train-arc-agi-2 \
  +checkpoint_path="$CHECKPOINT_PATH" \
  ema=True \
  log_interval=100 \
  skip_eval=True \
  checkpoint_interval=5000
