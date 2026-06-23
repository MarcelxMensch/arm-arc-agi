#!/usr/bin/env bash
# Short run to validate W&B charts (q_*, full_act_*, exact_accuracy). Kept separate from production launcher.
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=4090
#SBATCH --cpus-per-task=4
#SBATCH --job-name=emp64-wandb-v-reptile
#SBATCH --output=logs/train/wandb-verify-reptile-%j.out
#SBATCH --error=logs/train/wandb-verify-reptile-%j.err

set -euo pipefail

export EXPERIMENT_NAME="train-trm-abstraction-support-ttt-emp64-supervised-reptile"
export EXPERIMENT_CATEGORY="training"
export EXPERIMENT_DESCRIPTION="W&B logging verification: openai_reptile shared batch, max_steps=450."

if [ -n "${SLURM_JOB_ID:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

# shellcheck source=scripts/experiments/set-up-experiment.sh
source "$ROOT/scripts/experiments/set-up-experiment.sh"

if [ -z "${EXPERIMENT_DIR:-}" ]; then
  echo "Scaffold failed." >&2
  exit 1
fi

cd "$EXPERIMENT_ROOT" || exit 1
mkdir -p "$ROOT/logs/train"

CHECKPOINT_PATH="experiments/${EXPERIMENT_NAME}/checkpoints"
mkdir -p "$CHECKPOINT_PATH"

export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:?WANDB_ENTITY must be set}"
export WANDB_PROJECT="${WANDB_PROJECT:-arm-arc-agi}"
export WANDB_MODE="${WANDB_MODE:-online}"
export DISABLE_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_SUFFIX="${SLURM_JOB_ID:-local}"
PYTHONPATH="$EXPERIMENT_ROOT" uv run --no-project python pretrain_support_ttt_supervised_reptile.py \
  arch=trm_abstraction_support_ttt_emp64_supervised_reptile \
  support_ttt_mode=true \
  data_paths="[data/trm-abstraction-support-ttt-arc1-aug-1000]" \
  global_batch_size=96 \
  gradient_accumulation_steps=1 \
  lr=5e-5 \
  weight_decay=0.1 \
  "+run_name=emp64-openai-reptile-wandb-verify-${RUN_SUFFIX}" \
  +checkpoint_path="$CHECKPOINT_PATH" \
  arch.meta.max_steps=450 \
  log_interval=10 \
  debug=True \
  ema=False \
  skip_eval=True \
  checkpoint_interval=5000
