#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S
#SBATCH --job-name=train-trm-abstraction-arc-agi-1-single-gpu-debug
#SBATCH --output=logs/train/train-trm-abstraction-arc-agi-1-single-gpu-debug.out
#SBATCH --error=logs/train/train-trm-abstraction-arc-agi-1-single-gpu-debug.err

# Accelerated Recursive Reasoning Model v2: Perceiver-style cross-attention H-level.
# L-level: D=512, 2 blocks, full 900-token self-attention (same cost as baseline).
# H-level: 32 learned latent tokens, D=256, 3 self-attention blocks.
# Cross-attention bridges: L→H (perceive) + H→L (broadcast), ~10% overhead over baseline.
# H_cycles=2, L_cycles=4, global_batch_size=660, gradient_accumulation=3 (micro=220).
# Same micro_batch as baseline since L-level attention dominates.

# --- Define this experiment ---
export EXPERIMENT_NAME="train-trm-abstraction-arc-agi-1-single-gpu-debug"
export EXPERIMENT_CATEGORY="training"
export EXPERIMENT_DESCRIPTION="Training the Accelerated Recursive Reasoning Model on ARC-AGI-1 (1 GPU, debug logging). Spatial compression forces object-level abstraction at H-level."

# --- Shared scaffold ---
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

CHECKPOINT_PATH="experiments/${EXPERIMENT_NAME}/checkpoints"

# NOTE: DATALOADER_NUM_WORKERS > 1 crashes on L40S partition
export DATALOADER_NUM_WORKERS=1
PYTHONPATH="$EXPERIMENT_ROOT" uv run --no-project python pretrain.py \
  arch=trm_abstraction \
  data_paths="[data/arc1concept-aug-1000]" \
  +arch.num_latent_tokens=32 \
  arch.gate_latent_input=true \
  global_batch_size=660 \
  gradient_accumulation_steps=3 \
  +run_name=train-trm-abstraction-arc-agi-1-single-gpu-debug \
  +checkpoint_path="$CHECKPOINT_PATH" \
  ema=True \
  debug=True \
  log_interval=1 \
  skip_eval=True \
  checkpoint_interval=5000
