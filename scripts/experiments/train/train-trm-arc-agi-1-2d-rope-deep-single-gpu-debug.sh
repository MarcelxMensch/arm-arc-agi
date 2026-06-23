#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S
#SBATCH --job-name=train-trm-arc-agi-1-2d-rope-deep-single-gpu-debug
#SBATCH --output=logs/train/train-trm-arc-agi-1-2d-rope-deep-single-gpu-debug.out
#SBATCH --error=logs/train/train-trm-arc-agi-1-2d-rope-deep-single-gpu-debug.err

# Experiment: 2D-rope-mixed positional encoding + deeper L-level (3 layers instead of 2).
# Motivated by ViTARC (Li et al., 2024): spatially-aware positional encoding is critical
# for ARC grid reasoning. Extra layer uses the ~8% VRAM headroom from baseline.
# No FiLM/spatial-bias/consistency — those added overhead without convergence benefit.

# --- Define this experiment (used by set-up-experiment.sh to create the experiment folder) ---
export EXPERIMENT_NAME="train-trm-arc-agi-1-2d-rope-deep-single-gpu-debug"
export EXPERIMENT_CATEGORY="training"
export EXPERIMENT_DESCRIPTION="TRM on ARC-AGI-1: 2D-rope-mixed positional encoding, 3 L-layers (1 GPU, debug logging)."

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

# micro_batch=128 on L40S, 3 accum steps -> effective 384 (reduced for L_layers=3)
PYTHONPATH="$EXPERIMENT_ROOT" uv run --no-project python pretrain.py \
  arch=trm \
  arch.pos_encodings=2d-rope-mixed \
  arch.L_layers=3 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  data_paths="[data/arc1concept-aug-1000]" \
  global_batch_size=384 \
  gradient_accumulation_steps=3 \
  +run_name=train-arc-agi-1-2d-rope-deep-single-gpu-debug \
  +checkpoint_path="$CHECKPOINT_PATH" \
  ema=True \
  debug=True \
  log_interval=1 \
  skip_eval=True \
  checkpoint_interval=5000
