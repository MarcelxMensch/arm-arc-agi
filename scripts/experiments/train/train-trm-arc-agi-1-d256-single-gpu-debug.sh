#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S
#SBATCH --job-name=train-trm-arc-agi-1-d256-single-gpu-debug
#SBATCH --output=logs/train/train-trm-arc-agi-1-d256-single-gpu-debug.out
#SBATCH --error=logs/train/train-trm-arc-agi-1-d256-single-gpu-debug.err

# TRM with hidden_size=256 (no spatial compression).
# Tests whether width reduction alone (512->256) improves throughput
# while retaining full spatial resolution (900 tokens).
# ~1.7M params vs baseline 6.8M. Should be significantly faster per step.
# Neither the TRM paper nor the supporting paper tested this configuration.
#
# Hypothesis: The TRM's "less is more" principle may extend to model width.
# If 2 layers beats 4 layers due to reduced overfitting, 256 hidden may
# beat 512 for the same reason -- with a large throughput bonus.

# --- Define this experiment ---
export EXPERIMENT_NAME="train-trm-arc-agi-1-d256-single-gpu-debug"
export EXPERIMENT_CATEGORY="training"
export EXPERIMENT_DESCRIPTION="Training TRM with hidden_size=256 on ARC-AGI-1 (1 GPU). Width reduction ablation: no spatial compression, full 900-token resolution. Tests if smaller width follows the 'less is more' principle."

# --- Shared scaffold ---
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
source "$ROOT/scripts/experiments/set-up-experiment.sh"

if [ -z "${EXPERIMENT_DIR:-}" ]; then
  echo "Scaffold failed." >&2
  exit 1
fi

cd "$EXPERIMENT_ROOT" || exit 1
mkdir -p "$ROOT/logs/train"

CHECKPOINT_PATH="experiments/${EXPERIMENT_NAME}/checkpoints"

# hidden_size=256: ~4x fewer params (1.7M vs 6.8M), attention is half cost.
# Baseline fits micro=220 at D=512. At D=256 we should fit ~600+ per micro_batch.
# global_batch=1800, grad_accum=3 -> micro=600.
# If OOM, reduce to global_batch=1200 (micro=400).
export DATALOADER_NUM_WORKERS=1
PYTHONPATH="$EXPERIMENT_ROOT" uv run --no-project python pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.hidden_size=256 \
  arch.num_heads=8 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  arch.puzzle_emb_ndim=256 \
  arch.puzzle_emb_len=16 \
  arch.gate_latent_input=true \
  global_batch_size=1200 \
  gradient_accumulation_steps=3 \
  +run_name=train-trm-arc-agi-1-d256-single-gpu-debug \
  +checkpoint_path="$CHECKPOINT_PATH" \
  ema=True \
  debug=True \
  log_interval=1 \
  skip_eval=True \
  checkpoint_interval=5000
