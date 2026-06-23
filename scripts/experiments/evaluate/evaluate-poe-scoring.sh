#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S
#SBATCH --job-name=eval-poe-scoring
#SBATCH --output=logs/evaluate/eval-poe-scoring.out
#SBATCH --error=logs/evaluate/eval-poe-scoring.err

# Compare scoring methods (voting vs PoE vs PoE-norm) on TRM and Abstraction checkpoints.
# Evaluates the same checkpoint 3x with different scoring, writes results side-by-side.
#
# Usage:
#   # Default: evaluate both TRM and Abstraction baselines
#   sbatch scripts/experiments/evaluate/evaluate-poe-scoring.sh
#
#   # Custom checkpoint dir and steps
#   sbatch scripts/experiments/evaluate/evaluate-poe-scoring.sh \
#       --ckpt-dir experiments/train-trm-arc-agi-1-single-gpu-debug/checkpoints \
#       --steps 75000 100000

set -e

# ── Defaults ────────────────────────────────────────────────────────
CKPT_DIRS=()
STEPS=()
OUTPUT_DIR="experiments/eval-abstraction-poe-scoring"
EVAL_BATCH_SIZE=64
USE_EMA="--ema"

# ── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-dir)
      CKPT_DIRS+=("$2"); shift 2 ;;
    --steps)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        STEPS+=("$1"); shift
      done ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --no-ema)
      USE_EMA=""; shift ;;
    --ema)
      USE_EMA="--ema"; shift ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Defaults when no args given ─────────────────────────────────────
if [ ${#CKPT_DIRS[@]} -eq 0 ]; then
  CKPT_DIRS=(
    "experiments/train-trm-arc-agi-1-single-gpu-debug/checkpoints"
    "experiments/train-trm-abstraction-arc-agi-1-single-gpu-debug/checkpoints"
  )
fi

if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(100000)
fi

# ── Root dir ────────────────────────────────────────────────────────
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

cd "$ROOT"
mkdir -p logs/evaluate "$OUTPUT_DIR"

# Use a random port to avoid conflict with eval-watcher
export MASTER_PORT=$((29600 + RANDOM % 100))

SCORING_METHODS="voting poe poe_norm"

for CKPT_DIR in "${CKPT_DIRS[@]}"; do
  # Derive a short experiment name for result filenames
  EXP_NAME=$(basename "$(dirname "$CKPT_DIR")")

  for step in "${STEPS[@]}"; do
    CHECKPOINT="${CKPT_DIR}/step_${step}"

    if [ ! -f "$CHECKPOINT" ]; then
      echo "Checkpoint not found: $CHECKPOINT — skipping"
      continue
    fi

    echo ""
    echo "============================================================"
    echo "  Experiment: $EXP_NAME"
    echo "  Checkpoint: $CHECKPOINT"
    echo "============================================================"

    for scoring in $SCORING_METHODS; do
      echo ""
      echo "------------------------------------------------------------"
      echo "  Scoring: $scoring | Step: $step"
      echo "------------------------------------------------------------"

      RESULT_FILE="${OUTPUT_DIR}/eval_${EXP_NAME}_step_${step}_${scoring}.json"

      if [ -f "$RESULT_FILE" ]; then
        echo "  Already evaluated, skipping. Delete $RESULT_FILE to re-run."
        continue
      fi

      PYTHONPATH="$ROOT" uv run --no-project python evaluate_checkpoint.py \
        --checkpoint "$CHECKPOINT" \
        $USE_EMA \
        --eval-batch-size "$EVAL_BATCH_SIZE" \
        --scoring "$scoring"

      # Copy the generic eval result to a scoring-specific filename
      GENERIC_RESULT="${CKPT_DIR}/eval_results/eval_step_${step}.json"
      if [ -f "$GENERIC_RESULT" ]; then
        cp "$GENERIC_RESULT" "$RESULT_FILE"
        echo "  Result: $RESULT_FILE"
      fi
    done
  done
done

echo ""
echo "============================================================"
echo "  All evaluations complete. Results in: $OUTPUT_DIR"
echo "============================================================"
