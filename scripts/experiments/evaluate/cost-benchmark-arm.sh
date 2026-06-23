#!/usr/bin/env bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S
#SBATCH --job-name=cost-bench-arm
#SBATCH --output=logs/evaluate/cost-bench-arm.out
#SBATCH --error=logs/evaluate/cost-bench-arm.err

# Timed cost benchmark: ARM on the ARC-AGI-1 public eval (400 tasks).
#
# Times ONLY the inference pass (cuda-synced inside evaluate_checkpoint.py) and
# converts wall-clock to USD/task the SAME way ARC Prize costs open-weight models:
#   cost_total   = inference_hours x retail_GPU_usd_per_hour
#   cost_per_task = cost_total / num_tasks   (num_tasks = 400 public eval puzzles)
#
# Run this AND cost-benchmark-trm-opt.sh on the SAME L40S partition, then
# compare_cost.py builds the controlled ARM vs TRM-Opt table. Same GPU + same
# protocol is what makes the "cheaper than TRM" claim airtight.
#
# Usage:
#   sbatch scripts/experiments/evaluate/cost-benchmark-arm.sh
#   USD_PER_HOUR=0.79 sbatch scripts/experiments/evaluate/cost-benchmark-arm.sh
set -e

# -- Tunables (KEEP IDENTICAL to the TRM-Opt script) ------------------
# Defaults = canonical ARC-AGI-1 ARM (n_H=0). Override via env for ARC-AGI-2, e.g.
#   CKPT=experiments/train-trm-abstraction-arc-agi-2-resume/checkpoints/step_X \
#   DATA_TEST=data/arc2concept-aug-1000 sbatch .../cost-benchmark-arm.sh
CKPT="${CKPT:-experiments/abl-v1-noh/checkpoints/step_700000}"   # canonical ARM (n_H=0); latest surviving ckpt, pass@2~0.39 (flat 575k-625k, step_625000 was pruned)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
DATA_TEST="${DATA_TEST:-data/arc1concept-aug-1000}"
# Retail L40S rental rate, USD/hour. NOT $0: the cluster is free to us, but the
# ARC-style number prices the GPU at what it would rent for. RunPod L40S public
# rate ~ $0.86/h (2026-06). Override with USD_PER_HOUR=x sbatch ...
USD_PER_HOUR="${USD_PER_HOUR:-0.86}"

# -- Root ------------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$ROOT"
mkdir -p logs/evaluate

if [ ! -f "$CKPT" ]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

export DATALOADER_NUM_WORKERS=1                 # >1 hard-asserts in puzzle_dataset.py
export MASTER_PORT=$((29600 + RANDOM % 100))

echo "============================================================"
echo "  ARM cost benchmark"
echo "  GPU:        $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Checkpoint: $CKPT"
echo "  Batch:      $EVAL_BATCH_SIZE | Rate: \$$USD_PER_HOUR/h"
echo "============================================================"

PYTHONPATH="$ROOT" uv run --no-project python evaluate_checkpoint.py \
  --checkpoint "$CKPT" \
  --ema \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --scoring voting \
  --data-path-test "$DATA_TEST" \
  --usd-per-hour "$USD_PER_HOUR"

STEP="$(basename "$CKPT" | sed 's/step_//')"
echo ""
echo "Cost JSON: $(dirname "$CKPT")/eval_results/cost_step_${STEP}.json"
echo "Sanity check (ARC-AGI-1 canonical ARM): pass@2 should be ~0.39."
