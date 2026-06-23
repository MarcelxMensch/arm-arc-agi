#!/usr/bin/env bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S
#SBATCH --job-name=cost-bench-trm-opt
#SBATCH --output=logs/evaluate/cost-bench-trm-opt.out
#SBATCH --error=logs/evaluate/cost-bench-trm-opt.err

# Timed cost benchmark: TRM-Opt on the ARC-AGI-1 public eval (400 tasks).
#
# The matched-compute baseline for ARM. TRM-Opt trained with skip_eval=True, so
# it was never evaluated during training; this is its first eval. Reuses the same
# arch-agnostic evaluate_checkpoint.py path as ARM, so the timing boundary and
# protocol are byte-identical between the two runs.
#
# MUST run on the SAME L40S partition, SAME batch size, SAME USD_PER_HOUR as
# cost-benchmark-arm.sh, otherwise the comparison is not controlled.
#
# Usage:
#   sbatch scripts/experiments/evaluate/cost-benchmark-trm-opt.sh
#   TRM_CKPT=experiments/abl-c3-trmopt/checkpoints/step_70000 sbatch scripts/experiments/evaluate/cost-benchmark-trm-opt.sh
set -e

# -- Tunables (KEEP IDENTICAL to the ARM script) ----------------------
# Defaults = matched ARC-AGI-1 TRM-Opt baseline. Override via env for ARC-AGI-2, e.g.
#   CKPT_DIR=experiments/train-trm-arc-agi-2/checkpoints \
#   DATA_TEST=data/arc2concept-aug-1000 sbatch .../cost-benchmark-trm-opt.sh
CKPT_DIR="${CKPT_DIR:-experiments/abl-c3-trmopt/checkpoints}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
DATA_TEST="${DATA_TEST:-data/arc1concept-aug-1000}"
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

# Pick the final checkpoint (highest step) unless TRM_CKPT is given explicitly.
# Exclude _ema / _opt / _all_preds sidecars; match bare step_<digits>.
if [ -n "${TRM_CKPT:-}" ]; then
  CKPT="$TRM_CKPT"
else
  CKPT="$(ls -1 "${CKPT_DIR}"/step_* 2>/dev/null | grep -E 'step_[0-9]+$' | sort -t_ -k2 -n | tail -1)"
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "TRM-Opt checkpoint not found under ${CKPT_DIR} (got: '${CKPT}')" >&2
  echo "Set TRM_CKPT=path/to/step_X explicitly if checkpoints live elsewhere." >&2
  exit 1
fi

export DATALOADER_NUM_WORKERS=1
export MASTER_PORT=$((29600 + RANDOM % 100))

echo "============================================================"
echo "  TRM-Opt cost benchmark"
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
echo "Cost JSON: ${CKPT_DIR}/eval_results/cost_step_${STEP}.json"
echo "Sanity check: pass@1 should be ~0.209 (TRM-Opt registry number)."
