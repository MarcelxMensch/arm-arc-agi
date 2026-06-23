#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=4090
#SBATCH --job-name=dataset-arc-agi
#SBATCH --output=logs/dataset/dataset-arc-agi.out
#SBATCH --error=logs/dataset/dataset-arc-agi.err

# Project root: under SLURM use submit dir (sbatch copies script so $0 is wrong)
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
else
  ROOT=$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")/../.." && pwd)
fi
cd "$ROOT" || exit 1

PYTHONPATH="$ROOT" uv run --no-project python -m utils.dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000
