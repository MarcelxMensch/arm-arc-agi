#!/bin/bash
#SBATCH --job-name=arc-agi-2-sp
#SBATCH --output=experiments/datasets/arc-agi-2-softprompt.out
#SBATCH --error=experiments/datasets/arc-agi-2-softprompt.err
#SBATCH --time=24:00:00
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
# ARC-AGI-2 Dataset for Grid Encoder + Softprompt Pipeline
#
# Outputs per-example rows with demo grids, color_map, transform_id,
# and task/puzzle identifiers. Split by original_task_id (train 80% / test 20%)
# so the test set contains only truly unseen tasks.
#
# Generate with: sbatch scripts/dataset/arc-agi-2-softprompt.sh
# Use for training: arch=hrm_softprompt, data_paths=[.../arc-agi-2-X-Y/train]

# Configuration
# MAX_TASKS: unset = use all tasks; set to N for debugging (e.g. MAX_TASKS=2 sbatch ...)
N_AUGMENTATIONS="${N_AUGMENTATIONS:-1000}" # Augmentations per task
DIFFICULTY="${DIFFICULTY:-}"               # Optional filter (e.g., "10x10", "30x30")
SEED="${SEED:-42}"                         # Random seed
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
TEST_RATIO="${TEST_RATIO:-0.2}"

# Paths
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

OUTPUT_DIR="${PROJECT_ROOT}/data"
RAW_DATA_DIR="${PROJECT_ROOT}/utils/dataset/raw-data"
PYTHON_SCRIPT="${PROJECT_ROOT}/utils/dataset/build_arc_agi_2_dataset.py"
VENV_DIR="${PROJECT_ROOT}/.venv"

mkdir -p "${PROJECT_ROOT}/experiments/datasets"

if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "Error: Project root not found at ${PROJECT_ROOT}"
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "Error: Python script not found at ${PYTHON_SCRIPT}"
    exit 1
fi

if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated virtual environment: ${VENV_DIR}"
else
    echo "Warning: Virtual environment not found at ${VENV_DIR}, using system Python"
fi

ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --raw-data-dir "${RAW_DATA_DIR}"
    --seed "${SEED}"
    --n-augmentations "${N_AUGMENTATIONS}"
    --train-ratio "${TRAIN_RATIO}"
    --test-ratio "${TEST_RATIO}"
)

if [ -n "${MAX_TASKS:+x}" ]; then
    ARGS+=(--max-tasks "${MAX_TASKS}")
fi

if [ -n "$DIFFICULTY" ]; then
    ARGS+=(--difficulty "${DIFFICULTY}")
fi

echo "=== ARC-AGI-2 Grid Encoder Dataset ==="
echo "Tasks: ${MAX_TASKS:-all}"
echo "N_AUGMENTATIONS: ${N_AUGMENTATIONS}"
echo "SEED: ${SEED}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "TRAIN_RATIO: ${TRAIN_RATIO}"
echo "TEST_RATIO: ${TEST_RATIO}"
echo "======================================="

cd "${PROJECT_ROOT}"
python3 "${PYTHON_SCRIPT}" "${ARGS[@]}"
