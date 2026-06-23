#!/bin/bash
#SBATCH --job-name=arc-agi-1-concept
#SBATCH --output=experiments/datasets/arc-agi-1-concept.out
#SBATCH --error=experiments/datasets/arc-agi-1-concept.err
#SBATCH --time=1:00:00
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
# ARC-AGI-1 + ConceptARC Dataset Generation Script

# Configuration
N_AUGMENTATIONS=1000              # Number of augmented versions per task
DIFFICULTY=""                  # Optional difficulty filter (e.g., "10x10", "30x30")
SEED=42                        # Random seed for reproducibility
INCLUDE_EVALUATION=false       # Include ARC-AGI evaluation set
INCLUDE_TEST=true              # Include test examples
INCLUDE_MINIMAL=true           # Include ConceptARC MinimalTasks

# Paths
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

OUTPUT_DIR="${PROJECT_ROOT}/data"
RAW_DATA_DIR="${PROJECT_ROOT}/utils/dataset/raw-data"
PYTHON_SCRIPT="${PROJECT_ROOT}/utils/dataset/build_arc_agi_1_concept_4_dataset.py"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Ensure experiments/datasets exists (sbatch writes .out/.err here)
mkdir -p "${PROJECT_ROOT}/experiments/datasets"

# Verify paths
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "Error: Project root not found at ${PROJECT_ROOT}"
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "Error: Python script not found at ${PYTHON_SCRIPT}"
    exit 1
fi

# Activate virtual environment
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated virtual environment: ${VENV_DIR}"
else
    echo "Error: Virtual environment not found at ${VENV_DIR}"
    exit 1
fi

# Build command arguments
ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --raw-data-dir "${RAW_DATA_DIR}"
    --seed "${SEED}"
    --n-augmentations "${N_AUGMENTATIONS}"
)

if [ "$INCLUDE_EVALUATION" = "true" ]; then
    ARGS+=(--include-evaluation)
fi

if [ "$INCLUDE_TEST" = "true" ]; then
    ARGS+=(--include-test)
fi

if [ "$INCLUDE_MINIMAL" = "false" ]; then
    ARGS+=(--no-minimal)
fi

if [ -n "$DIFFICULTY" ]; then
    ARGS+=(--difficulty "${DIFFICULTY}")
fi

# Run dataset builder
cd "${PROJECT_ROOT}"
python3 "${PYTHON_SCRIPT}" "${ARGS[@]}"
