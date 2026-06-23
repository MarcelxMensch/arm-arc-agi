#!/bin/bash
# Submit all dataset generation scripts in scripts/dataset/ via sbatch.
# Each script is run from the project root so SLURM_SUBMIT_DIR is correct.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${SCRIPT_DIR}/dataset"

cd "${PROJECT_ROOT}"

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Error: Dataset scripts directory not found at ${DATASET_DIR}"
    exit 1
fi

for script in "${DATASET_DIR}"/*.sh; do
    if [ -f "${script}" ]; then
        name="$(basename "${script}" .sh)"
        echo "Submitting ${name}..."
        sbatch "${script}"
    fi
done

echo "Done. All dataset scripts submitted."
