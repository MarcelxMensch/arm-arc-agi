#!/bin/bash
#SBATCH --job-name=install
#SBATCH --output=experiments/install.out
#SBATCH --error=experiments/install.err
#SBATCH --time=1:00:00
#SBATCH --partition=4090
#SBATCH --gres=gpu:1

# Installation script for arm-arc-agi project
# Uses uv to install dependencies and initializes git submodules

set -e

# Get project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv locally..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
else
    echo "uv is already installed"
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv installation failed or not in PATH"
    exit 1
fi

# Initialize git submodules (shallow clone, no nested submodules, parallel fetch)
echo "Initializing git submodules..."
# Use --depth 1 for shallow clones (faster, less disk space)
# Use --jobs to parallelize fetching
# Don't use --recursive to avoid unnecessary nested submodules (e.g., ConceptARC/editor)
git submodule update --init --depth 1 --jobs 4

# Clean install: remove existing venv and lock file for fresh dependency resolution
if [ -d ".venv" ]; then
    echo "Removing existing .venv for clean install..."
    rm -rf .venv
fi

if [ -f "uv.lock" ]; then
    echo "Removing uv.lock for fresh dependency resolution..."
    rm -f uv.lock
fi

# Install dependencies using uv
echo "Installing dependencies with uv..."
# uv sync creates .venv and installs dependencies with proper version resolution
uv sync

echo "Installation complete!"
echo "Verify torch + sympy:"
uv run python3 -c "import torch; import sympy; print(f'torch={torch.__version__}, sympy={sympy.__version__}')"
