#!/usr/bin/env bash
# Shared scaffold for all experiments: creates the experiment output folder and
# exports EXPERIMENT_NAME / EXPERIMENT_DIR / EXPERIMENT_ROOT for the calling script.
# Use by sourcing from an experiment script after setting EXPERIMENT_NAME (and optionally
# EXPERIMENT_CATEGORY, EXPERIMENT_DESCRIPTION), or call with: set-up-experiment.sh <name> [category] [description]

set_up_experiment() {
  local name="$1"
  local category="${2:-experiment}"
  local description="${3:-No description.}"

  if [ -z "$name" ]; then
    echo "Usage: set-up-experiment.sh <name> [category] [description]" >&2
    echo "   or: set EXPERIMENT_NAME=... [EXPERIMENT_CATEGORY=...] [EXPERIMENT_DESCRIPTION=...]; source set-up-experiment.sh" >&2
    return 1
  fi

  # Project root: same logic as dataset scripts (SLURM or script-relative)
  local root
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    root="${SLURM_SUBMIT_DIR:-$(pwd)}"
  else
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    root="$(cd "$script_dir/../.." && pwd)"
  fi

  local exp_dir="$root/experiments/$name"
  mkdir -p "$exp_dir"

  echo "Created experiment directory: $exp_dir"
  export EXPERIMENT_NAME="$name"
  export EXPERIMENT_DIR="$exp_dir"
  export EXPERIMENT_ROOT="$root"
}

# If script is sourced and EXPERIMENT_NAME is set, run set-up
if [ -n "${EXPERIMENT_NAME:-}" ]; then
  set_up_experiment "$EXPERIMENT_NAME" "${EXPERIMENT_CATEGORY:-experiment}" "${EXPERIMENT_DESCRIPTION:-No description.}"
# If script is run with arguments
elif [ "${BASH_SOURCE[0]:-}" = "$0" ] && [ $# -gt 0 ]; then
  set_up_experiment "$1" "${2:-experiment}" "${3:-No description.}"
fi
