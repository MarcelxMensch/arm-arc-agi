#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=L40S
#SBATCH --job-name=eval-watcher
#SBATCH --output=logs/evaluate/eval-watcher.out
#SBATCH --error=logs/evaluate/eval-watcher.err

# Evaluation watcher: polls experiments/ for new checkpoints and evaluates them.
# 4 GPUs, 48h. Each GPU runs one worker; workers coordinate via atomic lock files.
#
# Usage:
#   sbatch scripts/experiments/evaluate/evaluate-watcher.sh
#   # or watch specific dirs:
#   sbatch scripts/experiments/evaluate/evaluate-watcher.sh \
#       experiments/abl-v6-hc1/checkpoints

set -e

# Auto-discovers all experiments under experiments/.
# Override by passing checkpoint dirs as CLI args.
INCLUDE_PATHS=()
EXCLUDE_PATTERNS=()

if [ -n "${SLURM_JOB_ID:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

cd "$ROOT"
mkdir -p logs/evaluate

# Detect number of GPUs
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#GPU_IDS[@]}
else
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  NUM_GPUS=${NUM_GPUS:-1}
  # Build GPU_IDS array: 0, 1, 2, ...
  GPU_IDS=()
  for ((i=0; i<NUM_GPUS; i++)); do
    GPU_IDS+=("$i")
  done
fi

echo "Detected ${NUM_GPUS} GPU(s): ${GPU_IDS[*]}"

# Build args: use --watch for explicit dirs, or --watch-root for dynamic discovery
EVAL_ARGS=(--ema --poll-interval 120 --eval-batch-size 64)

if [ $# -gt 0 ]; then
  # Explicit directories provided as CLI arguments (highest priority)
  EVAL_ARGS+=(--watch "$@")
  echo "Watching explicitly provided directories:"
  printf '  %s\n' "$@"
elif [ ${#INCLUDE_PATHS[@]} -gt 0 ]; then
  # Use the curated include list
  EVAL_ARGS+=(--watch "${INCLUDE_PATHS[@]}")
  echo "Watching include-listed directories:"
  printf '  %s\n' "${INCLUDE_PATHS[@]}"
else
  # Auto-discover with exclude filtering
  EVAL_ARGS+=(--watch-root experiments)
  if [ ${#EXCLUDE_PATTERNS[@]} -gt 0 ]; then
    EVAL_ARGS+=(--exclude-pattern "${EXCLUDE_PATTERNS[@]}")
    echo "Auto-discovering experiments under experiments/ (excluding: ${EXCLUDE_PATTERNS[*]})"
  else
    echo "Auto-discovering experiments under experiments/ (re-scans each cycle)"
  fi
fi

EVAL_ARGS+=(--num-workers "$NUM_GPUS")

# Launch one worker per GPU
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
  GPU="${GPU_IDS[$i]}"
  # Each worker gets its own GPU, unique MASTER_PORT, and worker ID
  MASTER_PORT=$((29500 + i))
  echo "Launching worker ${i} on GPU ${GPU} (MASTER_PORT=${MASTER_PORT})"

  CUDA_VISIBLE_DEVICES="$GPU" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="$MASTER_PORT" \
  PYTHONPATH="$ROOT" \
  uv run --no-project python evaluate_checkpoint.py \
    "${EVAL_ARGS[@]}" --worker-id "$i" \
    > "logs/evaluate/eval-watcher-worker-${i}.out" \
    2> "logs/evaluate/eval-watcher-worker-${i}.err" &

  PIDS+=($!)
done

echo "All ${NUM_GPUS} workers launched. PIDs: ${PIDS[*]}"

# Wait for all workers — if any exits, kill the rest and exit with its code
wait_and_propagate() {
  while true; do
    for idx in "${!PIDS[@]}"; do
      pid="${PIDS[$idx]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
        EXIT_CODE=$?
        if [ "$EXIT_CODE" -ne 0 ]; then
          echo "Worker ${idx} (PID ${pid}) exited with code ${EXIT_CODE}. Stopping all workers."
          for p in "${PIDS[@]}"; do
            kill "$p" 2>/dev/null || true
          done
          exit "$EXIT_CODE"
        fi
      fi
    done
    sleep 5
  done
}

wait_and_propagate
