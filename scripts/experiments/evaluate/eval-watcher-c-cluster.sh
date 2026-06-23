#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:3
#SBATCH --partition=4090
#SBATCH --cpus-per-task=6
#SBATCH --job-name=eval-watcher-c
#SBATCH --output=logs/evaluate/eval-watcher-c-cluster.out
#SBATCH --error=logs/evaluate/eval-watcher-c-cluster.err

# Eval watcher for the finished C-cluster ablations (abl-c1/c2/c3).
# 3 workers, one per 4090 GPU, atomic lock-files via the --worker-id mechanism
# in evaluate_checkpoint.py. Watches ONLY the c-cluster checkpoint dirs.
#
# NB: no `set -e` — this script manages background worker processes and `wait`s
# on them; a non-zero worker exit must be handled explicitly
# (wait_and_propagate), not abort the script silently.

if [ -n "${SLURM_JOB_ID:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

cd "$ROOT"
mkdir -p logs/evaluate

INCLUDE_PATHS=(
  experiments/abl-c1-nobcnogate/checkpoints
  experiments/abl-c2-strip/checkpoints
  experiments/abl-c3-trmopt/checkpoints
)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#GPU_IDS[@]}
else
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
  NUM_GPUS=${NUM_GPUS:-1}
  GPU_IDS=()
  for ((i=0; i<NUM_GPUS; i++)); do
    GPU_IDS+=("$i")
  done
fi

echo "Detected ${NUM_GPUS} GPU(s): ${GPU_IDS[*]}"
echo "Watching: ${INCLUDE_PATHS[*]}"

EVAL_ARGS=(--ema --poll-interval 120 --eval-batch-size 64 --watch "${INCLUDE_PATHS[@]}" --num-workers "$NUM_GPUS")

PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
  GPU="${GPU_IDS[$i]}"
  MASTER_PORT=$((29500 + i))
  echo "Launching worker ${i} on GPU ${GPU} (MASTER_PORT=${MASTER_PORT})"

  CUDA_VISIBLE_DEVICES="$GPU" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="$MASTER_PORT" \
  PYTHONPATH="$ROOT" \
  uv run --no-project python evaluate_checkpoint.py \
    "${EVAL_ARGS[@]}" --worker-id "$i" \
    > "logs/evaluate/eval-watcher-c-cluster-worker-${i}.out" \
    2> "logs/evaluate/eval-watcher-c-cluster-worker-${i}.err" &

  PIDS+=($!)
done

echo "All ${NUM_GPUS} workers launched. PIDs: ${PIDS[*]}"

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
