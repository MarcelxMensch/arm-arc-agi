#!/usr/bin/env python3
"""Standalone checkpoint evaluator. Architecture-agnostic.

Loads config from checkpoint directory's all_config.yaml, reconstructs the model,
runs evaluation, and writes results to JSON. The training process picks up these
results and logs them to the same W&B run.

Usage:
    # Evaluate a single checkpoint
    python evaluate_checkpoint.py --checkpoint experiments/my-exp/checkpoints/step_10000

    # Use EMA weights
    python evaluate_checkpoint.py --checkpoint experiments/my-exp/checkpoints/step_10000 --ema

    # Watch for new checkpoints and evaluate them as they appear
    python evaluate_checkpoint.py --watch experiments/my-exp/checkpoints --poll-interval 60

    # Override data path (e.g. evaluate arc-agi-1 model on arc-agi-2 data)
    python evaluate_checkpoint.py --checkpoint path/to/step_X --data-path data/arc2concept-aug-1000

    # Log results to W&B (resumes the training run if wandb_run_id.txt exists)
    python evaluate_checkpoint.py --checkpoint path/to/step_X --wandb

    # Auto-discover experiments under a root dir (re-scans each cycle for new experiments)
    python evaluate_checkpoint.py --watch-root experiments --poll-interval 120

    # Multi-GPU parallel evaluation (each worker claims checkpoints atomically)
    python evaluate_checkpoint.py --watch-root experiments --worker-id 0 --num-workers 4
"""
import argparse
import glob
import json
import os
import sys
import threading
import time

import torch
import torch.distributed as dist
import yaml

# ── Performance: enable TF32 for matmuls and cuDNN auto-tuning ──
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Ensure repo root is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import from pretrain_ttt rather than pretrain so that ARM checkpoints
# (arm_episode_mode=True in all_config.yaml) get the arm_episode batch
# contract. pretrain_ttt is a superset of pretrain — baseline TRM evals are
# unaffected because arm_episode_mode defaults to False in PretrainConfig.
from pretrain_ttt import (
    PretrainConfig,
    TrainState,
    create_dataloader,
    create_evaluators,
    evaluate,
)
from utils.functions import load_model_class


def load_config(checkpoint_path: str) -> PretrainConfig:
    """Load config from all_config.yaml in the checkpoint directory."""
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            f"Ensure all_config.yaml exists in the checkpoint directory."
        )
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    if "evaluators" not in raw or not raw["evaluators"]:
        raw["evaluators"] = [{"name": "arc@ARC"}]
    return PretrainConfig(**raw)


def _strip_orig_mod(state_dict: dict) -> dict:
    """Strip _orig_mod. prefix added by torch.compile when saving state_dict."""
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state_dict.items()}


def create_model_for_eval(config: PretrainConfig, train_metadata, checkpoint_path: str, use_ema: bool = False):
    """Create model from config and load checkpoint. Architecture-agnostic."""
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        num_task_identifiers=getattr(train_metadata, 'num_task_identifiers', None),
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

        # Load weights into UNCOMPILED model — _strip_orig_mod handles checkpoints
        # saved from compiled models (strips _orig_mod. prefix); checkpoints saved
        # from uncompiled models load directly. Compile happens after load.
        if use_ema and os.path.isfile(checkpoint_path + "_ema"):
            # Load base weights first, then apply EMA
            state_dict = torch.load(checkpoint_path, map_location="cuda")
            state_dict = _strip_orig_mod(state_dict)
            model.load_state_dict(state_dict, assign=True)
            ema_shadow = torch.load(checkpoint_path + "_ema", map_location="cuda")
            ema_shadow = _strip_orig_mod(ema_shadow)
            for name, param in model.named_parameters():
                if name in ema_shadow:
                    param.data.copy_(ema_shadow[name])
            print(f"Loaded EMA weights from {checkpoint_path}_ema")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cuda")
            state_dict = _strip_orig_mod(state_dict)
            model.load_state_dict(state_dict, assign=True)
            if use_ema:
                print(f"Warning: --ema requested but no EMA file found at {checkpoint_path}_ema, using base weights")

        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

    return model


def extract_step(checkpoint_path: str) -> int:
    """Extract step number from checkpoint filename like step_10000."""
    base = os.path.basename(checkpoint_path)
    if base.startswith("step_"):
        try:
            return int(base.split("_")[1])
        except (IndexError, ValueError):
            pass
    return 0


def evaluate_single(checkpoint_path: str, config: PretrainConfig, use_ema: bool = False, eval_batch_size: int = 64, timing_sink: dict | None = None):
    """Evaluate a single checkpoint and return metrics dict."""
    config.global_batch_size = eval_batch_size
    config.load_checkpoint = None  # We load manually

    rank = 0
    world_size = 1

    # Init process group for evaluators that need dist.
    #
    # Pick a FREE port per worker (not a fixed 29500). The watcher script
    # spawns one evaluate_checkpoint.py process per GPU, and all children
    # inherit `MASTER_PORT` from the parent env — `setdefault` is useless
    # here because the parent already exported a value. Multiple workers
    # racing to bind the same TCP port raise EADDRINUSE. Asking the OS for
    # an ephemeral port (bind to 0 → read back assigned port) gives every
    # worker a unique address with no coordination.
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        import socket
        with socket.socket() as _sock:
            _sock.bind(("", 0))
            _free_port = _sock.getsockname()[1]
        os.environ["MASTER_PORT"] = str(_free_port)
        dist.init_process_group(backend="gloo", init_method="env://", rank=0, world_size=1)
    cpu_group = dist.new_group(backend="gloo")

    # Create dataloaders
    train_loader, train_metadata = create_dataloader(
        config, "train", rank=rank, world_size=world_size,
        test_set_mode=False, epochs_per_iter=1, global_batch_size=eval_batch_size,
    )
    eval_loader, eval_metadata = create_dataloader(
        config, "test", rank=rank, world_size=world_size,
        test_set_mode=True, epochs_per_iter=1, global_batch_size=eval_batch_size,
    )
    evaluators = create_evaluators(config, eval_metadata)

    # Create model
    model = create_model_for_eval(config, train_metadata, checkpoint_path, use_ema=use_ema)
    step = extract_step(checkpoint_path)

    train_state = TrainState(
        model=model, optimizers=(), optimizer_lrs=(), carry=None,
        step=step, total_steps=0,
    )
    train_state.model.eval()

    print(f"Evaluating checkpoint step {step}...")
    # Time ONLY the inference pass over the full eval set, cuda-synced both ends.
    # This is the cost boundary ARC Prize uses for open models: inference dollars,
    # NOT model load / torch.compile / data-prep / dist init. CUDA is async, so
    # synchronize before reading the clock or the number is meaningless.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    metrics = evaluate(
        config, train_state, eval_loader, eval_metadata, evaluators,
        rank=rank, world_size=world_size, cpu_group=cpu_group,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _wall_s = time.perf_counter() - _t0
    if timing_sink is not None:
        timing_sink["eval_wall_clock_s"] = _wall_s
        timing_sink["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        timing_sink["eval_batch_size"] = eval_batch_size

    return metrics, step


def _flatten_metrics(metrics: dict, step: int) -> dict:
    """Flatten nested dicts (e.g. {"all": {"accuracy": 0.5}} -> {"all/accuracy": 0.5})."""
    flat = {"_step": step}
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, dict):
                for k2, v2 in value.items():
                    flat[f"{key}/{k2}"] = v2
            else:
                flat[key] = value
    return flat


def write_eval_results(metrics: dict, step: int, out_dir: str):
    """Write eval results to JSON for the training process to pick up."""
    eval_dir = os.path.join(out_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    flat = _flatten_metrics(metrics, step)

    out_path = os.path.join(eval_dir, f"eval_step_{step}.json")
    with open(out_path, "w") as f:
        json.dump(flat, f, indent=2, default=lambda o: float(o))
    print(f"Results written to {out_path}")


def write_cost_results(metrics: dict, step: int, ckpt_dir: str, config: PretrainConfig, timing: dict, args) -> None:
    """Write a timing + USD-cost breakdown to cost_step_X.json.

    Cost is computed the way ARC Prize costs open-weight models: inference
    wall-clock hours x retail GPU rental rate, reported per task (per test
    puzzle). num_tasks is read from the eval split's test_puzzles.json.
    Only runs when timing was captured (single --checkpoint path).
    """
    if not timing:
        return

    wall_s = timing.get("eval_wall_clock_s")

    # Number of eval tasks = test puzzles in the eval split (ARC-AGI-1 public = 400).
    num_tasks = None
    data_paths = (getattr(config, "data_paths_test", None)
                  or getattr(config, "data_paths", None) or [])
    for dp in data_paths:
        tp = os.path.join(dp, "test_puzzles.json")
        if os.path.isfile(tp):
            try:
                with open(tp) as f:
                    num_tasks = len(json.load(f))
                break
            except (OSError, ValueError):
                pass

    flat = _flatten_metrics(metrics, step)
    out = {
        "_step": step,
        "gpu": timing.get("gpu"),
        "eval_batch_size": timing.get("eval_batch_size"),
        "eval_wall_clock_s": wall_s,
        "eval_wall_clock_h": (wall_s / 3600.0) if wall_s else None,
        "num_tasks": num_tasks,
        "pass@1": flat.get("ARC/pass@1"),
        "pass@2": flat.get("ARC/pass@2"),
        "usd_per_hour": args.usd_per_hour,
    }
    if args.usd_per_hour is not None and wall_s is not None:
        cost_total = wall_s / 3600.0 * args.usd_per_hour
        out["cost_total_usd"] = cost_total
        out["cost_per_task_usd"] = (cost_total / num_tasks) if num_tasks else None

    out_path = args.cost_out or os.path.join(ckpt_dir, "eval_results", f"cost_step_{step}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o))

    print(f"Cost results written to {out_path}")
    if wall_s is not None:
        print(f"  wall_clock: {wall_s:.1f}s ({wall_s / 3600:.3f}h) on {out['gpu']} | pass@2={out['pass@2']}")
    if out.get("cost_per_task_usd") is not None:
        print(f"  cost: ${out['cost_total_usd']:.4f} total, ${out['cost_per_task_usd']:.5f}/task "
              f"over {num_tasks} tasks @ ${args.usd_per_hour}/h")


def log_to_wandb(metrics: dict, step: int, config: PretrainConfig, ckpt_dir: str):
    """Log eval results to W&B, resuming the training run if possible."""
    import wandb

    flat = _flatten_metrics(metrics, step)
    flat.pop("_step", None)  # W&B uses the step parameter instead

    # Try to find existing run ID
    run_id_path = os.path.join(ckpt_dir, "wandb_run_id.txt")
    run_id = None
    if os.path.isfile(run_id_path):
        with open(run_id_path) as f:
            run_id = f.read().strip()

    if run_id:
        wandb.init(
            project=config.project_name,
            id=run_id,
            resume="allow",
        )
        print(f"Resumed W&B run {run_id}")
    else:
        wandb.init(
            project=config.project_name,
            name=f"{config.run_name or 'eval'}-eval",
            config=config.model_dump(),
            tags=["eval"],
        )
        print(f"Created new W&B eval run (no run ID found at {run_id_path})")

    wandb.log(flat, step=step)
    wandb.finish()
    print(f"Logged eval results to W&B at step {step}")


def find_checkpoints(ckpt_dir: str) -> list:
    """Find all step_* checkpoint files (excluding _ema, _opt, _all_preds sidecars)."""
    files = glob.glob(os.path.join(ckpt_dir, "step_*"))
    checkpoints = [f for f in files if os.path.isfile(f) and "_ema" not in f and "_opt" not in f and "_all_preds" not in f]
    return sorted(checkpoints, key=extract_step)


def _get_evaluated_set(ckpt_dir: str) -> set:
    """Return set of checkpoint names already evaluated."""
    evaluated = set()
    eval_dir = os.path.join(ckpt_dir, "eval_results")
    if os.path.isdir(eval_dir):
        for f in os.listdir(eval_dir):
            if f.endswith(".json"):
                evaluated.add(f.replace("eval_", "").replace(".json", ""))
    return evaluated


def _get_claimed_set(ckpt_dir: str) -> set:
    """Return set of checkpoint names that are claimed (in-progress or done)."""
    claimed = set()
    eval_dir = os.path.join(ckpt_dir, "eval_results")
    if os.path.isdir(eval_dir):
        for f in os.listdir(eval_dir):
            # Both .json (done) and .lock (in-progress) count as claimed
            if f.endswith(".json"):
                claimed.add(f.replace("eval_", "").replace(".json", ""))
            elif f.endswith(".lock"):
                lock_path = os.path.join(eval_dir, f)
                # Stale lock detection: if no heartbeat (mtime update) in 10 minutes,
                # the owning worker is dead and we can reclaim.
                try:
                    age = time.time() - os.path.getmtime(lock_path)
                    if age > 600:
                        os.remove(lock_path)
                        print(f"Removed stale lock: {f} (no heartbeat for {age/60:.0f}m)")
                        continue
                except OSError:
                    continue
                claimed.add(f.replace("eval_", "").replace(".lock", ""))
    return claimed


class LockHeartbeat:
    """Background thread that periodically touches a lock file to signal liveness."""

    def __init__(self, lock_path: str, interval: float = 120):
        self._lock_path = lock_path
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self._interval):
            try:
                os.utime(self._lock_path, None)
            except OSError:
                break

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)


def try_claim_checkpoint(ckpt_dir: str, ckpt_name: str, worker_id: int) -> "LockHeartbeat | None":
    """Atomically claim a checkpoint for evaluation.

    Returns a LockHeartbeat if claimed successfully (keeps the lock alive),
    or None if another worker already holds the claim.
    """
    eval_dir = os.path.join(ckpt_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    lock_path = os.path.join(eval_dir, f"eval_{ckpt_name}.lock")
    try:
        # O_CREAT | O_EXCL is atomic: fails if file already exists
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{worker_id}\n".encode())
        os.close(fd)
        return LockHeartbeat(lock_path)
    except FileExistsError:
        return None


def release_claim(ckpt_dir: str, ckpt_name: str, heartbeat: "LockHeartbeat | None" = None):
    """Remove the lock file after evaluation completes (result JSON is the real record)."""
    if heartbeat:
        heartbeat.stop()
    lock_path = os.path.join(ckpt_dir, "eval_results", f"eval_{ckpt_name}.lock")
    try:
        os.remove(lock_path)
    except OSError:
        pass


def discover_checkpoint_dirs(roots: list, exclude_patterns: list | None = None) -> list:
    """Discover checkpoint directories under root dirs by looking for all_config.yaml.

    Args:
        roots: Root directories to search.
        exclude_patterns: Substrings to exclude — any path containing one of these is skipped.
    """
    exclude_patterns = exclude_patterns or []
    dirs = []
    for root in roots:
        for config_file in glob.glob(os.path.join(root, "*/checkpoints/all_config.yaml")):
            ckpt_dir = os.path.dirname(config_file)
            if any(pat in ckpt_dir for pat in exclude_patterns):
                continue
            dirs.append(ckpt_dir)
    return sorted(set(dirs))


def watch_directories(watch_dirs: list, poll_interval: int, use_ema: bool, data_path: str | None, data_path_test: str | None, eval_batch_size: int, use_wandb: bool = False, watch_roots: list | None = None, exclude_patterns: list | None = None, worker_id: int = 0, num_workers: int = 1):
    """Watch multiple checkpoint directories, evaluate new checkpoints as they appear.

    If watch_roots is provided, re-discovers checkpoint dirs under those roots each cycle,
    so new experiments are picked up automatically.

    Multi-GPU: Each worker has a unique worker_id. Workers claim checkpoints atomically
    via lock files so no two workers evaluate the same checkpoint.
    """
    dir_list = "\n  ".join(watch_dirs) if watch_dirs else "(dynamic discovery)"
    print(f"[Worker {worker_id}/{num_workers}] Watching {len(watch_dirs)} initial checkpoint dir(s) (poll every {poll_interval}s):\n  {dir_list}")
    if watch_roots:
        print(f"[Worker {worker_id}] Auto-discovering new experiments under: {', '.join(watch_roots)}")

    known_dirs = set(watch_dirs)

    while True:
        # Re-discover checkpoint dirs if roots are provided
        if watch_roots:
            discovered = discover_checkpoint_dirs(watch_roots, exclude_patterns=exclude_patterns)
            for d in discovered:
                if d not in known_dirs:
                    known_dirs.add(d)
                    print(f"\n[Worker {worker_id}] Discovered new experiment: {d}")
            current_dirs = sorted(known_dirs)
        else:
            current_dirs = watch_dirs

        for ckpt_dir in current_dirs:
            if not os.path.isdir(ckpt_dir):
                continue

            # Get the set of already-claimed checkpoints (done + in-progress)
            claimed = _get_claimed_set(ckpt_dir)

            checkpoints = find_checkpoints(ckpt_dir)
            for ckpt_path in checkpoints:
                ckpt_name = os.path.basename(ckpt_path)

                # Fast skip: already done or claimed by another worker
                if ckpt_name in claimed:
                    continue

                # Atomic claim: only one worker wins
                heartbeat = try_claim_checkpoint(ckpt_dir, ckpt_name, worker_id)
                if heartbeat is None:
                    continue

                exp_name = os.path.basename(os.path.dirname(ckpt_dir))
                print(f"\n[Worker {worker_id}] [{exp_name}] Claimed checkpoint: {ckpt_name}")
                try:
                    config = load_config(ckpt_path)
                    if data_path:
                        config.data_paths = [data_path]
                    if data_path_test:
                        config.data_paths_test = [data_path_test]

                    metrics, step = evaluate_single(ckpt_path, config, use_ema=use_ema, eval_batch_size=eval_batch_size)
                    write_eval_results(metrics, step, ckpt_dir)
                    release_claim(ckpt_dir, ckpt_name, heartbeat)
                    if use_wandb and metrics:
                        log_to_wandb(metrics, step, config, ckpt_dir)
                except RuntimeError as e:
                    err_msg = str(e)
                    if "Missing key(s) in state_dict" in err_msg or "Unexpected key(s) in state_dict" in err_msg or "size mismatch" in err_msg:
                        # Architecture mismatch — permanent failure, mark as done so we don't retry
                        print(f"[Worker {worker_id}] Skipping {ckpt_dir}/{ckpt_name}: incompatible checkpoint (architecture mismatch)")
                        print(f"[Worker {worker_id}] Mismatch detail: {err_msg[:500]}")
                        step = extract_step(ckpt_path)
                        skip_path = os.path.join(ckpt_dir, "eval_results", f"eval_step_{step}.json")
                        with open(skip_path, "w") as f:
                            json.dump({"step": step, "skipped": True, "reason": "architecture_mismatch"}, f)
                        release_claim(ckpt_dir, ckpt_name, heartbeat)
                    else:
                        print(f"[Worker {worker_id}] Eval failed for {ckpt_dir}/{ckpt_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        release_claim(ckpt_dir, ckpt_name, heartbeat)
                except Exception as e:
                    print(f"[Worker {worker_id}] Eval failed for {ckpt_dir}/{ckpt_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Release claim on failure so another worker (or retry) can pick it up
                    release_claim(ckpt_dir, ckpt_name, heartbeat)

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Standalone checkpoint evaluator")
    parser.add_argument("--checkpoint", default=None, help="Path to a single checkpoint file")
    parser.add_argument("--watch", nargs="+", default=None, help="Watch one or more checkpoint directories")
    parser.add_argument("--watch-root", nargs="+", default=None, help="Root dirs to auto-discover experiments (re-scans each cycle)")
    parser.add_argument("--exclude-pattern", nargs="+", default=None, help="Substrings to exclude from auto-discovered paths (e.g. '-opt' 'softprompt')")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls in watch mode")
    parser.add_argument("--ema", action="store_true", help="Use EMA weights if available")
    parser.add_argument("--data-path", default=None, help="Override data path from config")
    parser.add_argument("--data-path-test", default=None, help="Override test data path")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--wandb", action="store_true", help="Log results to W&B (resumes training run if possible)")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for multi-GPU mode (usually auto-set by launcher)")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of parallel workers (usually auto-set by launcher)")
    parser.add_argument("--scoring", choices=["voting", "poe", "poe_norm"], default=None,
                        help="Scoring method: voting (default), poe (Product of Experts), poe_norm (normalized PoE)")
    parser.add_argument("--usd-per-hour", type=float, default=None,
                        help="Retail GPU rental rate in USD/hour. If set (single --checkpoint mode), "
                             "writes cost_step_X.json with inference wall-clock and $/task.")
    parser.add_argument("--cost-out", default=None,
                        help="Path for the cost JSON (default: <ckpt_dir>/eval_results/cost_step_X.json)")
    args = parser.parse_args()

    if args.checkpoint is None and args.watch is None and args.watch_root is None:
        parser.error("Provide --checkpoint, --watch, or --watch-root")

    if args.checkpoint:
        config = load_config(args.checkpoint)
        if args.data_path:
            config.data_paths = [args.data_path]
        if args.data_path_test:
            config.data_paths_test = [args.data_path_test]
        if args.scoring:
            for cfg in config.evaluators:
                cfg.__pydantic_extra__["scoring"] = args.scoring

        timing: dict = {}
        metrics, step = evaluate_single(args.checkpoint, config, use_ema=args.ema, eval_batch_size=args.eval_batch_size, timing_sink=timing)
        ckpt_dir = os.path.dirname(args.checkpoint)
        write_eval_results(metrics, step, ckpt_dir)
        write_cost_results(metrics, step, ckpt_dir, config, timing, args)

        if args.wandb and metrics:
            log_to_wandb(metrics, step, config, ckpt_dir)

        if metrics:
            print(f"\nStep {step} results:")
            for k, v in sorted(metrics.items()) if isinstance(metrics, dict) else []:
                print(f"  {k}: {v}")

    elif args.watch or args.watch_root:
        initial_dirs = args.watch or []
        # If watch_root provided, do an initial discovery too
        if args.watch_root:
            initial_dirs += discover_checkpoint_dirs(args.watch_root, exclude_patterns=args.exclude_pattern)
        watch_directories(
            watch_dirs=initial_dirs,
            poll_interval=args.poll_interval,
            use_ema=args.ema,
            data_path=args.data_path,
            data_path_test=args.data_path_test,
            eval_batch_size=args.eval_batch_size,
            use_wandb=args.wandb,
            watch_roots=args.watch_root,
            exclude_patterns=args.exclude_pattern,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
