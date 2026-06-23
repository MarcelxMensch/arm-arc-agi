"""Export W&B run history to local CSV for use in experiment reports.

Usage:
    python -m utils.wandb_export <entity>/<project>/<run_id> [--out-dir DIR] [--metrics KEY1,KEY2,...]

Examples:
    # Export all train/ metrics for a run into its experiment directory
    python -m utils.wandb_export <entity>/<project>/<run_id>

    # Export specific metrics to a custom directory
    python -m utils.wandb_export <entity>/<project>/<run_id> \
        --metrics train/lm_loss,train/accuracy,train/exact_accuracy \
        --out-dir experiments/train-trm-arc-agi-2-single-gpu
"""
import argparse
import os

import wandb
import pandas as pd


DEFAULT_METRICS = [
    "train/lm_loss",
    "train/accuracy",
    "train/exact_accuracy",
    "train/q_halt_loss",
    "train/q_halt_accuracy",
    "train/steps",
    "train/lr",
    "train/samples_seen",
]


def export_run(entity: str, project: str, run_id: str, out_dir: str, metrics: list[str] | None = None):
    """Download run history from W&B and save as CSV."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    keys = metrics or DEFAULT_METRICS
    # Also grab _step and _runtime for x-axis options
    keys_with_meta = list(set(["_step", "_runtime"] + keys))

    print(f"Downloading history for {entity}/{project}/{run_id}...")
    print(f"  Metrics: {keys}")

    history = run.history(keys=keys_with_meta, pandas=True)

    # Drop rows where all requested metrics are NaN
    history = history.dropna(subset=[k for k in keys if k in history.columns], how="all")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wandb_{run_id}.csv")
    history.to_csv(out_path, index=False)

    print(f"  Saved {len(history)} rows -> {out_path}")

    # Also save run config as JSON
    import json
    config_path = os.path.join(out_dir, f"wandb_{run_id}_config.json")
    with open(config_path, "w") as f:
        json.dump(run.config, f, indent=2)
    print(f"  Config  -> {config_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export W&B run history to CSV")
    parser.add_argument("run_path", help="entity/project/run_id")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: experiment dir from run name)")
    parser.add_argument("--metrics", default=None, help="Comma-separated metric keys")
    args = parser.parse_args()

    parts = args.run_path.split("/")
    if len(parts) != 3:
        parser.error("run_path must be entity/project/run_id")
    entity, project, run_id = parts

    metrics = args.metrics.split(",") if args.metrics else None

    # Default out_dir: look up the run's display name and use experiments/<name>/
    out_dir = args.out_dir
    if out_dir is None:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        display_name = run.name
        out_dir = os.path.join("experiments", display_name)
        print(f"  Auto-detected experiment dir: {out_dir}")

    export_run(entity, project, run_id, out_dir, metrics)


if __name__ == "__main__":
    main()
