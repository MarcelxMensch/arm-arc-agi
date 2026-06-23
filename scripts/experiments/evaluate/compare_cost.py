#!/usr/bin/env python3
"""Build the controlled ARM vs TRM-Opt cost comparison from two cost_step_*.json files.

Both JSONs are produced by evaluate_checkpoint.py --usd-per-hour. The comparison
is only valid if both ran on the SAME GPU, SAME eval batch size, and SAME task
count; this script warns loudly when any of those invariants is violated.

Usage:
    python scripts/experiments/evaluate/compare_cost.py \
        --arm experiments/abl-v1-noh/checkpoints/eval_results/cost_step_700000.json \
        --trm experiments/abl-c3-trmopt/checkpoints/eval_results/cost_step_71823.json
"""
import argparse
import json


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _fmt(v, spec: str = "") -> str:
    if v is None:
        return "n/a"
    try:
        return format(v, spec) if spec else str(v)
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled ARM vs TRM-Opt cost table")
    ap.add_argument("--arm", required=True, help="ARM cost_step_X.json")
    ap.add_argument("--trm", required=True, help="TRM-Opt cost_step_X.json")
    args = ap.parse_args()

    arm = _load(args.arm)
    trm = _load(args.trm)

    cols = ["model", "gpu", "pass@1", "pass@2", "$/task", "$ total", "wall(s)", "bs", "tasks"]
    widths = [8, 22, 7, 7, 10, 9, 9, 4, 6]

    def row(label: str, d: dict) -> str:
        cells = [
            label,
            _fmt(d.get("gpu")),
            _fmt(d.get("pass@1"), ".3f"),
            _fmt(d.get("pass@2"), ".3f"),
            _fmt(d.get("cost_per_task_usd"), ".5f"),
            _fmt(d.get("cost_total_usd"), ".3f"),
            _fmt(d.get("eval_wall_clock_s"), ".1f"),
            _fmt(d.get("eval_batch_size")),
            _fmt(d.get("num_tasks")),
        ]
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print()
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    print(row("ARM", arm))
    print(row("TRM-Opt", trm))
    print()

    # -- Headline deltas --------------------------------------------------
    a_cpt, t_cpt = arm.get("cost_per_task_usd"), trm.get("cost_per_task_usd")
    if a_cpt and t_cpt:
        print(f"Cost:   ARM ${a_cpt:.5f}/task vs TRM-Opt ${t_cpt:.5f}/task "
              f"=> ARM {t_cpt / a_cpt:.2f}x cheaper per task")
    a_p1, t_p1 = arm.get("pass@1"), trm.get("pass@1")
    if a_p1 is not None and t_p1 is not None:
        print(f"Score:  ARM pass@1 {a_p1:.3f} vs TRM-Opt {t_p1:.3f} "
              f"=> +{(a_p1 - t_p1) * 100:.1f} pp")

    # -- Invariant guardrails (controlled comparison) ---------------------
    warnings = []
    if arm.get("gpu") != trm.get("gpu"):
        warnings.append(f"different GPUs: ARM '{arm.get('gpu')}' vs TRM-Opt '{trm.get('gpu')}'")
    if arm.get("eval_batch_size") != trm.get("eval_batch_size"):
        warnings.append(f"different batch size: {arm.get('eval_batch_size')} vs {trm.get('eval_batch_size')}")
    if arm.get("num_tasks") != trm.get("num_tasks"):
        warnings.append(f"different task count: {arm.get('num_tasks')} vs {trm.get('num_tasks')}")
    if arm.get("usd_per_hour") != trm.get("usd_per_hour"):
        warnings.append(f"different $/hour: {arm.get('usd_per_hour')} vs {trm.get('usd_per_hour')}")

    print()
    if warnings:
        print("WARNING: comparison is NOT controlled:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("OK: same GPU, batch size, task count, and rate. Controlled comparison.")


if __name__ == "__main__":
    main()
