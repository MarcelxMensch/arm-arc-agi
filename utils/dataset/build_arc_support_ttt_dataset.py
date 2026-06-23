"""Build ARC support-query episodes for trm-abstraction-support-ttt.

This builder intentionally writes a new dataset layout instead of modifying the
legacy per-example ARC format. Each row is an episode:

  support_inputs/support_outputs: K support pairs from one base problem
  inputs/labels: one query pair from the same base problem

Splitting is by base problem id, so all augmentations and all query episodes of
the same problem stay entirely in train or test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.dataset.common import PuzzleDatasetMetadata, dihedral_transform


ARC_MAX_GRID_SIZE = 30
PAD_ID = 0
EOS_ID = 1
VOCAB_SIZE = 12
BLANK_IDENTIFIER_ID = 0
PUZZLE_ID_SEPARATOR = "|||"


@dataclass(frozen=True)
class Example:
    input: np.ndarray
    output: np.ndarray


@dataclass(frozen=True)
class RawTask:
    base_id: str
    subset: str
    train: Tuple[Example, ...]
    test: Tuple[Example, ...]


@dataclass(frozen=True)
class QuerySpec:
    source: str  # "train" or "test"
    index: int


@dataclass(frozen=True)
class AugPlan:
    transform_id: int
    color_map: np.ndarray


def arc_grid_to_np(grid: Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.array(grid, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"ARC grid must be 2D, got shape={arr.shape}")
    if arr.shape[0] > ARC_MAX_GRID_SIZE or arr.shape[1] > ARC_MAX_GRID_SIZE:
        raise ValueError(f"ARC grid too large for 30x30 canvas: shape={arr.shape}")
    if not np.all((arr >= 0) & (arr <= 9)):
        raise ValueError("ARC grid values must be in [0, 9]")
    return arr


def grid_to_seq(grid: np.ndarray) -> np.ndarray:
    """Pad one grid to 30x30 and encode PAD/EOS/colors as 0/1/2..11."""
    nrow, ncol = grid.shape
    out = np.zeros((ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE), dtype=np.uint8)
    out[:nrow, :ncol] = grid.astype(np.uint8) + 2
    if nrow < ARC_MAX_GRID_SIZE:
        out[nrow, :ncol] = EOS_ID
    if ncol < ARC_MAX_GRID_SIZE:
        out[:nrow, ncol] = EOS_ID
    return out.reshape(-1)


def color_permutation(rng: np.random.Generator) -> np.ndarray:
    """Permutation over ARC colors with black/background 0 fixed."""
    return np.concatenate(
        [np.array([0], dtype=np.uint8), rng.permutation(np.arange(1, 10, dtype=np.uint8))]
    )


def apply_aug(grid: np.ndarray, plan: AugPlan) -> np.ndarray:
    transformed = dihedral_transform(grid, int(plan.transform_id))
    return plan.color_map[transformed].astype(np.uint8, copy=False)


def identity_plan() -> AugPlan:
    return AugPlan(transform_id=0, color_map=np.arange(10, dtype=np.uint8))


def random_plan(rng: np.random.Generator) -> AugPlan:
    return AugPlan(transform_id=int(rng.integers(0, 8)), color_map=color_permutation(rng))


def stable_fraction(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def split_for_task(
    task: RawTask,
    seed: int,
    train_ratio: float,
    force_test_subsets: set[str],
) -> str:
    if task.subset in force_test_subsets:
        return "test"
    return "train" if stable_fraction(seed, task.base_id) < train_ratio else "test"


def load_raw_tasks(
    input_file_prefix: str,
    subsets: Sequence[str],
    include_official_test_queries: bool,
) -> List[RawTask]:
    tasks: List[RawTask] = []
    for subset in subsets:
        challenges_path = Path(f"{input_file_prefix}_{subset}_challenges.json")
        solutions_path = Path(f"{input_file_prefix}_{subset}_solutions.json")
        if not challenges_path.exists():
            raise FileNotFoundError(f"Missing challenges file: {challenges_path}")

        with open(challenges_path, "r") as f:
            puzzles = json.load(f)

        solutions = None
        if include_official_test_queries and solutions_path.exists():
            with open(solutions_path, "r") as f:
                solutions = json.load(f)

        for task_name, puzzle in sorted(puzzles.items()):
            train_examples = []
            for ex in puzzle.get("train", []):
                if "input" in ex and "output" in ex:
                    train_examples.append(Example(arc_grid_to_np(ex["input"]), arc_grid_to_np(ex["output"])))

            test_examples = []
            raw_tests = puzzle.get("test", [])
            for idx, ex in enumerate(raw_tests):
                output_grid = ex.get("output")
                if output_grid is None and solutions is not None and task_name in solutions:
                    if idx < len(solutions[task_name]):
                        output_grid = solutions[task_name][idx]
                if "input" in ex and output_grid is not None:
                    test_examples.append(Example(arc_grid_to_np(ex["input"]), arc_grid_to_np(output_grid)))

            if not train_examples:
                continue

            tasks.append(
                RawTask(
                    base_id=f"{subset}:{task_name}",
                    subset=subset,
                    train=tuple(train_examples),
                    test=tuple(test_examples),
                )
            )
    return tasks


def query_specs_for_task(
    task: RawTask,
    include_train_leave_one_out: bool,
    include_official_test_queries: bool,
) -> List[QuerySpec]:
    specs: List[QuerySpec] = []
    if include_train_leave_one_out and len(task.train) >= 2:
        specs.extend(QuerySpec("train", i) for i in range(len(task.train)))
    if include_official_test_queries and len(task.train) >= 1:
        specs.extend(QuerySpec("test", i) for i in range(len(task.test)))
    return specs


def support_for_query(task: RawTask, query: QuerySpec) -> Tuple[Example, List[Example]]:
    if query.source == "train":
        query_ex = task.train[query.index]
        support = [ex for i, ex in enumerate(task.train) if i != query.index]
    elif query.source == "test":
        query_ex = task.test[query.index]
        support = list(task.train)
    else:
        raise ValueError(f"Unknown query source: {query.source}")
    return query_ex, support


def choose_support_indices(
    num_support: int,
    max_support_examples: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> np.ndarray:
    indices = np.arange(num_support, dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    if num_support > max_support_examples:
        indices = indices[:max_support_examples]
    return indices


def make_identifier(task: RawTask, aug_idx: int, plan: AugPlan) -> str:
    color = "".join(str(int(x)) for x in plan.color_map.tolist())
    return f"{task.base_id}{PUZZLE_ID_SEPARATOR}aug{aug_idx}{PUZZLE_ID_SEPARATOR}t{plan.transform_id}{PUZZLE_ID_SEPARATOR}{color}"


def count_rows(
    tasks: Sequence[RawTask],
    num_aug: int,
    include_train_leave_one_out: bool,
    include_official_test_queries: bool,
    split_kwargs: Dict[str, object],
) -> Dict[str, int]:
    counts = {"train": 0, "test": 0}
    for task in tasks:
        split = split_for_task(task, **split_kwargs)
        n_queries = len(query_specs_for_task(task, include_train_leave_one_out, include_official_test_queries))
        counts[split] += n_queries * num_aug
    return counts


def allocate_split_arrays(split_dir: Path, count: int, max_support_examples: int) -> Dict[str, np.ndarray]:
    split_dir.mkdir(parents=True, exist_ok=True)
    shape_seq = (count, ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE)
    shape_support = (count, max_support_examples, ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE)
    arrays = {
        "inputs": np.lib.format.open_memmap(split_dir / "all__inputs.npy", mode="w+", dtype=np.uint8, shape=shape_seq),
        "labels": np.lib.format.open_memmap(split_dir / "all__labels.npy", mode="w+", dtype=np.uint8, shape=shape_seq),
        "support_inputs": np.lib.format.open_memmap(split_dir / "all__support_inputs.npy", mode="w+", dtype=np.uint8, shape=shape_support),
        "support_outputs": np.lib.format.open_memmap(split_dir / "all__support_outputs.npy", mode="w+", dtype=np.uint8, shape=shape_support),
        "support_mask": np.lib.format.open_memmap(split_dir / "all__support_mask.npy", mode="w+", dtype=np.uint8, shape=(count, max_support_examples)),
        "puzzle_identifiers": np.lib.format.open_memmap(split_dir / "all__puzzle_identifiers.npy", mode="w+", dtype=np.int32, shape=(count,)),
        "task_identifiers": np.lib.format.open_memmap(split_dir / "all__task_identifiers.npy", mode="w+", dtype=np.int32, shape=(count,)),
        "query_indices": np.lib.format.open_memmap(split_dir / "all__query_indices.npy", mode="w+", dtype=np.int32, shape=(count,)),
        "query_sources": np.lib.format.open_memmap(split_dir / "all__query_sources.npy", mode="w+", dtype=np.int8, shape=(count,)),
        "transform_ids": np.lib.format.open_memmap(split_dir / "all__transform_ids.npy", mode="w+", dtype=np.int32, shape=(count,)),
        "color_maps": np.lib.format.open_memmap(split_dir / "all__color_maps.npy", mode="w+", dtype=np.int32, shape=(count, 10)),
    }
    arrays["support_inputs"][:] = PAD_ID
    arrays["support_outputs"][:] = PAD_ID
    arrays["support_mask"][:] = 0
    return arrays


def write_metadata(
    output_dir: Path,
    split: str,
    count: int,
    num_task_identifiers: int,
    mean_support_examples: float,
) -> None:
    metadata = PuzzleDatasetMetadata(
        seq_len=ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE,
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        ignore_label_id=PAD_ID,
        blank_identifier_id=BLANK_IDENTIFIER_ID,
        num_puzzle_identifiers=count + 1,
        total_groups=count,
        mean_puzzle_examples=mean_support_examples,
        total_puzzles=count,
        sets=["all"],
        num_task_identifiers=num_task_identifiers,
    )
    with open(output_dir / split / "dataset.json", "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)


def flush_arrays(arrays_by_split: Dict[str, Dict[str, np.ndarray]]) -> None:
    for arrays in arrays_by_split.values():
        for arr in arrays.values():
            if hasattr(arr, "flush"):
                arr.flush()


def build_dataset(args: argparse.Namespace) -> None:
    if args.num_aug < 1:
        raise ValueError("--num-aug must be >= 1")
    if args.max_support_examples < 1:
        raise ValueError("--max-support-examples must be >= 1")
    if abs(args.train_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("--train-ratio and --test-ratio must sum to 1.0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    force_test_subsets = {s for s in [args.test_set_name, args.test_set_name2] if s}
    force_test_subsets.update(args.force_test_subset or [])

    raw_tasks = load_raw_tasks(
        args.input_file_prefix,
        args.subsets,
        include_official_test_queries=args.include_official_test_queries,
    )
    if args.max_tasks is not None:
        raw_tasks = raw_tasks[: args.max_tasks]

    split_kwargs = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "force_test_subsets": force_test_subsets,
    }
    row_counts = count_rows(
        raw_tasks,
        args.num_aug,
        args.include_train_leave_one_out,
        args.include_official_test_queries,
        split_kwargs,
    )

    print(f"Loaded base tasks: {len(raw_tasks)}")
    print(f"Force-test subsets: {sorted(force_test_subsets)}")
    print(f"Rows to write: train={row_counts['train']} test={row_counts['test']}")
    if row_counts["train"] == 0:
        raise RuntimeError("No train rows would be written")
    if row_counts["test"] == 0:
        raise RuntimeError("No test rows would be written")

    arrays_by_split = {
        split: allocate_split_arrays(output_dir / split, row_counts[split], args.max_support_examples)
        for split in ("train", "test")
    }

    rng = np.random.default_rng(args.seed)
    task_id_map: Dict[str, int] = {}
    split_task_ids = {"train": set(), "test": set()}
    identifiers = {"train": ["<blank>"], "test": ["<blank>"]}
    write_offsets = {"train": 0, "test": 0}
    support_counts = {"train": [], "test": []}

    for task in raw_tasks:
        split = split_for_task(task, **split_kwargs)
        split_task_ids[split].add(task.base_id)
        if task.base_id not in task_id_map:
            task_id_map[task.base_id] = len(task_id_map)
        task_identifier = task_id_map[task.base_id]

        query_specs = query_specs_for_task(
            task,
            args.include_train_leave_one_out,
            args.include_official_test_queries,
        )
        if not query_specs:
            continue

        plans = [identity_plan()]
        plans.extend(random_plan(rng) for _ in range(args.num_aug - 1))

        for aug_idx, plan in enumerate(plans):
            identifier = make_identifier(task, aug_idx, plan)
            puzzle_identifier = len(identifiers[split])
            identifiers[split].append(identifier)

            for query in query_specs:
                row = write_offsets[split]
                arrays = arrays_by_split[split]
                query_ex, support_examples = support_for_query(task, query)
                if not support_examples:
                    continue

                query_input = apply_aug(query_ex.input, plan)
                query_output = apply_aug(query_ex.output, plan)
                arrays["inputs"][row] = grid_to_seq(query_input)
                arrays["labels"][row] = grid_to_seq(query_output)

                support_indices = choose_support_indices(
                    len(support_examples),
                    args.max_support_examples,
                    rng,
                    # Unconditional shuffle: support order must not depend on split,
                    # otherwise train (shuffled) and test (fixed) episodes differ.
                    shuffle=True,
                )
                for support_slot, support_idx in enumerate(support_indices):
                    support_ex = support_examples[int(support_idx)]
                    arrays["support_inputs"][row, support_slot] = grid_to_seq(apply_aug(support_ex.input, plan))
                    arrays["support_outputs"][row, support_slot] = grid_to_seq(apply_aug(support_ex.output, plan))
                    arrays["support_mask"][row, support_slot] = 1

                arrays["puzzle_identifiers"][row] = puzzle_identifier
                arrays["task_identifiers"][row] = task_identifier
                arrays["query_indices"][row] = query.index
                arrays["query_sources"][row] = 0 if query.source == "train" else 1
                arrays["transform_ids"][row] = plan.transform_id
                arrays["color_maps"][row] = plan.color_map.astype(np.int32)

                support_counts[split].append(int(len(support_indices)))
                write_offsets[split] += 1

    for split in ("train", "test"):
        expected = row_counts[split]
        actual = write_offsets[split]
        if actual != expected:
            raise RuntimeError(f"{split}: wrote {actual} rows, expected {expected}")

    flush_arrays(arrays_by_split)

    overlap = split_task_ids["train"] & split_task_ids["test"]
    if overlap:
        raise RuntimeError(f"Train/test base-task overlap detected: {sorted(overlap)[:10]}")

    for split in ("train", "test"):
        mean_support = float(np.mean(support_counts[split])) if support_counts[split] else 0.0
        write_metadata(output_dir, split, row_counts[split], len(task_id_map), mean_support)

    task_identifier_list = [""] * len(task_id_map)
    for task_id, idx in task_id_map.items():
        task_identifier_list[idx] = task_id

    with open(output_dir / "identifiers.json", "w") as f:
        json.dump(identifiers, f, indent=2)
    with open(output_dir / "task_identifiers.json", "w") as f:
        json.dump(task_identifier_list, f, indent=2)
    with open(output_dir / "split_task_ids.json", "w") as f:
        json.dump({k: sorted(v) for k, v in split_task_ids.items()}, f, indent=2)
    with open(output_dir / "build_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Build complete")
    for split in ("train", "test"):
        sc = support_counts[split]
        print(
            f"{split}: rows={row_counts[split]} tasks={len(split_task_ids[split])} "
            f"mean_support={np.mean(sc):.2f} min_support={min(sc)} max_support={max(sc)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build support-query ARC TTT episodes")
    parser.add_argument("--input-file-prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsets", nargs="+", required=True)
    parser.add_argument("--test-set-name", default=None)
    parser.add_argument("--test-set-name2", default=None)
    parser.add_argument("--force-test-subset", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--max-support-examples", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--include-train-leave-one-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-official-test-queries", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
