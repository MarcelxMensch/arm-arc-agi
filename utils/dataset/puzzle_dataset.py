"""Puzzle dataset loader for sharded JSONL datasets."""

from typing import Dict, Any, List, Optional, Iterator, Tuple
from dataclasses import dataclass
import json
import yaml
import logging
from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

# ARC grid constants
ARC_MAX_GRID_SIZE = 30
PAD_ID = 0
EOS_ID = 1
VOCAB_OFFSET = 2  # PAD=0, EOS=1, then colors 0-9 become 2-11
MAX_DEMOS = 5  # Maximum number of demonstration pairs per task
# Match losses.IGNORE_LABEL_ID when we want to ignore certain positions in the loss
IGNORE_LABEL_ID = -100


def _sample_legacy_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Sample one legacy ARC batch, preserving puzzle-group sampling behavior."""
    batch_rows: List[np.ndarray] = []
    batch_puzzle_indices: List[np.ndarray] = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        group_id = int(group_order[start_index])
        puzzle_id = int(rng.integers(group_indices[group_id], group_indices[group_id + 1]))
        start_index += 1

        puzzle_start = int(puzzle_indices[puzzle_id])
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)
        append_size = min(puzzle_size, global_batch_size - current_size)

        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int64))
        batch_rows.append(puzzle_start + rng.choice(puzzle_size, append_size, replace=False))
        current_size += append_size

    return start_index, np.concatenate(batch_rows), np.concatenate(batch_puzzle_indices)


@dataclass
class PuzzleDatasetMetadata:
    """Metadata for a puzzle dataset."""
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: List[str]
    pad_id: Optional[int] = None
    ignore_label_id: Optional[int] = None
    blank_identifier_id: Optional[int] = None
    # Soft prompt table sizes (from meta.yaml when dataset was built with --write-softprompt-fields)
    num_task_identifiers: Optional[int] = None
    num_color_identifiers: Optional[int] = None


@dataclass
class PuzzleDatasetConfig:
    """Configuration for puzzle dataset."""
    dataset_paths: List[str]
    seed: int = 0
    rank: int = 0
    num_replicas: int = 1
    epochs_per_iter: int = 1
    global_batch_size: int = 32
    test_set_mode: bool = False


class PuzzleDataset(IterableDataset):
    """
    Iterable dataset for loading puzzle tasks from sharded JSONL files.

    Yields batches of tasks with:
    - inputs: Flattened input grids (B, seq_len)
    - labels: Flattened output grids (B, seq_len)
    - puzzle_identifiers: Task IDs mapped to integers (B,)

    When the dataset was built with the new per-example format the batch also
    contains grid-encoder fields:
    - demo_inputs: (B, MAX_DEMOS, 30, 30) encoded demo input grids
    - demo_outputs: (B, MAX_DEMOS, 30, 30) encoded demo output grids
    - num_demos: (B,) number of valid demos per example
    - color_maps: (B, 10) color permutation map
    - transform_ids: (B,) dihedral transform IDs
    """

    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        self.config = config
        self.split = split

        # Load all shard info from all dataset paths.
        # If dataset has train/test subdirs (each with meta.yaml + index.jsonl + shards),
        # resolve path to dataset_path / split; otherwise support legacy arc-2-aug
        # layout (dataset.json + all__*.npy).
        self.shards: List[Dict[str, Any]] = []
        self.task_id_to_idx: Dict[str, int] = {}
        self._legacy_mode = False
        self._legacy_root: Optional[Path] = None
        self._legacy_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._legacy_iters = 0
        num_task_identifiers: Optional[int] = None
        num_color_identifiers: Optional[int] = None

        # Composite softprompt lookup tables (built from identifiers.json)
        self._task_id_lookup: Optional[np.ndarray] = None
        self._transform_id_lookup: Optional[np.ndarray] = None
        self._color_map_lookup: Optional[np.ndarray] = None
        self._num_task_identifiers_from_ident: Optional[int] = None

        total_tasks = 0
        legacy_roots: List[Path] = []
        for dataset_path in config.dataset_paths:
            dataset_path = Path(dataset_path)
            split_dir = dataset_path / split

            # New sharded layout.
            root = split_dir if split_dir.is_dir() and (split_dir / "meta.yaml").exists() else dataset_path
            meta_path = root / "meta.yaml"
            index_path = root / "index.jsonl"
            if meta_path.exists() and index_path.exists():
                with open(meta_path) as f:
                    meta = yaml.safe_load(f)
                if meta:
                    if meta.get("num_task_identifiers") is not None:
                        num_task_identifiers = meta["num_task_identifiers"]
                    if meta.get("num_color_identifiers") is not None:
                        num_color_identifiers = meta["num_color_identifiers"]

                with open(index_path) as f:
                    for line in f:
                        entry = json.loads(line)
                        shard_path = root / entry["shard_path"]
                        self.shards.append({
                            "path": shard_path,
                            "num_tasks": entry["num_tasks"],
                            "offset": total_tasks,
                        })
                        total_tasks += entry["num_tasks"]
                continue

            # Legacy layout (dataset.json + all__*.npy); support both root or root/split.
            legacy_root = None
            for candidate in (split_dir, dataset_path):
                if (candidate / "dataset.json").exists():
                    legacy_root = candidate
                    break
            if legacy_root is not None:
                legacy_roots.append(legacy_root)
                continue

            raise FileNotFoundError(
                f"Could not find dataset at {dataset_path} for split '{split}'. "
                "Expected either meta.yaml + index.jsonl (new sharded layout) or "
                "dataset.json + all__*.npy (legacy arc-2-aug layout)."
            )

        if self.shards and legacy_roots:
            raise ValueError("Mixing sharded and legacy dataset layouts in one run is not supported")

        if legacy_roots:
            if len(legacy_roots) != 1:
                raise ValueError("Legacy dataset mode currently supports a single dataset path")
            self._legacy_mode = True
            self._legacy_root = legacy_roots[0]
            self.metadata = self._load_legacy_metadata(self._legacy_root)
            # Build composite softprompt lookups from identifiers.json if available
            # identifiers.json lives at the dataset root, which may be the parent of legacy_root
            ident_path = self._legacy_root / "identifiers.json"
            if not ident_path.is_file():
                ident_path = self._legacy_root.parent / "identifiers.json"
            if ident_path.is_file():
                self._build_softprompt_lookups(str(ident_path))
                if self._num_task_identifiers_from_ident is not None:
                    self.metadata.num_task_identifiers = self._num_task_identifiers_from_ident
            logger.info(
                "Loaded legacy dataset at %s with %d set(s), %d examples",
                self._legacy_root,
                len(self.metadata.sets),
                self.metadata.total_puzzles,
            )
        else:
            self.metadata = PuzzleDatasetMetadata(
                vocab_size=VOCAB_OFFSET + 10,  # PAD, EOS, colors 0-9
                seq_len=ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE,
                num_puzzle_identifiers=total_tasks + 1,  # +1 for blank
                total_groups=len(self.shards),
                mean_puzzle_examples=total_tasks / max(len(self.shards), 1),
                total_puzzles=total_tasks,
                sets=[split],
                num_task_identifiers=num_task_identifiers,
                num_color_identifiers=num_color_identifiers,
            )
            logger.info(f"Loaded {len(self.shards)} shards with {total_tasks} total tasks")

    def _build_softprompt_lookups(self, ident_path: str) -> None:
        """Parse identifiers.json to build task/transform/color lookup arrays."""
        with open(ident_path) as f:
            identifiers = json.load(f)  # list of strings
        n = len(identifiers)
        task_names: Dict[str, int] = {}
        self._task_id_lookup = np.zeros(n, dtype=np.int32)
        self._transform_id_lookup = np.zeros(n, dtype=np.int32)
        self._color_map_lookup = np.zeros((n, 10), dtype=np.int32)
        for i, ident in enumerate(identifiers):
            if ident == "<blank>" or "|||" not in ident:
                continue
            parts = ident.split("|||")
            task_name = parts[0]
            if task_name not in task_names:
                task_names[task_name] = len(task_names)
            self._task_id_lookup[i] = task_names[task_name]
            self._transform_id_lookup[i] = int(parts[1][1:])  # strip 't' prefix
            self._color_map_lookup[i] = [int(c) for c in parts[2]]
        self._num_task_identifiers_from_ident = len(task_names)
        logger.info("Built composite softprompt lookups: %d task identifiers from %d entries", len(task_names), n)

    def _load_legacy_metadata(self, root: Path) -> PuzzleDatasetMetadata:
        """Load metadata for legacy arc-2-aug datasets."""
        with open(root / "dataset.json") as f:
            legacy_meta = json.load(f)

        sets = legacy_meta.get("sets", ["all"])
        if not isinstance(sets, list) or not sets:
            sets = ["all"]

        total_examples = 0
        total_groups = 0
        for set_name in sets:
            req_files = (
                root / f"{set_name}__inputs.npy",
                root / f"{set_name}__labels.npy",
                root / f"{set_name}__puzzle_identifiers.npy",
                root / f"{set_name}__puzzle_indices.npy",
                root / f"{set_name}__group_indices.npy",
            )
            missing = [str(p) for p in req_files if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Legacy dataset split '{set_name}' is missing files: {missing}"
                )
            total_examples += int(np.load(req_files[0], mmap_mode="r").shape[0])
            total_groups += int(np.load(req_files[4], mmap_mode="r").shape[0] - 1)

        return PuzzleDatasetMetadata(
            vocab_size=int(legacy_meta.get("vocab_size", VOCAB_OFFSET + 10)),
            seq_len=int(legacy_meta.get("seq_len", ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE)),
            num_puzzle_identifiers=int(legacy_meta.get("num_puzzle_identifiers", total_examples + 1)),
            total_groups=int(legacy_meta.get("total_groups", total_groups)),
            mean_puzzle_examples=float(
                legacy_meta.get("mean_puzzle_examples", total_examples / max(total_groups, 1))
            ),
            total_puzzles=total_examples,
            sets=sets,
            pad_id=int(legacy_meta.get("pad_id", PAD_ID)),
            ignore_label_id=legacy_meta.get("ignore_label_id"),
            blank_identifier_id=int(legacy_meta.get("blank_identifier_id", 0)),
        )

    def _lazy_load_legacy(self) -> None:
        if self._legacy_data is not None:
            return
        assert self._legacy_root is not None

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
        }
        self._legacy_data = {}
        for set_name in self.metadata.sets:
            self._legacy_data[set_name] = {
                field_name: np.load(
                    self._legacy_root / f"{set_name}__{field_name}.npy",
                    mmap_mode=mmap_mode,
                )
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    def _collate_legacy_batch(self, batch: Dict[str, np.ndarray], local_batch_size: int) -> Dict[str, torch.Tensor]:
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == int(self.metadata.ignore_label_id)] = IGNORE_LABEL_ID

        if batch["puzzle_identifiers"].size < local_batch_size:
            pad_size = local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": int(self.metadata.pad_id if self.metadata.pad_id is not None else PAD_ID),
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": int(
                    self.metadata.blank_identifier_id
                    if self.metadata.blank_identifier_id is not None
                    else 0
                ),
            }
            batch = {
                k: np.pad(
                    v,
                    ((0, pad_size),) + ((0, 0),) * (v.ndim - 1),
                    constant_values=pad_values[k],
                )
                for k, v in batch.items()
            }

        result = {k: torch.from_numpy(v) for k, v in batch.items()}

        # Enrich with composite softprompt fields if lookups are available
        if self._task_id_lookup is not None:
            pids = batch["puzzle_identifiers"]  # numpy int32
            result["task_identifiers"] = torch.from_numpy(self._task_id_lookup[pids])
            result["transform_ids"] = torch.from_numpy(self._transform_id_lookup[pids])
            result["color_maps"] = torch.from_numpy(self._color_map_lookup[pids])

        return result

    def _iter_legacy_test(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        assert self._legacy_data is not None
        local_batch_size = self.config.global_batch_size // self.config.num_replicas

        for _set_name, dataset in self._legacy_data.items():
            total_examples = int(len(dataset["inputs"]))
            start_index = 0
            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.config.global_batch_size)

                local_start = start_index + self.config.rank * local_batch_size
                local_end = min(start_index + (self.config.rank + 1) * local_batch_size, end_index)

                puzzle_indices = []
                puzzle_index = int(np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1)
                for i in range(local_start, local_end):
                    while (
                        puzzle_index + 1 < len(dataset["puzzle_indices"])
                        and i >= dataset["puzzle_indices"][puzzle_index + 1]
                    ):
                        puzzle_index += 1
                    puzzle_indices.append(puzzle_index)

                batch = self._collate_legacy_batch(
                    {
                        "inputs": dataset["inputs"][local_start:local_end],
                        "labels": dataset["labels"][local_start:local_end],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices],
                    },
                    local_batch_size=local_batch_size,
                )
                yield self.split, batch, end_index - start_index
                start_index += self.config.global_batch_size

    def _iter_legacy_train(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        assert self._legacy_data is not None
        local_batch_size = self.config.global_batch_size // self.config.num_replicas

        for _set_name, dataset in self._legacy_data.items():
            self._legacy_iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._legacy_iters))
            group_order = np.concatenate(
                [rng.permutation(dataset["group_indices"].size - 1) for _ in range(self.config.epochs_per_iter)]
            )
            start_index = 0

            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_legacy_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )
                global_effective_batch_size = int(batch_puzzle_indices.size)

                if global_effective_batch_size < self.config.global_batch_size:
                    break

                lo = self.config.rank * local_batch_size
                hi = (self.config.rank + 1) * local_batch_size
                batch_indices = batch_indices[lo:hi]
                batch_puzzle_indices = batch_puzzle_indices[lo:hi]

                batch = self._collate_legacy_batch(
                    {
                        "inputs": dataset["inputs"][batch_indices],
                        "labels": dataset["labels"][batch_indices],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices],
                    },
                    local_batch_size=local_batch_size,
                )
                yield self.split, batch, global_effective_batch_size

    # ------------------------------------------------------------------
    # Grid encoding helpers
    # ------------------------------------------------------------------

    def _grid_to_seq(self, grid: List[List[int]]) -> torch.Tensor:
        """Convert a 2D grid to a flattened sequence with padding and EOS."""
        seq = torch.zeros(self.metadata.seq_len, dtype=torch.long)

        if not grid:
            return seq

        height = len(grid)
        width = len(grid[0]) if grid else 0

        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                idx = i * ARC_MAX_GRID_SIZE + j
                if idx < self.metadata.seq_len:
                    seq[idx] = val + VOCAB_OFFSET

        eos_row = height
        eos_col = width
        if eos_row < ARC_MAX_GRID_SIZE:
            for j in range(eos_col):
                idx = eos_row * ARC_MAX_GRID_SIZE + j
                if idx < self.metadata.seq_len:
                    seq[idx] = EOS_ID
        if eos_col < ARC_MAX_GRID_SIZE:
            for i in range(eos_row):
                idx = i * ARC_MAX_GRID_SIZE + eos_col
                if idx < self.metadata.seq_len:
                    seq[idx] = EOS_ID

        return seq

    def _grid_to_2d_tensor(self, grid: List[List[int]]) -> torch.Tensor:
        """Encode a 2D grid into a (30, 30) tensor using VOCAB_OFFSET encoding.

        Colors are stored as ``color_value + VOCAB_OFFSET`` (2-11).  Spatial
        padding is ``PAD_ID=0``, distinct from any color.  EOS markers are
        placed at the grid boundary, same as ``_grid_to_seq``.
        """
        t = torch.zeros(ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE, dtype=torch.long)
        if not grid:
            return t

        height = len(grid)
        width = len(grid[0]) if grid else 0

        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if i < ARC_MAX_GRID_SIZE and j < ARC_MAX_GRID_SIZE:
                    t[i, j] = val + VOCAB_OFFSET

        # EOS markers
        if height < ARC_MAX_GRID_SIZE:
            for j in range(min(width, ARC_MAX_GRID_SIZE)):
                t[height, j] = EOS_ID
        if width < ARC_MAX_GRID_SIZE:
            for i in range(min(height, ARC_MAX_GRID_SIZE)):
                t[i, width] = EOS_ID

        return t

    # ------------------------------------------------------------------
    # Task processing
    # ------------------------------------------------------------------

    def _load_shard(self, shard_info: Dict) -> List[Dict[str, Any]]:
        """Load all tasks from a shard file."""
        tasks = []
        with open(shard_info["path"]) as f:
            for line in f:
                task = json.loads(line)
                tasks.append(task)
        return tasks

    def _process_task(self, task: Dict[str, Any], task_idx: int) -> Dict[str, torch.Tensor]:
        """Process a single task row into tensors.

        Supports three JSONL row formats:

        1. **Per-example format** (new, from ``build_arc_agi_2_dataset.py``):
           Top-level ``input``/``output`` with ``demo_inputs``/``demo_outputs``.

        2. **Task-level format** (from concept_4 / arc_agi_1 builders):
           ``train`` and ``test`` arrays of ``{input, output}`` dicts.
           The first test example is used as the prediction target; train
           examples are surfaced as demo grids for the grid encoder.

        3. **Legacy flat format**: ``examples`` list (oldest shards).
        """
        input_grid = task.get("input", [])
        output_grid = task.get("output", [])
        demo_examples: Optional[List[Dict[str, Any]]] = None

        # --- Format 2: task-level rows with "train" / "test" arrays --------
        if not input_grid and not output_grid and ("train" in task or "test" in task):
            train_examples_raw = task.get("train", [])
            test_examples_raw = task.get("test", [])
            # Use first test example as prediction target (fallback to first train)
            target_list = test_examples_raw or train_examples_raw
            if target_list:
                input_grid = target_list[0].get("input", [])
                output_grid = target_list[0].get("output", [])
            # Train examples become demo grids
            if train_examples_raw:
                demo_examples = train_examples_raw

        # --- Format 3: legacy "examples" list --------
        if not input_grid and not output_grid:
            examples = task.get("examples", [])
            if examples:
                input_grid = examples[0].get("input", [])
                output_grid = examples[0].get("output", [])

        if "puzzle_identifier" in task:
            puzzle_id = int(task["puzzle_identifier"])
        else:
            puzzle_id = task_idx + 1

        labels = self._grid_to_seq(output_grid)
        result: Dict[str, Any] = {
            "inputs": self._grid_to_seq(input_grid),
            "labels": labels,
            "puzzle_identifiers": torch.tensor(puzzle_id, dtype=torch.long),
        }

        # --- Demo grids (grid encoder pipeline) ----------------------------
        # Prefer explicit "demo_inputs"/"demo_outputs" (per-example format);
        # fall back to demo_examples extracted from task-level "train" array.
        demo_inputs_raw: Optional[List] = None
        demo_outputs_raw: Optional[List] = None

        if "demo_inputs" in task and "demo_outputs" in task:
            demo_inputs_raw = task["demo_inputs"]
            demo_outputs_raw = task["demo_outputs"]
        elif demo_examples is not None:
            demo_inputs_raw = [ex.get("input", []) for ex in demo_examples]
            demo_outputs_raw = [ex.get("output", []) for ex in demo_examples]

        if demo_inputs_raw is not None and demo_outputs_raw is not None:
            n_demos = min(len(demo_inputs_raw), MAX_DEMOS)
            demo_in_t = torch.zeros(MAX_DEMOS, ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE, dtype=torch.long)
            demo_out_t = torch.zeros(MAX_DEMOS, ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE, dtype=torch.long)
            for d in range(n_demos):
                demo_in_t[d] = self._grid_to_2d_tensor(demo_inputs_raw[d])
                demo_out_t[d] = self._grid_to_2d_tensor(demo_outputs_raw[d])

            result["demo_inputs"] = demo_in_t
            result["demo_outputs"] = demo_out_t
            result["num_demos"] = torch.tensor(n_demos, dtype=torch.long)

            # So the new model (grid encoder softprompt) can use this path: provide identity
            # transform/color when the dataset does not (e.g. concept task-level format).
            if "color_map" in task:
                result["color_maps"] = torch.tensor(task["color_map"], dtype=torch.long)
            else:
                result["color_maps"] = torch.arange(10, dtype=torch.long)

            if "transform_id" in task:
                result["transform_ids"] = torch.tensor(int(task["transform_id"]), dtype=torch.long)
            else:
                result["transform_ids"] = torch.tensor(0, dtype=torch.long)

        if "task_identifier" in task:
            result["task_identifiers"] = torch.tensor(int(task["task_identifier"]), dtype=torch.long)

        # Legacy soft prompt fields (old dataset format)
        if "transform_identifier" in task:
            result["transform_identifiers"] = torch.tensor(int(task["transform_identifier"]), dtype=torch.long)
        if "color_identifier" in task:
            result["color_identifiers"] = torch.tensor(int(task["color_identifier"]), dtype=torch.long)

        return result

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        """Iterate over batches of tasks."""
        if self._legacy_mode:
            self._lazy_load_legacy()
            if self.config.test_set_mode:
                yield from self._iter_legacy_test()
            else:
                yield from self._iter_legacy_train()
            return

        rng = random.Random(self.config.seed + self.config.rank)
        batch_size = self.config.global_batch_size // self.config.num_replicas

        for _epoch in range(self.config.epochs_per_iter):
            shard_indices = list(range(len(self.shards)))
            if not self.config.test_set_mode:
                rng.shuffle(shard_indices)

            shard_indices = shard_indices[self.config.rank::self.config.num_replicas]

            for shard_idx in shard_indices:
                shard_info = self.shards[shard_idx]
                tasks = self._load_shard(shard_info)

                if not self.config.test_set_mode:
                    rng.shuffle(tasks)

                for i in range(0, len(tasks), batch_size):
                    batch_tasks = tasks[i:i + batch_size]

                    if len(batch_tasks) < batch_size and not self.config.test_set_mode:
                        continue

                    required_keys = {"inputs", "labels", "puzzle_identifiers"}
                    batch: Dict[str, Any] = {k: [] for k in required_keys}
                    processed_tasks = []

                    for j, task in enumerate(batch_tasks):
                        task_idx = shard_info["offset"] + i + j
                        processed = self._process_task(task, task_idx)
                        processed_tasks.append(processed)

                    # Only add optional fields if present in ALL tasks of this batch
                    optional_fields: set = set()
                    if processed_tasks:
                        candidate_keys = set(processed_tasks[0].keys()) - required_keys
                        for key in candidate_keys:
                            if all(key in p for p in processed_tasks):
                                optional_fields.add(key)

                    for field in optional_fields:
                        batch[field] = []

                    for processed in processed_tasks:
                        for k in batch:
                            batch[k].append(processed[k])

                    # Stack tensors
                    batch = {k: torch.stack(v) for k, v in batch.items()}

                    def _external_task_id(t: Dict[str, Any]) -> Optional[str]:
                        oid = t.get("original_task_id")
                        if oid:
                            return str(oid)
                        tid = t.get("task_id")
                        if tid:
                            return str(tid).split("_")[0] or None
                        return None
                    batch["original_task_ids"] = [_external_task_id(t) for t in batch_tasks]

                    yield self.split, batch, len(batch_tasks) * self.config.num_replicas
