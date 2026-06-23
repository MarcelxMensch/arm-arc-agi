import os
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from utils.models.losses import IGNORE_LABEL_ID
from utils.dataset.common import PuzzleDatasetMetadata
# Note: `utils.data.arm_collator` is imported lazily inside the ARM branch of
# __init__ so baseline (non-ARM) runs do not hard-require the utils/data
# directory to exist. See abstraction_poe branch Phase 5 cluster regression.

from safetensors.numpy import load_file

from argdantic import ArgParser
from pydantic import BaseModel

def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int

    # --- ARM episode mode (Phase 1 of ARM plan) ---
    # When enabled, each batch item yields K demo (input, output) pairs plus a
    # held-out target (input, output) sampled without replacement from the same
    # puzzle. Puzzles with fewer than k_demos+1 examples are skipped.
    arm_episode_mode: bool = False
    k_demos: int = 2

    # --- Support TTT mode (trm-abstraction-support-ttt) ---
    # New row-based support/query layout:
    # inputs/labels plus support_inputs/support_outputs/support_mask.
    support_ttt_mode: bool = False

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Composite softprompt lookup tables (built from identifiers.json)
        self._task_id_lookup: Optional[np.ndarray] = None
        self._transform_id_lookup: Optional[np.ndarray] = None
        self._color_map_lookup: Optional[np.ndarray] = None
        self._num_task_identifiers_from_ident: Optional[int] = None

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        num_task_identifiers: Optional[int] = None
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
            if current_metadata.num_task_identifiers is not None:
                num_task_identifiers = max(
                    num_task_identifiers or 0,
                    int(current_metadata.num_task_identifiers),
                )

            # Build composite softprompt lookups from identifiers.json if available.
            # Support-TTT datasets store identifiers as split dictionaries and
            # do not use learned puzzle/task softprompt lookup tables.
            ident_path = os.path.join(dataset_path, "identifiers.json")
            if os.path.isfile(ident_path) and not self.config.support_ttt_mode:
                self._build_softprompt_lookups(ident_path)

        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets,
            num_task_identifiers=self._num_task_identifiers_from_ident or num_task_identifiers,
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

        # ARM episode collator + eligible-puzzle cache (lazily initialised per set).
        # Import is deferred so baseline runs never require utils/data/.
        self._arm_collator = None
        self._arm_eligible_cache: Dict[str, np.ndarray] = {}
        if self.config.arm_episode_mode:
            from utils.data.arm_collator import ArmCollator  # local import
            self._arm_collator = ArmCollator(
                k_demos=self.config.k_demos,
                pad_id=self.metadata.pad_id,
                ignore_label_id=self.metadata.ignore_label_id,
            )

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

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _load_dataset_npy(self, dataset_path: str, set_name: str) -> dict:
        """Load from .npy files (all__inputs.npy, all__labels.npy, etc.)."""
        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
        }
        return {
            field_name: np.load(
                os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"),
                mmap_mode=mmap_mode,
            )
            for field_name, mmap_mode in field_mmap_modes.items()
        }

    def _load_support_ttt_dataset_npy(self, dataset_path: str, set_name: str) -> dict:
        """Load support-query TTT episode arrays.

        The large sequence tensors are memory-mapped. Identifier/metadata arrays
        are small enough to load normally, mirroring the legacy loader.
        """
        split_dir = os.path.join(dataset_path, self.split)
        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "support_inputs": "r",
            "support_outputs": "r",
            "support_mask": "r",
            "puzzle_identifiers": None,
            "task_identifiers": None,
            "query_indices": None,
            "query_sources": None,
            "transform_ids": None,
            "color_maps": None,
        }
        return {
            field_name: np.load(
                os.path.join(split_dir, f"{set_name}__{field_name}.npy"),
                mmap_mode=mmap_mode,
            )
            for field_name, mmap_mode in field_mmap_modes.items()
        }

    def _load_dataset_safetensor(self, dataset_path: str) -> dict:
        """Load from all.safetensors (single file with inputs, labels, etc.)."""
        path = os.path.join(dataset_path, self.split, "all.safetensors")
        tensors = load_file(path)
        return {k: tensors[k] for k in ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")}

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        self._data = {}
        for set_name in self.metadata.sets:
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                split_dir = os.path.join(dataset_path, self.split)
                safetensor_path = os.path.join(split_dir, "all.safetensors")
                npy_path = os.path.join(split_dir, f"{set_name}__inputs.npy")
                support_path = os.path.join(split_dir, f"{set_name}__support_inputs.npy")
                if self.config.support_ttt_mode:
                    if os.path.isfile(npy_path) and os.path.isfile(support_path):
                        self._data[set_name_] = self._load_support_ttt_dataset_npy(dataset_path, set_name)
                    else:
                        raise FileNotFoundError(
                            f"Support-TTT dataset requires {npy_path} and {support_path}"
                        )
                elif os.path.isfile(safetensor_path):
                    self._data[set_name_] = self._load_dataset_safetensor(dataset_path)
                elif os.path.isfile(npy_path):
                    self._data[set_name_] = self._load_dataset_npy(dataset_path, set_name)
                else:
                    raise FileNotFoundError(
                        f"Neither all.safetensors nor {set_name}__inputs.npy found in {split_dir}"
                    )


    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        result = {k: torch.from_numpy(v) for k, v in batch.items()}

        # Enrich with composite softprompt fields if lookups are available
        if self._task_id_lookup is not None:
            pids = batch["puzzle_identifiers"]  # numpy int32
            result["task_identifiers"] = torch.from_numpy(self._task_id_lookup[pids])
            result["transform_ids"] = torch.from_numpy(self._transform_id_lookup[pids])
            result["color_maps"] = torch.from_numpy(self._color_map_lookup[pids])

        return result

    def _collate_support_ttt_batch(self, batch):
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # COLOR_PERM_AUG: per-episode random permutation of the 10 ARC colours.
        # Permutes only colour-range tokens; pad (pad_id) and EOS (1) are kept.
        # The SAME permutation is applied to inputs, labels, support_inputs,
        # support_outputs of the same row, so the input→output rule is preserved.
        # Token layout assumed: rearc-padtoken uses color_offset=2 (colours = 2..11).
        if os.environ.get("COLOR_PERM_AUG") == "1":
            color_offset = int(os.environ.get("COLOR_OFFSET", "2"))
            n_rows = batch["inputs"].shape[0]
            pad_id = int(self.metadata.pad_id or 0)
            for r in range(n_rows):
                perm = np.random.permutation(10).astype(np.int32)
                # Build full token-level remap: identity by default, only colours change
                vocab = int(self.metadata.vocab_size or (color_offset + 10))
                token_map = np.arange(vocab, dtype=np.int32)
                for c in range(10):
                    token_map[color_offset + c] = color_offset + perm[c]
                # Important: do NOT remap IGNORE_LABEL_ID (e.g. -100) — it's not in [0, vocab).
                # Do NOT remap pad_id or EOS (1) since they're outside [color_offset, color_offset+10).
                for key in ("inputs", "labels", "support_inputs", "support_outputs"):
                    if key not in batch: continue
                    arr = batch[key][r]
                    mask = (arr >= color_offset) & (arr < color_offset + 10)
                    if mask.any():
                        arr_remapped = arr.copy()
                        arr_remapped[mask] = token_map[arr[mask]]
                        batch[key][r] = arr_remapped

        if batch["inputs"].shape[0] < self.local_batch_size:
            pad_size = self.local_batch_size - batch["inputs"].shape[0]
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "support_inputs": self.metadata.pad_id,
                "support_outputs": self.metadata.pad_id,
                "support_mask": 0,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
                "task_identifiers": 0,
                "query_indices": 0,
                "query_sources": 0,
                "transform_ids": 0,
                "color_maps": 0,
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
        result["support_mask"] = result["support_mask"].to(torch.bool)
        return result
    
    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                })

                yield set_name, batch, global_effective_batch_size

    def _iter_support_ttt_test(self):
        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = len(dataset["inputs"])
            start_index = 0
            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)

                batch = self._collate_support_ttt_batch({
                    k: v[local_start:local_end]
                    for k, v in dataset.items()
                })
                yield set_name, batch, end_index - start_index
                start_index += self.config.global_batch_size

    def _iter_support_ttt_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))
            total_examples = len(dataset["inputs"])

            for _epoch in range(self.config.epochs_per_iter):
                order = rng.permutation(total_examples)
                start_index = 0

                while start_index < total_examples:
                    end_index = start_index + self.config.global_batch_size
                    if end_index > total_examples:
                        break

                    global_indices = order[start_index:end_index]
                    local_indices = global_indices[
                        self.config.rank * self.local_batch_size:
                        (self.config.rank + 1) * self.local_batch_size
                    ]

                    batch = self._collate_support_ttt_batch({
                        k: v[local_indices]
                        for k, v in dataset.items()
                    })
                    yield set_name, batch, self.config.global_batch_size
                    start_index = end_index
                
    # ------------------------------------------------------------------
    # ARM episode iteration
    # ------------------------------------------------------------------

    def _arm_eligible_for(self, set_name: str, dataset: dict) -> np.ndarray:
        cache = self._arm_eligible_cache.get(set_name)
        if cache is None:
            assert self._arm_collator is not None
            cache = self._arm_collator.eligible_puzzle_ids(dataset["puzzle_indices"])
            self._arm_eligible_cache[set_name] = cache
        return cache

    def _iter_arm_train(self):
        assert self._arm_collator is not None
        for set_name, dataset in self._data.items():  # type: ignore
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            eligible = self._arm_eligible_for(set_name, dataset)
            if eligible.size == 0:
                continue

            # Mirror _iter_train's epoch semantics: yield roughly one
            # "puzzle-pass" per epochs_per_iter.
            steps_per_epoch = max(1, eligible.size // self.local_batch_size)
            total_steps = steps_per_epoch * self.config.epochs_per_iter

            for _ in range(total_steps):
                batch = self._arm_collator.collate(
                    inputs=dataset["inputs"],
                    labels=dataset["labels"],
                    puzzle_indices=dataset["puzzle_indices"],
                    puzzle_identifiers=dataset["puzzle_identifiers"],
                    rng=rng,
                    batch_size=self.local_batch_size,
                    eligible=eligible,
                )
                yield set_name, batch.as_dict(), self.config.global_batch_size

    # ------------------------------------------------------------------
    # Train-split demo lookup for eval
    # ------------------------------------------------------------------

    def _lazy_load_arm_eval_demos(self):
        """Load the train-split raw data alongside the current (test) split.

        ARC-AGI's `test` split contains only the test pairs of each task
        (typically 1 example per puzzle), so the old "first K in same puzzle
        as demos" approach filters every puzzle out. For eval we need to
        pull demos from the task's TRAIN pairs, which live under
        `<dataset_path>/train/` with the SAME puzzle_identifier. This method
        loads that train-split data once and builds a
        `puzzle_identifier → (set_name, train_puzzle_idx)` map.
        """
        if getattr(self, "_arm_demo_data", None) is not None:
            return
        self._arm_demo_data: Dict[str, dict] = {}
        self._arm_demo_lookup: Dict[int, tuple] = {}

        for set_name in self.metadata.sets:
            for i, dataset_path in enumerate(self.config.dataset_paths):
                set_name_ = set_name if i == 0 else f"{set_name}{i}"
                split_dir = os.path.join(dataset_path, "train")
                safetensor_path = os.path.join(split_dir, "all.safetensors")
                npy_path = os.path.join(split_dir, f"{set_name}__inputs.npy")
                if os.path.isfile(safetensor_path):
                    td = {k: load_file(safetensor_path)[k]
                          for k in ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")}
                elif os.path.isfile(npy_path):
                    td = {
                        field: np.load(
                            os.path.join(split_dir, f"{set_name}__{field}.npy"),
                            mmap_mode=("r" if field in ("inputs", "labels") else None),
                        )
                        for field in (
                            "inputs",
                            "labels",
                            "puzzle_identifiers",
                            "puzzle_indices",
                            "group_indices",
                        )
                    }
                else:
                    continue

                self._arm_demo_data[set_name_] = td
                demo_pids = td["puzzle_identifiers"]
                demo_pi = td["puzzle_indices"]
                k_min = self._arm_collator.min_puzzle_size  # type: ignore
                for puzzle_idx in range(len(demo_pids)):
                    pid_val = int(demo_pids[puzzle_idx])
                    size = int(demo_pi[puzzle_idx + 1]) - int(demo_pi[puzzle_idx])
                    # Only register puzzles that actually have enough demo pairs
                    # to supply K demos.
                    if size >= self._arm_collator.k_demos:  # type: ignore
                        # First-match-wins on collision (shouldn't happen since
                        # each puzzle has a unique identifier).
                        if pid_val not in self._arm_demo_lookup:
                            self._arm_demo_lookup[pid_val] = (set_name_, puzzle_idx)

    def _iter_arm_test(self):
        """ARM eval: iterate every test-split example, pull K demos from the
        matching train-split puzzle (by `puzzle_identifier`)."""
        assert self._arm_collator is not None
        self._lazy_load_arm_eval_demos()

        k = self._arm_collator.k_demos
        n_total = 0
        n_yielded = 0
        n_no_demos = 0

        for set_name, dataset in self._data.items():  # type: ignore
            pi = dataset["puzzle_indices"]
            pids_arr = dataset["puzzle_identifiers"]
            inp = dataset["inputs"]
            lab = dataset["labels"]
            n_puzzles = len(pids_arr)

            batch_buf: List[dict] = []

            for puzzle_idx in range(n_puzzles):
                # Rank-shard deterministically at the puzzle level.
                if puzzle_idx % self.config.num_replicas != self.config.rank:
                    continue

                pid_val = int(pids_arr[puzzle_idx])
                lookup = self._arm_demo_lookup.get(pid_val)
                if lookup is None:
                    n_no_demos += 1
                    continue

                demo_set_name, demo_puzzle_idx = lookup
                demo_data = self._arm_demo_data[demo_set_name]
                demo_pi = demo_data["puzzle_indices"]
                demo_inp = demo_data["inputs"]
                demo_lab = demo_data["labels"]
                demo_start = int(demo_pi[demo_puzzle_idx])
                demo_rows = np.arange(demo_start, demo_start + k)
                demo_inputs = demo_inp[demo_rows].astype(np.int32)
                demo_outputs = demo_lab[demo_rows].astype(np.int32)

                # Iterate every test example in this test puzzle as a target.
                start = int(pi[puzzle_idx])
                end = int(pi[puzzle_idx + 1])
                for tgt_row in range(start, end):
                    n_total += 1
                    batch_buf.append(
                        dict(
                            demo_inputs=demo_inputs,
                            demo_outputs=demo_outputs,
                            inputs=inp[tgt_row].astype(np.int32),
                            labels=lab[tgt_row].astype(np.int32),
                            puzzle_identifiers=np.int32(pid_val),
                        )
                    )
                    if len(batch_buf) == self.local_batch_size:
                        n_yielded += len(batch_buf)
                        yield set_name, self._stack_arm_eval_batch(batch_buf), self.config.global_batch_size
                        batch_buf = []

            if batch_buf:
                # Pad with repeats so downstream shapes are stable.
                while len(batch_buf) < self.local_batch_size:
                    batch_buf.append(batch_buf[-1])
                n_yielded += len(batch_buf)
                yield set_name, self._stack_arm_eval_batch(batch_buf), self.config.global_batch_size

        if n_total > 0:
            print(
                f"[ArmEval] iterated {n_total} test examples across "
                f"{n_total + n_no_demos} candidates "
                f"(skipped {n_no_demos} with no train-split demos), "
                f"yielded {n_yielded} (incl. padding).",
                flush=True,
            )

    def _stack_arm_eval_batch(self, rows: List[dict]) -> Dict[str, torch.Tensor]:
        out = {
            "demo_inputs": torch.from_numpy(np.stack([r["demo_inputs"] for r in rows])),
            "demo_outputs": torch.from_numpy(np.stack([r["demo_outputs"] for r in rows])),
            "inputs": torch.from_numpy(np.stack([r["inputs"] for r in rows])),
            "labels": torch.from_numpy(np.stack([r["labels"] for r in rows])),
            "puzzle_identifiers": torch.from_numpy(
                np.array([r["puzzle_identifiers"] for r in rows], dtype=np.int32)
            ),
        }
        if self.metadata.ignore_label_id is not None:
            mask = out["labels"] == self.metadata.ignore_label_id
            out["labels"] = torch.where(
                mask, torch.full_like(out["labels"], IGNORE_LABEL_ID), out["labels"]
            )
        out["episode_puzzle_ids"] = out["puzzle_identifiers"].clone()
        return out

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."

        self._lazy_load_dataset()

        # Iterate using specified mode
        if self.config.support_ttt_mode:
            if self.config.test_set_mode:
                yield from self._iter_support_ttt_test()
            else:
                yield from self._iter_support_ttt_train()
        elif self.config.arm_episode_mode:
            if self.config.test_set_mode:
                yield from self._iter_arm_test()
            else:
                yield from self._iter_arm_train()
        elif self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()

