"""Abstract base class for dataset builders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Tuple
import hashlib
import json
import random
import yaml
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Default train/test split ratios (must sum to 1.0)
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_TEST_RATIO = 0.2
SPLIT_NAMES = ("train", "test")


@dataclass
class DatasetMetadata:
    """Metadata stored in meta.yaml for each generated dataset (or split)."""
    difficulty: Optional[str]
    seed: int
    n_tasks: int
    n_augmentations: int
    shard_size: int
    num_shards: int
    generator_config: Dict[str, Any]
    source_data: str
    split: Optional[str] = None  # "train" or "test" when using split layout
    # Soft prompt table sizes (set when write_softprompt_fields was used at build time)
    num_task_identifiers: Optional[int] = None
    num_color_identifiers: Optional[int] = None


class DatasetBuilder(ABC):
    """Abstract base class for building sharded datasets with train/test splits.

    Split is by *original task ID* so all augmentations of the same task land in
    the same split. This guarantees the test set contains only truly unseen tasks.
    """

    def __init__(
        self,
        output_dir: Path,
        raw_data_dir: Path,
        difficulty: Optional[str] = None,
        seed: int = 0,
        generator_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the dataset builder."""
        self.base_output_dir = Path(output_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.difficulty = difficulty
        self.seed = seed
        self.generator_config = generator_config or {}

        self.n_tasks = 0
        self.n_augmentations = 0
        self.shard_size = 0
        self.num_shards = 0

        # output_dir will be set after dataset name is determined; each split has its own subdir
        self.output_dir: Optional[Path] = None

    def build(self) -> DatasetMetadata:
        """Build the complete dataset with train/test splits, sharding and indexing."""
        logger.info(f"Starting dataset build in base directory: {self.base_output_dir}")
        logger.info(f"Difficulty: {self.difficulty}, Seed: {self.seed}")

        raw_data = self._load_raw_data()
        logger.info(f"Loaded raw data: {len(raw_data)} items")

        self.n_tasks = self._get_n_tasks(raw_data)
        self.n_augmentations = self._get_n_augmentations(raw_data)

        dataset_name = self._get_dataset_name()
        self.output_dir = self.base_output_dir / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for split_name in SPLIT_NAMES:
            (self.output_dir / split_name).mkdir(parents=True, exist_ok=True)

        estimated_tasks = self.n_tasks * self.n_augmentations
        self.shard_size = self._compute_shard_size(raw_data, estimated_tasks)

        logger.info(f"Dataset name: {dataset_name}, Output: {self.output_dir}")
        logger.info(f"n_tasks: {self.n_tasks}, n_augmentations: {self.n_augmentations}, "
                    f"Estimated tasks: {estimated_tasks}, Shard size: {self.shard_size}")
        logger.info("Train shards: examples within each shard are shuffled at build time (seed=%s)", self.seed)

        tasks = self._generate_tasks(raw_data)
        split_stats = self._write_shards_and_index_splits(tasks)

        # Log how train/test example counts were concluded
        train_r, test_r = self._get_split_ratios()
        total_written = sum(n for n, _ in split_stats.values())
        logger.info(
            "Split summary: assign each row by hash(seed, original_task_id); "
            f"train_ratio={train_r}, test_ratio={test_r}. "
            "All augmentations of the same task go to the same split (test tasks are unseen for TTT)."
        )
        for split_name in SPLIT_NAMES:
            n_examples, num_shards = split_stats[split_name]
            pct = (100.0 * n_examples / total_written) if total_written else 0
            logger.info(
                f"  {split_name}: {n_examples} examples in {num_shards} shard(s) ({pct:.1f}% of total)"
            )
        logger.info(f"Total rows written: {total_written}")

        for split_name, (n_tasks, num_shards) in split_stats.items():
            metadata = self._create_metadata_for_split(split_name, n_tasks, num_shards)
            self._write_metadata(metadata, split_name)

        # Return metadata for train split as the primary result
        train_n, train_shards = split_stats["train"]
        return self._create_metadata_for_split("train", train_n, train_shards)

    def _compute_shard_size(self, raw_data: List[Any], estimated_tasks: int) -> int:
        """Compute shard size inferred from raw dataset size and augmentation config."""
        if estimated_tasks < 50_000:
            return 1000
        elif estimated_tasks < 500_000:
            return 5000
        else:
            return 10000

    def _get_split_ratios(self) -> Tuple[float, float]:
        """Return (train_ratio, test_ratio) from config; must sum to 1.0."""
        train_r = self.generator_config.get("train_ratio", DEFAULT_TRAIN_RATIO)
        test_r = self.generator_config.get("test_ratio", DEFAULT_TEST_RATIO)
        total = train_r + test_r
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got train={train_r} test={test_r}")
        return train_r, test_r

    def _assign_split(self, task_index: int, original_task_id: Optional[str] = None) -> str:
        """Assign task to train/test split.

        When *original_task_id* is provided the split is determined by hashing
        the ID so that every augmentation of the same underlying task lands in
        the same split.  Otherwise falls back to index-based assignment for
        backward compatibility with older builders.
        """
        train_r, _test_r = self._get_split_ratios()

        if original_task_id is not None:
            h = hashlib.sha256(f"{self.seed}:{original_task_id}".encode()).hexdigest()
            frac = int(h[:8], 16) / 0xFFFFFFFF
        else:
            frac = (self.seed + task_index) % 10000 / 10000.0

        if frac < train_r:
            return "train"
        return "test"

    def _write_shards_and_index_splits(self, tasks: Iterator[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
        """Write tasks into train/test subdirs; each has index.jsonl, meta.yaml, shards."""
        buffers: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLIT_NAMES}
        shard_count: Dict[str, int] = {s: 0 for s in SPLIT_NAMES}
        task_offset: Dict[str, int] = {s: 0 for s in SPLIT_NAMES}
        index_files: Dict[str, Any] = {}

        for split_name in SPLIT_NAMES:
            index_path = self.output_dir / split_name / "index.jsonl"
            index_files[split_name] = open(index_path, "w")

        try:
            for task_index, task in enumerate(tasks):
                # Prefer task-ID-based splitting so all augmentations stay together
                original_task_id = task.get("original_task_id") if isinstance(task, dict) else None
                split_name = self._assign_split(task_index, original_task_id)
                buffers[split_name].append(task)

                while len(buffers[split_name]) >= self.shard_size:
                    chunk = buffers[split_name][: self.shard_size]
                    if split_name == "train":
                        rng = random.Random(self.seed + shard_count[split_name])
                        rng.shuffle(chunk)
                    self._flush_split_shard(
                        split_name,
                        chunk,
                        shard_count,
                        task_offset,
                        index_files,
                    )
                    buffers[split_name] = buffers[split_name][self.shard_size :]
                    shard_count[split_name] += 1

            for split_name in SPLIT_NAMES:
                while buffers[split_name]:
                    chunk = buffers[split_name][: self.shard_size]
                    if split_name == "train":
                        rng = random.Random(self.seed + shard_count[split_name])
                        rng.shuffle(chunk)
                    self._flush_split_shard(
                        split_name,
                        chunk,
                        shard_count,
                        task_offset,
                        index_files,
                    )
                    buffers[split_name] = buffers[split_name][len(chunk) :]
                    shard_count[split_name] += 1
        finally:
            for f in index_files.values():
                f.close()

        return {
            split_name: (task_offset[split_name], shard_count[split_name])
            for split_name in SPLIT_NAMES
        }

    def _flush_split_shard(
        self,
        split_name: str,
        tasks_chunk: List[Dict[str, Any]],
        shard_count: Dict[str, int],
        task_offset: Dict[str, int],
        index_files: Dict[str, Any],
    ):
        """Write one shard for a split and append to its index."""
        c = shard_count[split_name]
        shard_filename = f"shard_{c:05d}.jsonl"
        split_dir = self.output_dir / split_name
        shard_path = split_dir / shard_filename
        self._write_shard(shard_path, tasks_chunk)

        index_entry = {
            "shard": shard_filename,
            "offset": task_offset[split_name],
            "num_tasks": len(tasks_chunk),
            "shard_path": shard_filename,
        }
        index_files[split_name].write(json.dumps(index_entry) + "\n")
        task_offset[split_name] += len(tasks_chunk)

    def _write_shard(self, shard_path: Path, tasks: List[Dict[str, Any]]):
        """Write tasks to a shard file in JSONL format."""
        with open(shard_path, "w") as f:
            for task in tasks:
                jsonl_line = self._task_to_jsonl(task)
                f.write(jsonl_line + "\n")
        logger.debug(f"Wrote shard: {shard_path} ({len(tasks)} tasks)")

    def _create_metadata(self) -> DatasetMetadata:
        """Create metadata for backward compatibility; prefer _create_metadata_for_split."""
        return self._create_metadata_for_split("train", self.n_tasks * self.n_augmentations, self.num_shards)

    def _create_metadata_for_split(self, split: str, n_tasks_in_split: int, num_shards: int) -> DatasetMetadata:
        """Create metadata object for a single split (n_tasks_in_split = task count in this split)."""
        return DatasetMetadata(
            difficulty=self.difficulty,
            seed=self.seed,
            n_tasks=n_tasks_in_split,
            n_augmentations=self.n_augmentations,
            shard_size=self.shard_size,
            num_shards=num_shards,
            generator_config=self.generator_config,
            source_data=self._get_source_data_name(),
            split=split,
            num_task_identifiers=None,
            num_color_identifiers=None,
        )

    def _write_metadata(self, metadata: DatasetMetadata, split_name: str):
        """Write metadata to meta.yaml in the split subdir."""
        split_dir = self.output_dir / split_name
        meta_path = split_dir / "meta.yaml"
        with open(meta_path, "w") as f:
            yaml.dump(asdict(metadata), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Wrote metadata: {meta_path}")

    def _get_dataset_name(self) -> str:
        """
        Generate dataset name from components.

        Default format: {source_data}-{n_tasks}-{n_augmentations}-{difficulty}
        Subclasses can override for custom naming schemes.
        """
        source_name = self._get_source_data_name()
        parts = [source_name, str(self.n_tasks), str(self.n_augmentations)]
        if self.difficulty:
            parts.append(self.difficulty)
        return "-".join(parts)

    @abstractmethod
    def _load_raw_data(self) -> List[Any]:
        """Load raw source data from raw_data_dir."""
        pass

    @abstractmethod
    def _get_n_tasks(self, raw_data: List[Any]) -> int:
        """Get number of raw tasks from the dataset."""
        pass

    @abstractmethod
    def _get_n_augmentations(self, raw_data: List[Any]) -> int:
        """Get number of augmentations per task from the dataset."""
        pass

    @abstractmethod
    def _generate_tasks(self, raw_data: List[Any]) -> Iterator[Dict[str, Any]]:
        """Generate tasks from raw data with optional augmentations."""
        pass

    @abstractmethod
    def _task_to_jsonl(self, task: Dict[str, Any]) -> str:
        """Convert a task dictionary to a JSONL string."""
        pass

    @abstractmethod
    def _get_source_data_name(self) -> str:
        """Return the name of the source dataset."""
        pass
