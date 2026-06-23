"""ARC-AGI-2 dataset builder implementation.

Outputs per-example rows compatible with PuzzleDataset. Each row contains a
single test example together with the demonstration grids, augmentation
metadata (transform_id, color_map), and identifiers needed for the
softprompt pipeline.
"""

import json
import sys
import hashlib
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import logging
import numpy as np

# Handle both relative and absolute imports
try:
    from .build_dataset import DatasetBuilder, DatasetMetadata
    from .common import (
        augment_example, example_hash,
        dihedral_transform, apply_color_mapping, color_permutation,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.dataset.build_dataset import DatasetBuilder, DatasetMetadata
    from utils.dataset.common import (
        augment_example, example_hash,
        dihedral_transform, apply_color_mapping, color_permutation,
    )

logger = logging.getLogger(__name__)


class ARCAGI2DatasetBuilder(DatasetBuilder):
    """Builder for ARC-AGI-2 datasets.

    Produces per-example rows with demonstration grids, augmentation
    metadata, and identifiers for the softprompt/grid-encoder pipeline.
    """

    def __init__(
        self,
        output_dir: Path,
        raw_data_dir: Path,
        difficulty: Optional[str] = None,
        seed: int = 0,
        generator_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ARC-AGI-2 dataset builder.

        Args:
            output_dir: Base output directory
            raw_data_dir: Directory containing ARC-AGI-2 repository
            difficulty: Optional difficulty filter (e.g., "10x10", "30x30")
            seed: Random seed for reproducibility
            generator_config: Should contain:
                - n_augmentations: int (default 8) - augmented versions per task
                - include_evaluation: bool (default False) - include eval set
                - include_test: bool (default True) - include test examples
        """
        super().__init__(output_dir, raw_data_dir, difficulty, seed, generator_config)

        self.n_augmentations_config = self.generator_config.get("n_augmentations", 8)
        self.include_evaluation = self.generator_config.get("include_evaluation", False)
        self.include_test = self.generator_config.get("include_test", True)

        # ARC-AGI-2 specific paths
        self.arc_agi_dir = self.raw_data_dir / "ARC-AGI-2"
        self.training_dir = self.arc_agi_dir / "data" / "training"
        self.evaluation_dir = self.arc_agi_dir / "data" / "evaluation"

        # Task identifier mapping (original_task_id -> int)
        self._task_id_to_int: Dict[str, int] = {}
        self._next_task_int: int = 0

    # ------------------------------------------------------------------
    # Identifier helpers
    # ------------------------------------------------------------------

    def _get_task_identifier(self, original_task_id: str) -> int:
        """Return a stable integer identifier for *original_task_id*."""
        if original_task_id not in self._task_id_to_int:
            self._task_id_to_int[original_task_id] = self._next_task_int
            self._next_task_int += 1
        return self._task_id_to_int[original_task_id]

    # ------------------------------------------------------------------
    # DatasetBuilder interface
    # ------------------------------------------------------------------

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw ARC-AGI-2 task data."""
        if not self.arc_agi_dir.exists():
            raise FileNotFoundError(
                f"ARC-AGI-2 repository not found at {self.arc_agi_dir}. "
                "Please ensure the ARC-AGI-2 submodule is initialized."
            )

        raw_tasks = []

        # Load training tasks
        if self.training_dir.exists():
            for task_file in sorted(self.training_dir.glob("*.json")):
                task_id = task_file.stem
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    raw_tasks.append({
                        "task_id": task_id,
                        "source": "training",
                        "data": task_data,
                    })

        # Optionally load evaluation tasks
        if self.include_evaluation and self.evaluation_dir.exists():
            for task_file in sorted(self.evaluation_dir.glob("*.json")):
                task_id = task_file.stem
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    raw_tasks.append({
                        "task_id": task_id,
                        "source": "evaluation",
                        "data": task_data,
                    })

        logger.info(f"Loaded {len(raw_tasks)} ARC-AGI-2 tasks")
        return raw_tasks

    def _get_n_tasks(self, raw_data: List[Any]) -> int:
        return len(raw_data)

    def _get_n_augmentations(self, raw_data: List[Any]) -> int:
        return self.n_augmentations_config

    def _generate_tasks(self, raw_data: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Generate per-example rows from raw ARC-AGI-2 data with augmentations.

        For each augmentation of each task, every *test* example becomes its
        own row together with the (identically augmented) demonstration pairs.
        """
        rng = np.random.default_rng(self.seed)
        puzzle_counter = 0  # global unique counter across all rows

        for raw_task in raw_data:
            task_id = raw_task["task_id"]
            task_data = raw_task["data"]

            demo_examples = task_data.get("train", [])
            test_examples = task_data.get("test", []) if self.include_test else []

            if not demo_examples:
                logger.warning(f"Task {task_id} has no train examples, skipping")
                continue

            if self.difficulty and not self._matches_difficulty(demo_examples, self.difficulty):
                logger.debug(f"Task {task_id} filtered out by difficulty {self.difficulty}")
                continue

            task_identifier = self._get_task_identifier(task_id)

            # Original (identity augmentation) ---------------------------------
            identity_color_map = list(range(10))
            for test_ex in test_examples:
                puzzle_counter += 1
                yield self._make_row(
                    task_id=task_id,
                    task_identifier=task_identifier,
                    puzzle_identifier=puzzle_counter,
                    demo_examples=demo_examples,
                    test_input=test_ex["input"],
                    test_output=test_ex["output"],
                    transform_id=0,
                    color_map=identity_color_map,
                )

            # Augmented versions -----------------------------------------------
            seen_hashes: set = set()
            aug_count = 1
            max_attempts = self.n_augmentations_config * 5
            attempts = 0

            while aug_count < self.n_augmentations_config and attempts < max_attempts:
                attempts += 1

                # Pick a single augmentation for the whole task
                transform_id = int(rng.integers(0, 8))
                color_map_arr = color_permutation(rng)
                color_map = color_map_arr.tolist()

                # Augment demos
                aug_demos = []
                for ex in demo_examples:
                    inp = np.array(ex["input"], dtype=np.uint8)
                    out = np.array(ex["output"], dtype=np.uint8)
                    a_in = apply_color_mapping(dihedral_transform(inp, transform_id), color_map_arr)
                    a_out = apply_color_mapping(dihedral_transform(out, transform_id), color_map_arr)
                    aug_demos.append({"input": a_in.tolist(), "output": a_out.tolist()})

                # Augment test examples
                aug_tests = []
                for ex in test_examples:
                    inp = np.array(ex["input"], dtype=np.uint8)
                    out = np.array(ex["output"], dtype=np.uint8)
                    a_in = apply_color_mapping(dihedral_transform(inp, transform_id), color_map_arr)
                    a_out = apply_color_mapping(dihedral_transform(out, transform_id), color_map_arr)
                    aug_tests.append({"input": a_in.tolist(), "output": a_out.tolist()})

                # Deduplicate
                task_hash = self._compute_task_hash(aug_demos, aug_tests)
                if task_hash in seen_hashes:
                    continue
                seen_hashes.add(task_hash)

                for test_ex in aug_tests:
                    puzzle_counter += 1
                    yield self._make_row(
                        task_id=task_id,
                        task_identifier=task_identifier,
                        puzzle_identifier=puzzle_counter,
                        demo_examples=aug_demos,
                        test_input=test_ex["input"],
                        test_output=test_ex["output"],
                        transform_id=transform_id,
                        color_map=color_map,
                    )
                aug_count += 1

    def _make_row(
        self,
        task_id: str,
        task_identifier: int,
        puzzle_identifier: int,
        demo_examples: List[Dict[str, Any]],
        test_input: List[List[int]],
        test_output: List[List[int]],
        transform_id: int,
        color_map: List[int],
    ) -> Dict[str, Any]:
        """Build one per-example JSONL row."""
        return {
            "input": test_input,
            "output": test_output,
            "demo_inputs": [ex["input"] for ex in demo_examples],
            "demo_outputs": [ex["output"] for ex in demo_examples],
            "transform_id": transform_id,
            "color_map": color_map,
            "original_task_id": task_id,
            "puzzle_identifier": puzzle_identifier,
            "task_identifier": task_identifier,
        }

    def _compute_task_hash(self, demo_examples: List[Dict], test_examples: List[Dict]) -> str:
        """Compute hash of a task for deduplication."""
        hashes = []
        for ex in demo_examples:
            hashes.append(example_hash(ex["input"], ex["output"]))
        for ex in test_examples:
            hashes.append(example_hash(ex["input"], ex["output"]))
        hashes.sort()
        return "|".join(hashes)

    def _matches_difficulty(self, examples: List[Dict[str, Any]], difficulty: str) -> bool:
        """Check if task examples match difficulty filter (NxM max dimensions)."""
        if not examples:
            return False

        for ex in examples:
            input_grid = ex.get("input", [])
            output_grid = ex.get("output", [])

            for grid in [input_grid, output_grid]:
                if not grid:
                    continue
                height = len(grid)
                width = len(grid[0]) if grid else 0

                if "x" in difficulty:
                    try:
                        max_dim = int(difficulty.split("x")[0])
                        if height > max_dim or width > max_dim:
                            return False
                    except ValueError:
                        logger.warning(f"Invalid difficulty format: {difficulty}")
                        return True

        return True

    def _task_to_jsonl(self, task: Dict[str, Any]) -> str:
        return json.dumps(task, ensure_ascii=False)

    def _get_source_data_name(self) -> str:
        return "arc-agi-2"

    def _create_metadata_for_split(self, split: str, n_tasks_in_split: int, num_shards: int) -> DatasetMetadata:
        """Override to include num_task_identifiers."""
        base = super()._create_metadata_for_split(split, n_tasks_in_split, num_shards)
        base.num_task_identifiers = self._next_task_int
        return base


def main():
    """Main entry point for ARC-AGI-2 dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Build ARC-AGI-2 dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--raw-data-dir", type=str, required=True, help="Raw data directory containing ARC-AGI-2 repository")
    parser.add_argument("--difficulty", type=str, default=None, help="Difficulty filter (e.g., '10x10', '30x30')")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n-augmentations", type=int, default=8, help="Number of augmentations per task")
    parser.add_argument("--include-evaluation", action="store_true", help="Include evaluation set")
    parser.add_argument("--include-test", action="store_true", default=True, help="Include test examples")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default 0.8)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio (default 0.2)")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override dataset name")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks (for debugging)")

    args = parser.parse_args()

    # Configure logging
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    logger.info("Starting ARC-AGI-2 dataset generation")
    logger.info(
        f"Configuration: n_augmentations={args.n_augmentations}, difficulty={args.difficulty}, seed={args.seed}, "
        f"train_ratio={args.train_ratio}, test_ratio={args.test_ratio}"
    )

    generator_config = {
        "n_augmentations": args.n_augmentations,
        "include_evaluation": args.include_evaluation,
        "include_test": args.include_test,
        "train_ratio": args.train_ratio,
        "test_ratio": args.test_ratio,
    }

    builder = ARCAGI2DatasetBuilder(
        output_dir=Path(args.output_dir),
        raw_data_dir=Path(args.raw_data_dir),
        difficulty=args.difficulty,
        seed=args.seed,
        generator_config=generator_config,
    )

    metadata = builder.build()
    logger.info(f"Dataset built successfully: {builder.output_dir}")
    logger.info(
        f"Train split: {metadata.n_tasks} examples ({metadata.num_shards} shards). "
        f"Unique original_task_ids in dataset: {builder._next_task_int}. "
        "See split summary above for test count and how train/test were assigned."
    )


if __name__ == "__main__":
    main()
