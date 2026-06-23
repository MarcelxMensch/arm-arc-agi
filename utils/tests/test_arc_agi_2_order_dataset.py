"""Tests for ARC-AGI-2 ordered dataset builder and retention testing.

The arc-agi-2-order dataset is designed for retention testing:
- Train split: tasks from ARC-AGI-2 training/ (model sees these first).
- Test split: tasks from ARC-AGI-2 evaluation/ (held-out tasks).

Key properties for retention testing:
1. Each row is one example (input/output pair), not a task with examples array.
2. All examples from the same original_task_id share the same puzzle_identifier.
3. Examples are ordered: all examples from task A aug 0, then A aug 1, then task B, etc.
4. When loaded in order (test_set_mode=True), we see:
   - First: multiple examples with the same puzzle_identifier (problem group A)
   - Then: examples with different puzzle_identifiers (other problems)
   - Later: the same puzzle_identifier from earlier appears again (retention test)
"""

import json
import tempfile
import unittest
import yaml
from pathlib import Path
from collections import defaultdict

# Add project root for imports when running tests from repo root or utils/tests
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.dataset.build_arc_agi_2_order import ARCAGI2OrderDatasetBuilder
from utils.dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def _raw_data_dir() -> Path:
    """Path to ARC-AGI-2 raw data (training/ and evaluation/ under data/)."""
    return _PROJECT_ROOT / "utils" / "dataset" / "raw-data"


def _has_arc_agi_2_raw_data() -> bool:
    """Return True if ARC-AGI-2 raw data exists (required to build dataset)."""
    base = _raw_data_dir() / "ARC-AGI-2"
    return (base / "data" / "training").is_dir() and (base / "data" / "evaluation").is_dir()


def _build_minimal_dataset(tmp_path: Path, n_augmentations: int = 3) -> Path:
    """Build arc-agi-2-order dataset with minimal augmentations; return dataset root."""
    out = tmp_path / "arc-agi-2-order-test"
    out.mkdir(parents=True, exist_ok=True)
    raw = _raw_data_dir()
    builder = ARCAGI2OrderDatasetBuilder(
        output_dir=tmp_path,
        raw_data_dir=raw,
        difficulty=None,
        seed=42,
        generator_config={"n_augmentations": n_augmentations},
    )
    builder.build()
    # Dataset name is arc-agi-2-order-{n_tasks}-{n_augmentations}
    name = f"arc-agi-2-order-{builder.n_tasks}-{n_augmentations}"
    return tmp_path / name


def _load_meta(dataset_root: Path, split: str) -> dict:
    """Load meta.yaml for a split."""
    meta_path = dataset_root / split / "meta.yaml"
    assert meta_path.exists(), f"Missing {meta_path}"
    with open(meta_path) as f:
        return yaml.safe_load(f)


def _load_index(dataset_root: Path, split: str) -> list:
    """Load index.jsonl for a split (list of shard entries)."""
    index_path = dataset_root / split / "index.jsonl"
    assert index_path.exists(), f"Missing {index_path}"
    entries = []
    with open(index_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def _load_examples_from_shard(dataset_root: Path, split: str, shard_path: str) -> list:
    """Load all examples from one shard file."""
    path = dataset_root / split / shard_path
    assert path.exists(), f"Missing {path}"
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def _iter_all_examples(dataset_root: Path, split: str):
    """Yield every example dict from a split (all shards, in order)."""
    index = _load_index(dataset_root, split)
    for entry in index:
        for example in _load_examples_from_shard(dataset_root, split, entry["shard_path"]):
            yield example


class TestARCAGI2OrderDatasetStructure(unittest.TestCase):
    """Test the structure of the generated dataset."""

    @unittest.skipUnless(_has_arc_agi_2_raw_data(), "ARC-AGI-2 raw data not found")
    def test_build_produces_train_and_test_splits(self):
        """Building the dataset produces train/ and test/ with meta, index, and shards."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = _build_minimal_dataset(Path(tmp), n_augmentations=2)
            self.assertTrue(dataset_root.is_dir(), f"Dataset root should exist: {dataset_root}")

            for split in ("train", "test"):
                split_dir = dataset_root / split
                self.assertTrue(split_dir.is_dir(), f"Split dir should exist: {split_dir}")
                self.assertTrue((split_dir / "meta.yaml").exists(), f"{split}/meta.yaml")
                self.assertTrue((split_dir / "index.jsonl").exists(), f"{split}/index.jsonl")

    @unittest.skipUnless(_has_arc_agi_2_raw_data(), "ARC-AGI-2 raw data not found")
    def test_each_example_has_input_output_puzzle_identifier(self):
        """Each row has input, output, and puzzle_identifier fields."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = _build_minimal_dataset(Path(tmp), n_augmentations=2)
            for split in ("train", "test"):
                count = 0
                for example in _iter_all_examples(dataset_root, split):
                    self.assertIn("input", example, "Example should have 'input'")
                    self.assertIn("output", example, "Example should have 'output'")
                    self.assertIn("puzzle_identifier", example, "Example should have 'puzzle_identifier'")
                    self.assertIn("original_task_id", example, "Example should have 'original_task_id'")
                    self.assertIsInstance(example["puzzle_identifier"], int)
                    count += 1
                    if count >= 100:  # Check first 100
                        break

    @unittest.skipUnless(_has_arc_agi_2_raw_data(), "ARC-AGI-2 raw data not found")
    def test_same_original_task_id_has_same_puzzle_identifier(self):
        """All examples with the same original_task_id have the same puzzle_identifier."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = _build_minimal_dataset(Path(tmp), n_augmentations=2)
            for split in ("train", "test"):
                task_to_puzzle_id = {}
                for example in _iter_all_examples(dataset_root, split):
                    original_task_id = example["original_task_id"]
                    puzzle_id = example["puzzle_identifier"]
                    if original_task_id in task_to_puzzle_id:
                        self.assertEqual(
                            task_to_puzzle_id[original_task_id], puzzle_id,
                            f"All examples from {original_task_id} should have same puzzle_identifier"
                        )
                    else:
                        task_to_puzzle_id[original_task_id] = puzzle_id

    @unittest.skipUnless(_has_arc_agi_2_raw_data(), "ARC-AGI-2 raw data not found")
    def test_train_has_only_training_source(self):
        """Every example in train split has source=training."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = _build_minimal_dataset(Path(tmp), n_augmentations=2)
            count = 0
            for example in _iter_all_examples(dataset_root, "train"):
                self.assertEqual(example.get("source"), "training")
                count += 1
                if count >= 100:
                    break

    @unittest.skipUnless(_has_arc_agi_2_raw_data(), "ARC-AGI-2 raw data not found")
    def test_test_has_only_evaluation_source(self):
        """Every example in test split has source=evaluation."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = _build_minimal_dataset(Path(tmp), n_augmentations=2)
            count = 0
            for example in _iter_all_examples(dataset_root, "test"):
                self.assertEqual(example.get("source"), "evaluation")
                count += 1
                if count >= 100:
                    break


class TestARCAGI2OrderRetentionOrdering(unittest.TestCase):
    """Test the ordering properties needed for retention testing."""

    @unittest.skipUnless(_has_arc_agi_2_raw_data(), "ARC-AGI-2 raw data not found")
    def test_examples_grouped_by_task_and_augmentation(self):
        """
        Examples are ordered: all examples from task A aug 0, then A aug 1, then task B, etc.
        This means consecutive examples often have the same (original_task_id, augmentation_index).
        """
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = _build_minimal_dataset(Path(tmp), n_augmentations=3)
            
            for split in ("train",):  # Check train split
                prev_task_aug = None
                same_group_count = 0
                transitions = 0
                
                for example in _iter_all_examples(dataset_root, split):
                    current_task_aug = (example["original_task_id"], example["augmentation_index"])
                    if prev_task_aug is not None:
                        if current_task_aug == prev_task_aug:
                            same_group_count += 1
                        else:
                            transitions += 1
                    prev_task_aug = current_task_aug
                
                # With multiple examples per task, we should see many consecutive same-group examples
                self.assertGreater(same_group_count, 0, "Should have consecutive examples from same task+aug")

    @unittest.skipUnless(_has_arc_agi_2_raw_data(), "ARC-AGI-2 raw data not found")
    def test_puzzle_identifier_appears_multiple_times_in_dataset(self):
        """
        The same puzzle_identifier (original_task_id) appears multiple times because
        we have n_augmentations versions of each task. This is needed for retention testing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = _build_minimal_dataset(Path(tmp), n_augmentations=3)
            
            for split in ("train",):
                puzzle_id_counts = defaultdict(int)
                for example in _iter_all_examples(dataset_root, split):
                    puzzle_id_counts[example["puzzle_identifier"]] += 1
                
                # Each puzzle_id should appear at least n_augmentations * n_examples times
                # (where n_examples is the number of examples per task, typically 3-5)
                multi_appearance = sum(1 for count in puzzle_id_counts.values() if count > 1)
                self.assertGreater(multi_appearance, 0, "Should have puzzle_ids appearing multiple times")

    @unittest.skipUnless(_has_arc_agi_2_raw_data(), "ARC-AGI-2 raw data not found")
    def test_puzzle_identifier_reappears_after_other_tasks(self):
        """
        When iterating in order, we should see:
        1. puzzle_id X (from task A aug 0)
        2. other puzzle_ids (from other tasks or augmentations)
        3. puzzle_id X again (from task A aug 1, or later)
        
        This is the key property for retention testing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = _build_minimal_dataset(Path(tmp), n_augmentations=3)
            
            # Track the order of puzzle_ids
            puzzle_id_order = []
            for example in _iter_all_examples(dataset_root, "train"):
                puzzle_id_order.append(example["puzzle_identifier"])
            
            # Find a puzzle_id that appears, then doesn't appear, then appears again
            first_seen = {}  # puzzle_id -> index of first appearance
            last_seen = {}   # puzzle_id -> index of last appearance
            
            for i, pid in enumerate(puzzle_id_order):
                if pid not in first_seen:
                    first_seen[pid] = i
                last_seen[pid] = i
            
            # Check that at least some puzzle_ids have other puzzle_ids between their appearances
            reappearance_found = False
            for pid, first_idx in first_seen.items():
                last_idx = last_seen[pid]
                if last_idx > first_idx + 1:
                    # Check if there are other puzzle_ids between first and last
                    between = puzzle_id_order[first_idx+1:last_idx]
                    other_pids = set(between) - {pid}
                    if other_pids:
                        reappearance_found = True
                        break
            
            self.assertTrue(reappearance_found, 
                "Should find puzzle_ids that reappear after other puzzle_ids (for retention testing)")


class TestARCAGI2OrderPuzzleDatasetLoading(unittest.TestCase):
    """Test loading via PuzzleDataset (like train.py does)."""

    @unittest.skipUnless(_has_arc_agi_2_raw_data(), "ARC-AGI-2 raw data not found")
    def test_load_dataset_with_puzzle_dataset(self):
        """Load the dataset via PuzzleDataset and verify puzzle_identifiers are from data."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = _build_minimal_dataset(Path(tmp), n_augmentations=2)
            
            config = PuzzleDatasetConfig(
                dataset_paths=[str(dataset_root)],
                seed=42,
                rank=0,
                num_replicas=1,
                epochs_per_iter=1,
                global_batch_size=8,
                test_set_mode=True,  # No shuffle - preserve order
            )
            dataset = PuzzleDataset(config, split="train")
            
            # Collect puzzle_identifiers from batches
            puzzle_ids_from_loader = []
            batch_count = 0
            for split, batch, effective_batch_size in dataset:
                puzzle_ids_from_loader.extend(batch["puzzle_identifiers"].tolist())
                batch_count += 1
                if batch_count >= 10:  # Check first 10 batches
                    break
            
            self.assertGreater(len(puzzle_ids_from_loader), 0, "Should have loaded some examples")
            
            # Verify puzzle_ids are consistent with the data
            # (they should be the same puzzle_identifier values from the JSONL)
            expected_puzzle_ids = []
            count = 0
            for example in _iter_all_examples(dataset_root, "train"):
                expected_puzzle_ids.append(example["puzzle_identifier"])
                count += 1
                if count >= len(puzzle_ids_from_loader):
                    break
            
            self.assertEqual(
                puzzle_ids_from_loader[:len(expected_puzzle_ids)],
                expected_puzzle_ids,
                "Puzzle identifiers from loader should match data"
            )

    @unittest.skipUnless(_has_arc_agi_2_raw_data(), "ARC-AGI-2 raw data not found")
    def test_consecutive_batches_same_puzzle_id(self):
        """
        In test_set_mode (no shuffle), first batches should have examples from
        the same task (same puzzle_identifier) because examples are grouped.
        """
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = _build_minimal_dataset(Path(tmp), n_augmentations=2)
            
            config = PuzzleDatasetConfig(
                dataset_paths=[str(dataset_root)],
                seed=42,
                rank=0,
                num_replicas=1,
                epochs_per_iter=1,
                global_batch_size=2,  # Small batch to see grouping
                test_set_mode=True,
            )
            dataset = PuzzleDataset(config, split="train")
            
            puzzle_ids = []
            batch_count = 0
            for split, batch, effective_batch_size in dataset:
                puzzle_ids.extend(batch["puzzle_identifiers"].tolist())
                batch_count += 1
                if batch_count >= 5:
                    break
            
            # Check that we see some consecutive same puzzle_ids
            same_consecutive = sum(1 for i in range(1, len(puzzle_ids)) if puzzle_ids[i] == puzzle_ids[i-1])
            self.assertGreater(same_consecutive, 0, 
                "Should have consecutive examples with same puzzle_identifier (same task group)")


def run_tests():
    """Run tests (e.g. from __main__ or pytest)."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestARCAGI2OrderDatasetStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestARCAGI2OrderRetentionOrdering))
    suite.addTests(loader.loadTestsFromTestCase(TestARCAGI2OrderPuzzleDatasetLoading))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    run_tests()
