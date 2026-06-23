"""ARC-AGI-1 + ConceptARC dataset builder implementation."""

import json
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import logging
import numpy as np

# Handle both relative and absolute imports
try:
    from .build_dataset import DatasetBuilder, DatasetMetadata
    from .common import augment_example, example_hash
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.dataset.build_dataset import DatasetBuilder, DatasetMetadata
    from utils.dataset.common import augment_example, example_hash

logger = logging.getLogger(__name__)


class ARCAGI1ConceptARCDatasetBuilder(DatasetBuilder):
    """Builder for ARC-AGI-1 + ConceptARC combined datasets."""
    
    def __init__(
        self,
        output_dir: Path,
        raw_data_dir: Path,
        difficulty: Optional[str] = None,
        seed: int = 0,
        generator_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ARC-AGI-1 + ConceptARC dataset builder.
        
        Args:
            output_dir: Base output directory
            raw_data_dir: Directory containing ARC-AGI and ConceptARC repositories
            difficulty: Optional difficulty filter (e.g., "10x10", "30x30")
            seed: Random seed for reproducibility
            generator_config: Should contain:
                - n_augmentations: int (default 8) - number of augmented versions per task
                - include_evaluation: bool (default False) - include ARC-AGI evaluation set
                - include_test: bool (default True) - include test examples in output
                - include_minimal: bool (default True) - include ConceptARC MinimalTasks
        """
        super().__init__(output_dir, raw_data_dir, difficulty, seed, generator_config)
        
        self.n_augmentations_config = self.generator_config.get("n_augmentations", 8)
        self.include_evaluation = self.generator_config.get("include_evaluation", False)
        self.include_test = self.generator_config.get("include_test", True)
        self.include_minimal = self.generator_config.get("include_minimal", True)
        
        # Dataset paths
        self.arc_agi_dir = self.raw_data_dir / "ARC-AGI"
        self.concept_arc_dir = self.raw_data_dir / "ConceptARC"
        self.arc_training_dir = self.arc_agi_dir / "data" / "training"
        self.arc_evaluation_dir = self.arc_agi_dir / "data" / "evaluation"
        self.concept_corpus_dir = self.concept_arc_dir / "corpus"
        self.concept_minimal_dir = self.concept_arc_dir / "MinimalTasks"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw ARC-AGI and ConceptARC task data."""
        raw_tasks = []
        
        # Load ARC-AGI training tasks
        if self.arc_agi_dir.exists() and self.arc_training_dir.exists():
            for task_file in sorted(self.arc_training_dir.glob("*.json")):
                task_id = task_file.stem
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    raw_tasks.append({
                        "task_id": f"arc-agi-1_{task_id}",
                        "source": "arc-agi-1-training",
                        "data": task_data
                    })
        else:
            logger.warning(f"ARC-AGI repository not found at {self.arc_agi_dir}")
        
        # Load ARC-AGI evaluation tasks
        if self.include_evaluation and self.arc_evaluation_dir.exists():
            for task_file in sorted(self.arc_evaluation_dir.glob("*.json")):
                task_id = task_file.stem
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    raw_tasks.append({
                        "task_id": f"arc-agi-1_{task_id}",
                        "source": "arc-agi-1-evaluation",
                        "data": task_data
                    })
        
        # Load ConceptARC corpus tasks
        if self.concept_arc_dir.exists() and self.concept_corpus_dir.exists():
            for concept_dir in sorted(self.concept_corpus_dir.iterdir()):
                if concept_dir.is_dir():
                    concept_name = concept_dir.name
                    for task_file in sorted(concept_dir.glob("*.json")):
                        task_id = task_file.stem
                        with open(task_file, "r") as f:
                            task_data = json.load(f)
                            raw_tasks.append({
                                "task_id": f"concept_{concept_name}_{task_id}",
                                "source": f"concept-arc-{concept_name}",
                                "data": task_data
                            })
        else:
            logger.warning(f"ConceptARC repository not found at {self.concept_arc_dir}")
        
        # Load ConceptARC minimal tasks
        if self.include_minimal and self.concept_minimal_dir.exists():
            for task_file in sorted(self.concept_minimal_dir.glob("*.json")):
                task_id = task_file.stem
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    raw_tasks.append({
                        "task_id": f"concept_minimal_{task_id}",
                        "source": "concept-arc-minimal",
                        "data": task_data
                    })
        
        logger.info(f"Loaded {len(raw_tasks)} tasks (ARC-AGI-1 + ConceptARC)")
        return raw_tasks
    
    def _get_n_tasks(self, raw_data: List[Any]) -> int:
        """Get number of raw tasks from the dataset."""
        return len(raw_data)
    
    def _get_n_augmentations(self, raw_data: List[Any]) -> int:
        """Get number of augmentations per task from the dataset."""
        return self.n_augmentations_config
    
    def _generate_tasks(self, raw_data: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """
        Generate tasks from raw data with augmentations.
        
        Each task is augmented n_augmentations times using dihedral transforms
        and color permutations.
        """
        rng = np.random.default_rng(self.seed)
        
        for raw_task in raw_data:
            task_id = raw_task["task_id"]
            task_data = raw_task["data"]
            source = raw_task["source"]
            
            # Get train and test examples
            train_examples = task_data.get("train", [])
            test_examples = task_data.get("test", []) if self.include_test else []
            
            if not train_examples:
                logger.warning(f"Task {task_id} has no train examples, skipping")
                continue
            
            # Apply difficulty filtering if specified
            if self.difficulty and not self._matches_difficulty(train_examples, self.difficulty):
                logger.debug(f"Task {task_id} filtered out by difficulty {self.difficulty}")
                continue
            
            # Generate original (unaugmented) version
            yield self._create_task_entry(task_id, source, train_examples, test_examples, 0, "original")
            
            # Generate augmented versions
            seen_hashes = set()
            aug_count = 1
            max_attempts = self.n_augmentations_config * 5
            attempts = 0
            
            while aug_count < self.n_augmentations_config and attempts < max_attempts:
                attempts += 1
                
                # Augment all examples in the task
                aug_train = []
                aug_test = []
                aug_repr = None
                
                for ex in train_examples:
                    aug_in, aug_out, _tid, _cmap, repr_str = augment_example(
                        ex["input"], ex["output"], rng
                    )
                    aug_train.append({"input": aug_in, "output": aug_out})
                    if aug_repr is None:
                        aug_repr = repr_str
                
                for ex in test_examples:
                    aug_in, aug_out, _tid, _cmap, _ = augment_example(
                        ex["input"], ex["output"], rng
                    )
                    aug_test.append({"input": aug_in, "output": aug_out})
                
                # Check for duplicates
                task_hash = self._compute_task_hash(aug_train, aug_test)
                if task_hash not in seen_hashes:
                    seen_hashes.add(task_hash)
                    yield self._create_task_entry(task_id, source, aug_train, aug_test, aug_count, aug_repr)
                    aug_count += 1
    
    def _create_task_entry(
        self,
        task_id: str,
        source: str,
        train_examples: List[Dict],
        test_examples: List[Dict],
        aug_index: int,
        aug_repr: str
    ) -> Dict[str, Any]:
        """Create a task entry for the dataset."""
        return {
            "task_id": f"{task_id}_{aug_index}",
            "original_task_id": task_id,
            "source": source,
            "augmentation_index": aug_index,
            "augmentation_type": aug_repr,
            "train": train_examples,
            "test": test_examples,
        }
    
    def _compute_task_hash(self, train_examples: List[Dict], test_examples: List[Dict]) -> str:
        """Compute hash of a task for deduplication."""
        hashes = []
        for ex in train_examples:
            hashes.append(example_hash(ex["input"], ex["output"]))
        for ex in test_examples:
            hashes.append(example_hash(ex["input"], ex["output"]))
        hashes.sort()
        return "|".join(hashes)
    
    def _matches_difficulty(self, examples: List[Dict[str, Any]], difficulty: str) -> bool:
        """Check if task examples match difficulty filter."""
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
        """Convert a task dictionary to a JSONL string."""
        return json.dumps(task, ensure_ascii=False)
    
    def _get_source_data_name(self) -> str:
        """Return the name of the source dataset."""
        return "arc-agi-1-concept"


def main():
    """Main entry point for ARC-AGI-1 + ConceptARC dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ARC-AGI-1 + ConceptARC dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--raw-data-dir", type=str, required=True, help="Raw data directory")
    parser.add_argument("--difficulty", type=str, default=None, help="Difficulty filter (e.g., '10x10', '30x30')")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n-augmentations", type=int, default=8, help="Number of augmentations per task")
    parser.add_argument("--include-evaluation", action="store_true", help="Include ARC-AGI evaluation set")
    parser.add_argument("--include-test", action="store_true", default=True, help="Include test examples")
    parser.add_argument("--no-minimal", action="store_true", help="Exclude ConceptARC MinimalTasks")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default 0.8)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio (default 0.2)")
    
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
    
    logger.info("Starting ARC-AGI-1 + ConceptARC dataset generation")
    logger.info(
        f"Configuration: n_augmentations={args.n_augmentations}, difficulty={args.difficulty}, seed={args.seed}, "
        f"train_ratio={args.train_ratio}, test_ratio={args.test_ratio}"
    )
    
    generator_config = {
        "n_augmentations": args.n_augmentations,
        "include_evaluation": args.include_evaluation,
        "include_test": args.include_test,
        "include_minimal": not args.no_minimal,
        "train_ratio": args.train_ratio,
        "test_ratio": args.test_ratio,
    }
    
    builder = ARCAGI1ConceptARCDatasetBuilder(
        output_dir=Path(args.output_dir),
        raw_data_dir=Path(args.raw_data_dir),
        difficulty=args.difficulty,
        seed=args.seed,
        generator_config=generator_config
    )
    
    metadata = builder.build()
    logger.info(f"Dataset built successfully: {builder.output_dir}")
    logger.info(
        f"Train: {metadata.n_tasks} examples. Total rows and test count: see split summary above."
    )


if __name__ == "__main__":
    main()
