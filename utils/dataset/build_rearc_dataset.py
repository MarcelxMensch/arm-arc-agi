"""RE-ARC dataset builder implementation."""

import json
import os
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Tuple
import logging
import numpy as np

# Handle both relative and absolute imports
try:
    from .build_dataset import DatasetBuilder, DatasetMetadata
    from .common import augment_example
except ImportError:
    # When running as a script directly, use absolute import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.dataset.build_dataset import DatasetBuilder, DatasetMetadata
    from utils.dataset.common import augment_example

logger = logging.getLogger(__name__)


class REARCDatasetBuilder(DatasetBuilder):
    """Builder for RE-ARC (Reverse-Engineering ARC) datasets."""
    
    def __init__(
        self,
        output_dir: Path,
        raw_data_dir: Path,
        difficulty: Optional[str] = None,
        seed: int = 0,
        generator_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RE-ARC dataset builder.
        
        Args:
            output_dir: Base output directory (final dataset will be in a subdirectory)
            raw_data_dir: Directory containing RE-ARC repository
            difficulty: Optional difficulty filter (e.g., "10x10", "30x30")
            seed: Random seed for reproducibility
            generator_config: Should contain:
                - use_already_created_samples: bool (default False)
                - n_tasks: int (default 200)
                - n_examples: int (default 1000) - number of augmentations per task
        """
        super().__init__(output_dir, raw_data_dir, difficulty, seed, generator_config)
        
        self.use_already_created_samples = self.generator_config.get("use_already_created_samples", False)
        self.n_tasks_config = self.generator_config.get("n_tasks", 200)
        self.n_examples_config = self.generator_config.get("n_examples", 1000)
        self.write_softprompt_fields = self.generator_config.get("write_softprompt_fields", False)
        
        # RE-ARC specific paths
        self.rearc_repo_dir = self.raw_data_dir / "RE-ARC"
        self.rearc_tasks_dir = self.rearc_repo_dir / "re_arc" / "tasks"
        
        # Mapping from original_task_id -> numeric puzzle_identifier (1-indexed, 0=blank)
        self._puzzle_id_map: Dict[str, int] = {}
        # For soft prompts: color_map string -> color_identifier (0=identity)
        self._color_map_to_id: Dict[str, int] = {}
        self._next_color_id = 1
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw RE-ARC task data, extracting from zip or generating if necessary."""
        # Ensure RE-ARC repository exists
        if not self.rearc_repo_dir.exists():
            raise FileNotFoundError(
                f"RE-ARC repository not found at {self.rearc_repo_dir}. "
                "Please ensure the RE-ARC submodule is initialized."
            )
        
        # Check if tasks already exist
        tasks_exist = self.rearc_tasks_dir.exists() and any(self.rearc_tasks_dir.glob("*.json"))
        
        if not tasks_exist:
            # Try to extract from re_arc.zip first (contains pre-generated verified samples)
            zip_path = self.rearc_repo_dir / "re_arc.zip"
            if zip_path.exists():
                logger.info(f"Extracting pre-generated samples from {zip_path}")
                self._extract_rearc_zip(zip_path)
            elif not self.use_already_created_samples:
                # Only generate if explicitly requested (not using pre-generated)
                logger.warning(
                    "No pre-generated samples found. Generating new samples "
                    "(this may take a very long time)..."
                )
                self._generate_rearc_tasks()
            else:
                raise FileNotFoundError(
                    f"No task files found in {self.rearc_tasks_dir} and "
                    f"re_arc.zip not found at {zip_path}. "
                    "Please ensure RE-ARC is properly initialized."
                )
        
        # Load all task JSON files
        task_files = sorted(self.rearc_tasks_dir.glob("*.json"))
        if not task_files:
            raise ValueError(f"No task files found in {self.rearc_tasks_dir}")
        
        raw_tasks = []
        for task_file in task_files:
            with open(task_file, "r") as f:
                task_data = json.load(f)
                # Add task_id from filename if not present
                task_id = task_file.stem
                raw_tasks.append({
                    "task_id": task_id,
                    "examples": task_data if isinstance(task_data, list) else task_data.get("examples", [])
                })
        
        # Limit to n_tasks if specified and we have more
        if len(raw_tasks) > self.n_tasks_config:
            logger.info(f"Limiting tasks from {len(raw_tasks)} to {self.n_tasks_config}")
            raw_tasks = raw_tasks[:self.n_tasks_config]
        elif len(raw_tasks) < self.n_tasks_config:
            logger.warning(
                f"Only {len(raw_tasks)} tasks available, but {self.n_tasks_config} requested. "
                f"Using all available tasks."
            )
        
        # Build puzzle_id_map for soft prompts and puzzle_identifier (1-indexed)
        self._puzzle_id_map = {}
        next_id = 1
        for raw_task in raw_tasks:
            task_id = raw_task["task_id"]
            if task_id not in self._puzzle_id_map:
                self._puzzle_id_map[task_id] = next_id
                next_id += 1
        logger.info(f"Built puzzle_id_map with {len(self._puzzle_id_map)} unique task IDs")
        
        logger.info(f"Loaded {len(raw_tasks)} raw RE-ARC tasks from {self.rearc_tasks_dir}")
        return raw_tasks
    
    def _extract_rearc_zip(self, zip_path: Path):
        """Extract re_arc.zip to get pre-generated samples."""
        import zipfile
        
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.rearc_repo_dir)
        
        # Verify extraction was successful
        if not self.rearc_tasks_dir.exists():
            raise RuntimeError(
                f"Extraction completed but tasks directory not found at {self.rearc_tasks_dir}"
            )
        
        task_count = len(list(self.rearc_tasks_dir.glob("*.json")))
        logger.info(f"Extracted {task_count} task files to {self.rearc_tasks_dir}")
    
    def _generate_rearc_tasks(self):
        """Generate RE-ARC tasks using the generate_dataset function."""
        logger.info(f"Generating RE-ARC tasks: n_examples={self.n_examples_config}, seed={self.seed}")
        
        # Add RE-ARC repo to Python path temporarily
        rearc_repo_path = str(self.rearc_repo_dir)
        if rearc_repo_path not in sys.path:
            sys.path.insert(0, rearc_repo_path)
        
        try:
            from main import generate_dataset
            
            # Change to RE-ARC repo directory for generation
            original_cwd = Path.cwd()
            try:
                os.chdir(self.rearc_repo_dir)
                generate_dataset(n_examples=self.n_examples_config, seed=self.seed)
            finally:
                os.chdir(original_cwd)
            
            logger.info(f"RE-ARC tasks generated in {self.rearc_tasks_dir}")
        except ImportError as e:
            raise ImportError(
                f"Could not import generate_dataset from RE-ARC repository. "
                f"Make sure RE-ARC is properly cloned at {self.rearc_repo_dir}. "
                f"Error: {e}"
            )
        finally:
            # Remove from path
            if rearc_repo_path in sys.path:
                sys.path.remove(rearc_repo_path)
    
    def _get_n_tasks(self, raw_data: List[Any]) -> int:
        """Get number of raw tasks from the dataset."""
        return len(raw_data)
    
    def _get_n_augmentations(self, raw_data: List[Any]) -> int:
        """Get number of augmentations per task from the dataset."""
        # n_examples in RE-ARC config is the number of augmentations per task
        return self.n_examples_config
    
    def _parse_aug_repr(self, aug_repr: str) -> Tuple[int, Optional[str]]:
        """Parse aug_repr from augment_example (e.g. 't3_0123456789') -> (transform_id, color_map_string)."""
        try:
            parts = aug_repr.split("_", 1)
            transform_id = int(parts[0].lstrip("t"))
            color_map_string = parts[1] if len(parts) > 1 else None
            return (transform_id, color_map_string)
        except (ValueError, IndexError):
            return (0, None)
    
    def _get_color_identifier(self, color_map_string: Optional[str]) -> Optional[int]:
        """Get or create color_identifier for a color_map_string. Returns 0 for identity when write_softprompt_fields."""
        if not self.write_softprompt_fields:
            return None
        if color_map_string is None:
            return 0
        if color_map_string not in self._color_map_to_id:
            self._color_map_to_id[color_map_string] = self._next_color_id
            self._next_color_id += 1
        return self._color_map_to_id[color_map_string]
    
    def _generate_tasks(self, raw_data: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """
        Generate tasks from raw RE-ARC data with optional augmentations.
        
        Each raw task is augmented n_examples times, and optionally filtered by difficulty.
        RE-ARC tasks already contain examples from generation, which we use as augmentations.
        When write_softprompt_fields is True, applies dihedral+color augmentation and records
        task_identifier, transform_identifier, color_identifier for soft-prompt training.
        """
        import random
        
        rng_random = random.Random(self.seed)
        
        for raw_task in raw_data:
            task_id = raw_task.get("task_id", "unknown")
            puzzle_identifier = self._puzzle_id_map.get(task_id, 0)
            
            # RE-ARC tasks contain examples that serve as augmentations
            examples = raw_task.get("examples", [])
            
            if not examples:
                logger.warning(f"Task {task_id} has no examples, skipping")
                continue
            
            # Apply difficulty filtering if specified (check first example)
            if self.difficulty and not self._matches_difficulty_from_examples(examples, self.difficulty):
                logger.debug(f"Task {task_id} filtered out by difficulty {self.difficulty}")
                continue
            
            # Use examples as augmentations
            if len(examples) >= self.n_examples_config:
                selected_examples = examples[:self.n_examples_config]
            else:
                selected_examples = [rng_random.choice(examples) for _ in range(self.n_examples_config)]
            
            for i, example in enumerate(selected_examples):
                input_grid = example.get("input", [])
                output_grid = example.get("output", [])
                
                if self.write_softprompt_fields:
                    # Apply dihedral + color augmentation and record identifiers (same as ARC-AGI-2)
                    rng_np = np.random.default_rng(self.seed + hash(task_id) % (2**32) + i)
                    input_grid, output_grid, transform_id, color_map_list, aug_repr = augment_example(input_grid, output_grid, rng_np)
                    color_map_string = ''.join(str(x) for x in color_map_list)
                    color_id = self._get_color_identifier(color_map_string)
                
                augmented_task = {
                    "task_id": f"{task_id}_{i}",
                    "input": input_grid,
                    "output": output_grid,
                    "original_task_id": task_id,
                    "augmentation_index": i,
                    "puzzle_identifier": puzzle_identifier,
                }
                if self.write_softprompt_fields:
                    augmented_task["task_identifier"] = puzzle_identifier
                    augmented_task["transform_identifier"] = transform_id
                    augmented_task["color_identifier"] = color_id
                yield augmented_task
    
    def _matches_difficulty_from_examples(self, examples: List[Dict[str, Any]], difficulty: str) -> bool:
        """
        Check if task examples match difficulty filter.
        
        Difficulty format examples: "10x10", "30x30", etc.
        Filters tasks where grid dimensions exceed the specified max.
        """
        if not examples:
            return False
        
        # Check first example's input grid size
        first_example = examples[0]
        input_grid = first_example.get("input", [])
        
        if not input_grid:
            return False
        
        height = len(input_grid)
        width = len(input_grid[0]) if input_grid else 0
        
        # Parse difficulty (e.g., "10x10" -> max 10x10, "30x30" -> max 30x30)
        if "x" in difficulty:
            try:
                max_dim = int(difficulty.split("x")[0])
                return height <= max_dim and width <= max_dim
            except ValueError:
                logger.warning(f"Invalid difficulty format: {difficulty}")
                return True  # Don't filter if format is invalid
        else:
            # If format is unclear, don't filter
            logger.warning(f"Unrecognized difficulty format: {difficulty}")
            return True
    
    def _task_to_jsonl(self, task: Dict[str, Any]) -> str:
        """Convert a task dictionary to a JSONL string."""
        return json.dumps(task, ensure_ascii=False)
    
    def _get_source_data_name(self) -> str:
        """Return the name of the source dataset."""
        return "re-arc"
    
    def _create_metadata_for_split(self, split: str, n_tasks_in_split: int, num_shards: int) -> DatasetMetadata:
        """Add soft prompt table sizes to metadata when write_softprompt_fields was used."""
        base = super()._create_metadata_for_split(split, n_tasks_in_split, num_shards)
        if not self.write_softprompt_fields:
            return base
        num_task = (len(self._puzzle_id_map) + 1) if self._puzzle_id_map else None
        num_color = getattr(self, "_next_color_id", 1)
        return DatasetMetadata(
            difficulty=base.difficulty,
            seed=base.seed,
            n_tasks=base.n_tasks,
            n_augmentations=base.n_augmentations,
            shard_size=base.shard_size,
            num_shards=base.num_shards,
            generator_config=base.generator_config,
            source_data=base.source_data,
            split=base.split,
            num_task_identifiers=num_task,
            num_color_identifiers=num_color,
        )


def main():
    """Main entry point for RE-ARC dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build RE-ARC dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--raw-data-dir", type=str, required=True, help="Raw data directory containing RE-ARC repository")
    parser.add_argument("--difficulty", type=str, default=None, help="Difficulty filter (e.g., '10x10', '30x30')")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use-already-created-samples", action="store_true", help="Use existing samples instead of generating new ones")
    parser.add_argument("--n-tasks", type=int, default=200, help="Number of raw tasks")
    parser.add_argument("--n-examples", type=int, default=1000, help="Number of augmentations per task")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default 0.8)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio (default 0.2)")
    parser.add_argument("--write-softprompt-fields", action="store_true", help="Write soft prompt fields (task_identifier, transform_identifier, color_identifier) and apply dihedral+color augmentation")
    
    args = parser.parse_args()
    
    # Configure logging to stdout/stderr only (sbatch captures these to .out/.err files)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    logger.info("Starting RE-ARC dataset generation")
    logger.info(
        f"Configuration: n_tasks={args.n_tasks}, n_examples={args.n_examples}, difficulty={args.difficulty}, "
        f"seed={args.seed}, train_ratio={args.train_ratio}, test_ratio={args.test_ratio}, "
        f"write_softprompt_fields={args.write_softprompt_fields}"
    )
    
    generator_config = {
        "use_already_created_samples": args.use_already_created_samples,
        "n_tasks": args.n_tasks,
        "n_examples": args.n_examples,
        "train_ratio": args.train_ratio,
        "test_ratio": args.test_ratio,
        "write_softprompt_fields": args.write_softprompt_fields,
    }
    
    builder = REARCDatasetBuilder(
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
