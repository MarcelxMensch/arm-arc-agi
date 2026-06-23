"""Training script for hierarchical reasoning models on ARC-style tasks.

Supports both single-GPU (``python3 train.py``) and multi-GPU DDP
(``torchrun --nproc-per-node N train.py``).  DDP is auto-detected via
the ``LOCAL_RANK`` environment variable set by torchrun.
"""

from typing import Optional, Any, Sequence, List, Tuple, Dict
from dataclasses import dataclass
import os
import sys
import math
import yaml
import shutil
import copy
import logging
import importlib
from pathlib import Path
from datetime import timedelta
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

from evaluate import evaluate, log_train_images_batch

try:
    from utils.dataset.common import dihedral_transform, color_permutation
except ImportError:
    dihedral_transform = None
    color_permutation = None
from utils.dataset.common import (
    seq_to_grid,
    pad_grid_to_size,
    build_composite_image,
    GRID_TRANSPARENT_ID,
)

# Setup logging to stderr so training steps appear in .err (with wandb); .out keeps stdout (e.g. shell)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# #region agent log
DEBUG_LOG_PATH = os.environ.get("ARM_DEBUG_LOG_PATH", "")
DEBUG_SESSION_ID = os.environ.get("ARM_DEBUG_SESSION_ID", "")


def _debug_emit(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    """Append one NDJSON debug entry for runtime diagnosis."""
    if not DEBUG_LOG_PATH:
        return
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _inspect_dataset_path(path_str: str, sample_rows: int = 256) -> Dict[str, Any]:
    """Collect lightweight structural stats for dataset parity debugging."""
    p = Path(path_str)
    out: Dict[str, Any] = {"path": path_str, "exists": p.exists(), "layout": "unknown"}
    if not p.exists():
        return out

    train_dir = p / "train"
    root = train_dir if train_dir.is_dir() else p
    out["has_train_dir"] = train_dir.is_dir()
    out["root"] = str(root)
    out["has_meta"] = (root / "meta.yaml").exists()
    out["has_index"] = (root / "index.jsonl").exists()

    npy_files = sorted(root.glob("*.npy"))
    jsonl_files = sorted(root.glob("*.jsonl"))
    out["npy_files"] = len(npy_files)
    out["jsonl_files"] = len(jsonl_files)

    if npy_files and not (root / "index.jsonl").exists():
        out["layout"] = "legacy_npy"
        out["sample_files"] = [f.name for f in npy_files[:5]]
        return out

    if (root / "index.jsonl").exists():
        out["layout"] = "sharded_jsonl"
        total_rows = 0
        shard_paths: List[Path] = []
        try:
            with open(root / "index.jsonl") as f:
                for line in f:
                    e = json.loads(line)
                    total_rows += int(e.get("num_tasks", 0))
                    if len(shard_paths) < 3:
                        shard_paths.append(root / e["shard_path"])
        except Exception:
            pass
        out["index_total_rows"] = total_rows
        out["sample_shards"] = [s.name for s in shard_paths]

        # Sample content-level schema/identifier stats
        sampled = 0
        key_union: set = set()
        unique_puzzle_ids: set = set()
        unique_original_ids: set = set()
        train_pairs_sum = 0
        test_pairs_sum = 0
        with_demo = 0
        with_transform = 0
        with_color = 0
        for sp in shard_paths:
            if not sp.exists():
                continue
            with open(sp) as f:
                for line in f:
                    if sampled >= sample_rows:
                        break
                    row = json.loads(line)
                    sampled += 1
                    key_union.update(row.keys())
                    if "puzzle_identifier" in row:
                        unique_puzzle_ids.add(int(row["puzzle_identifier"]))
                    if "original_task_id" in row:
                        unique_original_ids.add(str(row["original_task_id"]))
                    if isinstance(row.get("train"), list):
                        train_pairs_sum += len(row["train"])
                    if isinstance(row.get("test"), list):
                        test_pairs_sum += len(row["test"])
                    if "demo_inputs" in row and "demo_outputs" in row:
                        with_demo += 1
                    if "transform_id" in row:
                        with_transform += 1
                    if "color_map" in row:
                        with_color += 1
            if sampled >= sample_rows:
                break

        out["sampled_rows"] = sampled
        out["sample_keys"] = sorted(list(key_union))
        out["unique_puzzle_ids_in_sample"] = len(unique_puzzle_ids)
        out["unique_original_task_ids_in_sample"] = len(unique_original_ids)
        out["puzzle_id_reuse_ratio"] = (
            1.0 - (len(unique_puzzle_ids) / max(sampled, 1))
            if sampled > 0 and len(unique_puzzle_ids) > 0
            else 0.0
        )
        out["avg_train_pairs_per_row"] = train_pairs_sum / max(sampled, 1)
        out["avg_test_pairs_per_row"] = test_pairs_sum / max(sampled, 1)
        out["rows_with_demo_fields"] = with_demo
        out["rows_with_transform_id"] = with_transform
        out["rows_with_color_map"] = with_color
        return out

    return out
# #endregion

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Import dataset and model utilities
from utils.dataset.puzzle_dataset import (
    PuzzleDataset,
    PuzzleDatasetConfig,
    PuzzleDatasetMetadata,
    ARC_MAX_GRID_SIZE,
    MAX_DEMOS as PUZZLE_MAX_DEMOS,
)


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig
    debug: bool = False


class PretrainConfig(pydantic.BaseModel):
    """Main training configuration."""
    # Architecture
    arch: ArchConfig
    
    # Data paths
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Optional reference dataset paths for parity debugging (e.g. old HRM pipeline)
    reference_data_paths: List[str] = []
    
    # Training hyperparameters
    global_batch_size: int
    epochs: int
    
    lr: float
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 100
    
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Puzzle embedding training
    puzzle_emb_lr: float = 1e-2
    puzzle_emb_weight_decay: float = 0.1
    
    # Naming and checkpointing
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    resume: Optional[str] = None  # Output dir; load latest epoch model + train state and continue
    checkpoint_path: Optional[str] = None
    # Single output folder for this script run (config.yaml + weights); overwritten on rerun.
    output_dir: str = "outputs/train-hrm-re-arc"
    
    # Training settings
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: int = 0
    eval_max_batches: Optional[int] = None  # cap validation batches (e.g. 20); None = full eval set
    eval_epoch_interval: int = 1  # run eval every N epochs (set to 10 for long training runs)

    # EMA
    ema: bool = False
    ema_rate: float = 0.999

    # Test-time training (TTT): adapt on task demos then predict test example
    ttt_enabled: bool = False
    ttt_mode: str = "full"  # "encoder" (grid encoder only) or "full" (whole model)
    ttt_steps: int = 100
    ttt_lr: float = 1e-3
    ttt_augment: bool = True  # each TTT step use random augmentation of demos

    # Debug mode (extra prints, more metrics; does not affect WandB)
    debug: bool = False
    # WandB: set True to disable logging to wandb.ai (e.g. offline runs)
    wandb_disabled: bool = False
    # Image logging: composite [input|pred|target]; Train/images (sparse), Eval/val_images, Eval/final_images
    log_eval_images: bool = True
    eval_val_max_images: int = 8
    eval_final_max_images: int = 64
    log_train_images: bool = True
    train_images_max: int = 4
    # When debug=True, log train/images every this many steps (0 = off).
    train_images_log_interval: int = 0
    max_images_per_step: int = 3
    # Grid encoding for image logging (from config; must match dataset)
    grid_max_size: int = 30
    grid_pad_id: int = 0
    grid_eos_id: int = 1
    grid_vocab_offset: int = 2
    # Soft-prompt embedding logging: interval in steps (None = off).
    log_softprompt_embeddings_interval: Optional[int] = None


@dataclass
class DDPInfo:
    """DDP runtime state (trivial when running single-GPU)."""
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_ddp: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_ddp() -> DDPInfo:
    """Detect torchrun environment and initialise DDP if present."""
    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        return DDPInfo()

    local_rank = int(local_rank_str)
    torch.cuda.set_device(local_rank)
    ddp_timeout_sec = int(os.environ.get("TORCH_DDP_TIMEOUT_SEC", "7200"))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=ddp_timeout_sec))
    info = DDPInfo(
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        is_ddp=True,
    )
    logger.info(
        f"DDP initialised: rank {info.rank}/{info.world_size}, "
        f"local_rank {info.local_rank}, device cuda:{info.local_rank}, "
        f"timeout={ddp_timeout_sec}s"
    )
    return info


@dataclass
class TrainState:
    """Holds the training state."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    step: int
    total_steps: int
    carry: Any = None


def load_model_class(name: str):
    """Dynamically load a model class from a module path."""
    # Format: module.path@ClassName
    if "@" in name:
        module_path, class_name = name.split("@")
    else:
        module_path = name
        class_name = name.split(".")[-1]
    
    # Try different module prefixes (utils.model. for short names like recursive_reasoning.hrm)
    prefixes = ["utils.model.", ""]
    
    for prefix in prefixes:
        try:
            full_path = prefix + module_path
            module = importlib.import_module(full_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    
    raise ImportError(f"Could not load class: {name}")


def create_dataloader(
    config: PretrainConfig,
    split: str,
    ddp: Optional[DDPInfo] = None,
    test_set_mode: bool = False,
    epochs_per_iter: int = 1,
) -> tuple:
    """Create a DataLoader for the specified split."""
    if ddp is None:
        ddp = DDPInfo()
    data_paths = config.data_paths_test if test_set_mode and config.data_paths_test else config.data_paths
    
    logger.info(f"Creating {split} dataloader with {len(data_paths)} path(s): {data_paths}")
    
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=data_paths,
            rank=ddp.rank,
            num_replicas=ddp.world_size,
            epochs_per_iter=epochs_per_iter,
            global_batch_size=config.global_batch_size,
            test_set_mode=test_set_mode,
        ),
        split=split,
    )
    
    logger.info(f"Dataset: vocab_size={dataset.metadata.vocab_size}, "
                f"seq_len={dataset.metadata.seq_len}, "
                f"num_puzzle_identifiers={dataset.metadata.num_puzzle_identifiers}, "
                f"total_puzzles={dataset.metadata.total_puzzles}")
    if dataset.metadata.pad_id is not None:
        logger.info("Dataset layout: legacy (dataset.json + all__*.npy)")
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    return dataloader, dataset.metadata


def create_model(
    config: PretrainConfig,
    metadata: PuzzleDatasetMetadata,
    ddp: Optional[DDPInfo] = None,
) -> nn.Module:
    """Create and initialize the model."""
    if ddp is None:
        ddp = DDPInfo()
    per_gpu_batch = config.global_batch_size // ddp.world_size
    logger.info(f"Creating model: arch={config.arch.name}, loss={config.arch.loss.name}")
    
    # Build model config
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=per_gpu_batch,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
        debug=config.debug or config.arch.debug,
    )
    # Soft prompt table sizes from dataset meta (when built with --write-softprompt-fields)
    if getattr(metadata, "num_task_identifiers", None) is not None:
        model_cfg["num_task_identifiers"] = metadata.num_task_identifiers
    if getattr(metadata, "num_color_identifiers", None) is not None:
        model_cfg["num_color_identifiers"] = metadata.num_color_identifiers
    
    # Load model and loss head classes
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        logger.info(f"Model architecture:\n{model}")
        
        # Wrap with loss head
        loss_cfg = config.arch.loss.__pydantic_extra__ or {}
        model = loss_head_cls(model, **loss_cfg)
        
        # Compile model (unless disabled or unavailable)
        if "DISABLE_COMPILE" not in os.environ and not config.debug:
            try:
                logger.info("Compiling model with torch.compile")
                model = torch.compile(model)
            except Exception as e:
                logger.warning(f"torch.compile failed, running without compilation: {e}")
        else:
            logger.info("Model compilation disabled")
        
        # Load checkpoint if specified (resume is handled in main after create_model)
        if config.load_checkpoint and not config.resume:
            load_checkpoint(model, config.load_checkpoint)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")
    
    return model


def create_optimizer(config: PretrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer for the model."""
    logger.info(
        "Creating optimizer: lr=%s, weight_decay=%s, puzzle_emb_lr=%s, puzzle_emb_weight_decay=%s",
        config.lr,
        config.weight_decay,
        config.puzzle_emb_lr,
        config.puzzle_emb_weight_decay,
    )

    named_params = list(model.named_parameters())
    puzzle_params = []
    base_params = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if "puzzle_emb" in name:
            puzzle_params.append(p)
        else:
            base_params.append(p)

    param_groups = []
    if base_params:
        param_groups.append(
            {
                "params": base_params,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
            }
        )
    if puzzle_params:
        param_groups.append(
            {
                "params": puzzle_params,
                "lr": config.puzzle_emb_lr,
                "weight_decay": config.puzzle_emb_weight_decay,
            }
        )
    logger.info(
        "Optimizer param groups: base=%d, puzzle=%d",
        len(base_params),
        len(puzzle_params),
    )

    return torch.optim.AdamW(
        param_groups,
        betas=(config.beta1, config.beta2),
    )


CHECKPOINTS_SUBDIR = "checkpoints"


def _checkpoints_dir(output_dir: str) -> str:
    """Return output_dir/checkpoints/."""
    return os.path.join(output_dir, CHECKPOINTS_SUBDIR)


def _weights_path(ckpt_dir: str) -> tuple:
    """Return (safetensors_path, pt_path) for the weights file in the given checkpoint dir."""
    return (
        os.path.join(ckpt_dir, "model.safetensors"),
        os.path.join(ckpt_dir, "model.pt"),
    )


def _latest_epoch_dir(checkpoints_dir: str) -> Optional[str]:
    """Return the path to the latest epoch_N folder in checkpoints_dir, or None."""
    if not os.path.isdir(checkpoints_dir):
        return None
    prefix = "epoch_"
    best = None
    best_num = -1
    for name in os.listdir(checkpoints_dir):
        if name.startswith(prefix):
            try:
                n = int(name[len(prefix):])
                if n > best_num:
                    best_num = n
                    best = os.path.join(checkpoints_dir, name)
            except ValueError:
                continue
    return best


def load_checkpoint(model: nn.Module, checkpoint_path: str):
    """Load model weights from a file or from a checkpoint dir (output_dir, or output_dir/checkpoints/, or output_dir/checkpoints/epoch_N)."""
    path = checkpoint_path
    if os.path.isdir(path):
        ckpt_base = _checkpoints_dir(path)
        # Prefer path as exact epoch dir, then latest epoch in checkpoints/, then flat checkpoints/
        candidates = [path]
        if os.path.isdir(ckpt_base):
            latest = _latest_epoch_dir(ckpt_base)
            if latest:
                candidates.insert(0, latest)
        candidates.append(ckpt_base)
        for ckpt_dir in candidates:
            if not os.path.isdir(ckpt_dir):
                continue
            sf_path, pt_path = _weights_path(ckpt_dir)
            if os.path.isfile(sf_path):
                path = sf_path
                break
            if os.path.isfile(pt_path):
                path = pt_path
                break
        else:
            raise FileNotFoundError(f"No model.safetensors or model.pt in {path} or under {ckpt_base}")
    logger.info(f"Loading checkpoint from {path}")
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(path, device="cuda")
        except ImportError:
            raise ImportError("safetensors required to load .safetensors; install with: pip install safetensors")
        model.load_state_dict(state_dict, strict=False)
    else:
        state_dict = torch.load(path, map_location="cuda")
        model.load_state_dict(state_dict, strict=False)
    logger.info("Checkpoint loaded successfully")


TRAIN_STATE_FILENAME = "train_state.pt"


def _resume_latest_epoch_dir(resume_path: str) -> Optional[str]:
    """Return the path to the latest epoch dir under resume_path (output dir or checkpoints dir)."""
    if not os.path.isdir(resume_path):
        return None
    ckpt_base = _checkpoints_dir(resume_path)
    if os.path.isdir(ckpt_base):
        return _latest_epoch_dir(ckpt_base)
    if os.path.basename(resume_path.rstrip(os.sep)) == CHECKPOINTS_SUBDIR:
        return _latest_epoch_dir(resume_path)
    return None


def load_train_state(optimizer: torch.optim.Optimizer, ckpt_dir: str) -> Tuple[int, int, Optional[str]]:
    """Load epoch (1-based), step, and optional wandb_run_id from ckpt_dir/train_state.pt; restore optimizer. Returns (epoch, step, wandb_run_id)."""
    path = os.path.join(ckpt_dir, TRAIN_STATE_FILENAME)
    if not os.path.isfile(path):
        return 0, 0, None
    data = torch.load(path, map_location="cuda")
    epoch = int(data.get("epoch", 0))
    step = int(data.get("step", 0))
    wandb_run_id = data.get("wandb_run_id")
    if isinstance(wandb_run_id, str):
        pass
    else:
        wandb_run_id = None
    opt_state = data.get("optimizer_state_dict")
    if opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
            logger.info("Optimizer state restored from checkpoint")
        except Exception as e:
            logger.warning(f"Could not restore optimizer state: {e}")
    return epoch, step, wandb_run_id


def save_checkpoint(config: PretrainConfig, train_state: TrainState, epoch: int):
    """Save model state_dict and train state to output_dir/checkpoints/epoch_N/. One folder per epoch."""
    if config.checkpoint_path is None:
        logger.warning("Output dir not set, skipping save")
        return
    # Per-epoch folder: checkpoints/epoch_1/, checkpoints/epoch_2/, ...
    ckpt_dir = os.path.join(_checkpoints_dir(config.checkpoint_path), f"epoch_{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dict = train_state.model.state_dict()
    sf_path, pt_path = _weights_path(ckpt_dir)
    try:
        from safetensors.torch import save_file
        save_file(state_dict, sf_path)
        logger.info(f"Saving checkpoint to {sf_path}")
        if os.path.isfile(pt_path):
            os.remove(pt_path)
    except ImportError:
        torch.save(state_dict, pt_path)
        logger.info(f"Saving checkpoint to {pt_path}")
    # Save training state for resume (epoch 1-based, step, optimizer, wandb run id)
    train_state_path = os.path.join(ckpt_dir, TRAIN_STATE_FILENAME)
    try:
        opt_state = train_state.optimizer.state_dict()
    except Exception:
        opt_state = None
    wandb_run_id = None
    if wandb.run is not None:
        wandb_run_id = getattr(wandb.run, "id", None)
    torch.save({
        "epoch": epoch,
        "step": train_state.step,
        "optimizer_state_dict": opt_state,
        "wandb_run_id": wandb_run_id,
    }, train_state_path)
    logger.info(f"Saving train state to {train_state_path}")
    logger.info("Checkpoint saved successfully")


def cosine_schedule_with_warmup(
    current_step: int,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.1,
) -> float:
    """Compute learning rate with cosine schedule and warmup."""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: dict,
    global_batch_size: int,
) -> Optional[dict]:
    """Train on a single batch."""
    train_state.step += 1
    
    if train_state.step > train_state.total_steps:
        return None
    
    # Move batch to GPU (skip non-tensors e.g. original_task_ids)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    # Reset carry every batch so current_data always comes from this batch's labels.
    # DDP wraps the model and does not expose custom methods; use .module when wrapped.
    unwrap = getattr(train_state.model, "module", train_state.model)
    with torch.device("cuda"):
        train_state.carry = unwrap.initial_carry(batch)

    # Forward pass (use DDP wrapper so gradients are reduced across ranks).
    # Unroll ACT steps on the same batch until all sequences halt (or max steps),
    # mirroring evaluation behavior and restoring q-continue / halting dynamics.
    max_act_steps = max(1, int(getattr(config.arch, "halt_max_steps", 1)))
    actual_batch_size = batch["inputs"].shape[0] if torch.is_tensor(batch.get("inputs")) else 0
    metrics = None
    metrics_accum: dict = {}
    act_steps = 0
    all_finish = False
    while not all_finish and act_steps < max_act_steps:
        train_state.carry, step_loss, step_metrics, _, all_finish = train_state.model(
            carry=train_state.carry, batch=batch, return_keys=[]
        )
        metrics = step_metrics
        for mk, mv in step_metrics.items():
            if torch.is_tensor(mv):
                mval = mv
            else:
                mval = torch.tensor(float(mv), device=step_loss.device)
            metrics_accum[mk] = mval if mk not in metrics_accum else (metrics_accum[mk] + mval)
        # Backprop per ACT step to avoid keeping all step graphs in memory.
        (step_loss / (global_batch_size * max_act_steps)).backward()
        act_steps += 1
    if metrics is None:
        return None
    # For logging, average step-wise metrics across the ACT rollout.
    if metrics is not None and act_steps > 1:
        merged_metrics = {}
        for mk, mv in metrics.items():
            if mk in ("count", "count_all"):
                merged_metrics[mk] = mv
            else:
                merged_metrics[mk] = metrics_accum[mk] / act_steps
        metrics = merged_metrics
    
    # Update learning rate
    lr = cosine_schedule_with_warmup(
        train_state.step,
        config.lr,
        config.lr_warmup_steps,
        train_state.total_steps,
        config.lr_min_ratio,
    )
    
    for param_group in train_state.optimizer.param_groups:
        param_group['lr'] = lr
    
    # Optimizer step
    train_state.optimizer.step()
    train_state.optimizer.zero_grad()
    
    # Process metrics
    if metrics:
        count = max(metrics.get("count", torch.tensor(1)).item(), 1)
        count_all = max(
            metrics.get("count_all", torch.tensor(1)).item() if isinstance(metrics.get("count_all"), torch.Tensor) else metrics.get("count_all", 1),
            1,
        )
        result_metrics = {}
        
        # Loss from the model is sum over this rank's batch; normalize by actual_batch_size so
        # logged value is per-example average (matches single-GPU and reference scale). For DDP,
        # global_batch_size is the logical total but each rank only has actual_batch_size examples.
        loss_norm = actual_batch_size
        for k, v in metrics.items():
            # Handle debug shape strings (already prefixed with "Train/")
            if k.startswith("Train/"):
                # Don't divide strings or layer counts
                if isinstance(v, str) or k.endswith("_layers"):
                    result_metrics[k] = v
                elif isinstance(v, torch.Tensor):
                    result_metrics[k] = v.item()
                else:
                    result_metrics[k] = v
            elif isinstance(v, torch.Tensor):
                val = v.item()
                if k.endswith("loss"):
                    result_metrics[f"Train/{k}"] = val / loss_norm
                elif k not in ("count", "count_all"):
                    # Do NOT normalize z_* norms by count; match original pretrain behavior
                    if k in ("z_H_norm", "z_L_norm", "z_H_delta_norm"):
                        result_metrics[f"Train/{k}"] = val
                    else:
                        norm_count = count_all if k in ("accuracy", "exact_accuracy") else count
                        result_metrics[f"Train/{k}"] = val / norm_count
            elif k not in ("count", "count_all"):
                # Non-tensor, non-string values (like int)
                if k.endswith("loss"):
                    result_metrics[f"Train/{k}"] = v / loss_norm
                else:
                    if k in ("z_H_norm", "z_L_norm", "z_H_delta_norm"):
                        result_metrics[f"Train/{k}"] = v
                    else:
                        norm_count = count_all if k in ("accuracy", "exact_accuracy") else count
                        result_metrics[f"Train/{k}"] = v / norm_count
        
        result_metrics["Train/act_steps"] = float(act_steps)
        result_metrics["Train/lr"] = lr
        result_metrics["Train/step"] = train_state.step

        return result_metrics
    
    return None


def save_config(config: PretrainConfig):
    """Save config.yaml to output dir (always when checkpoint_path set). Log code to wandb if run active."""
    if config.checkpoint_path is None:
        return
    
    os.makedirs(config.checkpoint_path, exist_ok=True)
    config_file = os.path.join(config.checkpoint_path, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config.model_dump(), f)
    logger.info(f"Config saved to {config_file}")
    if wandb.run is not None:
        wandb.run.log_code(config.checkpoint_path)


def _ttt_augment_demos(
    demo_in: torch.Tensor,
    demo_out: torch.Tensor,
    n_demos: int,
    step_seed: int,
    grid_vocab_offset: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, int, np.ndarray]:
    """Apply random dihedral + color permutation to demo grids (encoded 0,1,2..11). Returns (aug_in, aug_out, transform_id, color_map)."""
    if dihedral_transform is None or color_permutation is None:
        return demo_in, demo_out, 0, np.arange(10, dtype=np.int64)
    rng = np.random.default_rng(step_seed)
    transform_id = int(rng.integers(0, 8))
    color_map = color_permutation(rng)  # (10,) permutation
    g_in = demo_in.cpu().numpy()[:n_demos]
    g_out = demo_out.cpu().numpy()[:n_demos]
    aug_in_list = []
    aug_out_list = []
    for d in range(n_demos):
        for g, out_list in [(g_in[d], aug_in_list), (g_out[d], aug_out_list)]:
            g_spatial = dihedral_transform(g.astype(np.uint8), transform_id)
            g_color = np.where(g_spatial >= grid_vocab_offset,
                               grid_vocab_offset + color_map[np.clip(g_spatial.astype(np.int32) - grid_vocab_offset, 0, 9)],
                               g_spatial)
            out_list.append(g_color)
    aug_in_t = torch.tensor(np.stack(aug_in_list), dtype=demo_in.dtype, device=demo_in.device)
    aug_out_t = torch.tensor(np.stack(aug_out_list), dtype=demo_out.dtype, device=demo_out.device)
    return aug_in_t, aug_out_t, transform_id, color_map


def evaluate_with_ttt(
    model: nn.Module,
    eval_loader: DataLoader,
    ttt_steps: int,
    ttt_lr: float,
    ttt_mode: str,
    device: torch.device = torch.device("cuda"),
    debug: bool = False,
    max_batches: Optional[int] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    grid_max_size: int = 30,
    grid_pad_id: int = 0,
    grid_eos_id: int = 1,
    grid_vocab_offset: int = 2,
    ttt_augment: bool = True,
    log_images: bool = False,
    image_namespace: str = "TTT/val_images",
    max_images: int = 8,
    max_images_per_step: int = 3,
) -> dict:
    """Evaluate with test-time training: per-task adaptation on demos before prediction.

    For each task (batch_size=1 expected, or all examples share the same task):
    1. Deep-copy the model (or just encoder params if ttt_mode='encoder').
    2. Fine-tune on the task's demo pairs for ttt_steps gradient steps (optionally with random augmentations each step).
    3. Predict the test example(s) with the adapted model.
    4. Discard the clone to restore original weights.
    """
    model.eval()
    metrics_accum: dict = {}
    total_batches = 0
    total_correct = 0
    total_examples = 0
    ttt_composites: List[Any] = []
    ttt_task_ids: List[Optional[str]] = []

    for set_name, batch, global_batch_size in eval_loader:
        if max_batches is not None and total_batches >= max_batches:
            break
        total_batches += 1

        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        # Build TTT training data from demo grids (if present)
        has_demos = "demo_inputs" in batch and "demo_outputs" in batch and "num_demos" in batch

        infer_model = model
        model_clone = None
        ttt_optimizer = None

        if has_demos:
            # Concept/task-level data may lack transform_ids and color_maps; use identity so softprompt path works
            B = batch["demo_inputs"].shape[0]
            dev = batch["demo_inputs"].device
            if "transform_ids" not in batch:
                batch["transform_ids"] = torch.zeros(B, dtype=torch.long, device=dev)
            if "color_maps" not in batch:
                batch["color_maps"] = torch.arange(10, dtype=torch.long, device=dev).unsqueeze(0).expand(B, 10)

            # Clone model for per-task adaptation
            model_clone = copy.deepcopy(model)
            model_clone.train()

            # Select parameters to optimize
            if ttt_mode == "encoder":
                params_to_opt = []
                for name, p in model_clone.named_parameters():
                    if "grid_encoder" in name:
                        p.requires_grad_(True)
                        params_to_opt.append(p)
                    else:
                        p.requires_grad_(False)
            else:
                params_to_opt = [p for p in model_clone.parameters() if p.requires_grad]

            if params_to_opt:
                ttt_optimizer = torch.optim.Adam(params_to_opt, lr=ttt_lr)

                # Build a mini-batch from the demo grids for TTT training.
                b_demo_in = batch["demo_inputs"][0]   # (N, H, W)
                b_demo_out = batch["demo_outputs"][0]  # (N, H, W)
                n_demos = batch["num_demos"][0].item()

                demo_inputs_flat = b_demo_in[:n_demos].reshape(n_demos, -1)
                demo_outputs_flat = b_demo_out[:n_demos].reshape(n_demos, -1)

                ttt_batch = {
                    "inputs": demo_inputs_flat,
                    "labels": demo_outputs_flat,
                    "puzzle_identifiers": batch["puzzle_identifiers"][0:1].expand(n_demos),
                }
                for key in ("demo_inputs", "demo_outputs", "num_demos", "transform_ids", "color_maps", "task_identifiers"):
                    if key in batch:
                        val = batch[key][0:1]
                        ttt_batch[key] = val.expand(n_demos, *val.shape[1:]) if val.dim() > 0 else val.expand(n_demos)

                # TTT gradient steps
                for ttt_step in range(ttt_steps):
                    if ttt_augment and dihedral_transform is not None and color_permutation is not None:
                        step_seed = (epoch or 0) * 1000000 + (step or 0) * 1000 + total_batches * 100 + ttt_step
                        aug_in, aug_out, tid, cmap = _ttt_augment_demos(
                            b_demo_in, b_demo_out, n_demos, step_seed, grid_vocab_offset
                        )
                        demo_in_4d = torch.zeros(
                            n_demos, PUZZLE_MAX_DEMOS, ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE,
                            dtype=aug_in.dtype, device=device,
                        )
                        demo_out_4d = torch.zeros(
                            n_demos, PUZZLE_MAX_DEMOS, ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE,
                            dtype=aug_out.dtype, device=device,
                        )
                        for j in range(n_demos):
                            demo_in_4d[:, j, :, :] = aug_in[j]
                            demo_out_4d[:, j, :, :] = aug_out[j]
                        ttt_batch["demo_inputs"] = demo_in_4d
                        ttt_batch["demo_outputs"] = demo_out_4d
                        ttt_batch["inputs"] = aug_in.reshape(n_demos, -1)
                        ttt_batch["labels"] = aug_out.reshape(n_demos, -1)
                        ttt_batch["transform_ids"] = torch.full((n_demos,), tid, dtype=torch.long, device=device)
                        ttt_batch["color_maps"] = torch.tensor(cmap, dtype=torch.long, device=device).unsqueeze(0).expand(n_demos, 10)
                    ttt_optimizer.zero_grad()
                    unwrap = getattr(model_clone, "module", model_clone)
                    with torch.device(device):
                        carry = unwrap.initial_carry(ttt_batch)
                    carry, loss, _metrics, _, _ = model_clone(carry=carry, batch=ttt_batch, return_keys=[])
                    (loss / max(n_demos, 1)).backward()
                    ttt_optimizer.step()

            infer_model = model_clone if model_clone is not None else model
            infer_model.eval()

        # Predict (adapted model when demos exist; base model otherwise)
        with torch.inference_mode():
            unwrap = getattr(infer_model, "module", infer_model)
            with torch.device(device):
                carry = unwrap.initial_carry(batch)
            # Run full inference steps
            all_finish = False
            while not all_finish:
                carry, loss, metrics, _detached, all_finish = infer_model(
                    carry=carry, batch=batch, return_keys=["logits"]
                )

        if metrics:
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    val = v.item()
                else:
                    val = v
                metrics_accum[k] = metrics_accum.get(k, 0.0) + val

        # Check exact accuracy from logits
        if "logits" in _detached if _detached else {}:
            preds = _detached["logits"].argmax(dim=-1)
            labels = batch["labels"]
            correct = (preds == labels).all(dim=-1).sum().item()
            total_correct += correct
            total_examples += batch["labels"].shape[0]

        if log_images and _detached and "logits" in _detached and len(ttt_composites) < max_images:
            preds = _detached["logits"].argmax(dim=-1).detach().cpu().numpy()
            inputs_np = batch["inputs"].detach().cpu().numpy()
            labels_np = batch["labels"].detach().cpu().numpy()
            original_task_ids: Optional[List[str]] = batch.get("original_task_ids")
            for i in range(min(preds.shape[0], max_images_per_step)):
                if len(ttt_composites) >= max_images:
                    break
                input_grid = pad_grid_to_size(
                    seq_to_grid(inputs_np[i], max_rows=grid_max_size, max_cols=grid_max_size, pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset),
                    grid_max_size, grid_max_size, fill=GRID_TRANSPARENT_ID
                )
                target_grid = pad_grid_to_size(
                    seq_to_grid(labels_np[i], max_rows=grid_max_size, max_cols=grid_max_size, pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset),
                    grid_max_size, grid_max_size, fill=GRID_TRANSPARENT_ID
                )
                pred_grid = pad_grid_to_size(
                    seq_to_grid(preds[i], max_rows=grid_max_size, max_cols=grid_max_size, pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset),
                    grid_max_size, grid_max_size, fill=GRID_TRANSPARENT_ID
                )
                ttt_composites.append(build_composite_image(input_grid, pred_grid, target_grid, transparent_value=GRID_TRANSPARENT_ID))
                if original_task_ids is not None and i < len(original_task_ids):
                    ttt_task_ids.append(original_task_ids[i])
                else:
                    ttt_task_ids.append(None)

        if model_clone is not None:
            del model_clone
        if ttt_optimizer is not None:
            del ttt_optimizer

    # Aggregate metrics
    result: dict = {}
    n = max(total_batches, 1)
    for k, v in metrics_accum.items():
        if k.endswith("loss"):
            result[f"TTT/{k}"] = v / n
        elif k not in ("count", "count_all"):
            result[f"TTT/{k}"] = v / n

    if total_examples > 0:
        result["TTT/exact_accuracy"] = total_correct / total_examples
    result["TTT/lr"] = float(ttt_lr)
    result["TTT/num_tasks"] = total_batches
    result["TTT/batches"] = total_batches

    if log_images and ttt_composites:
        try:
            images = []
            for idx, composite in enumerate(ttt_composites):
                task_id_str = ttt_task_ids[idx] if idx < len(ttt_task_ids) else None
                caption = (
                    f"task={task_id_str}; step {step}; input; target; output"
                    if task_id_str
                    else f"step {step}; input; target; output"
                )
                images.append(wandb.Image(composite, caption=caption))
            if images:
                wandb.log({image_namespace: images}, step=step)
        except Exception as e:
            logger.warning("Could not log TTT images: %s", e)

    return result


def run(hydra_config: DictConfig):
    """Main training logic (called by launch() or pretrain.py).

    Supports single-GPU (``python3 train.py``) and multi-GPU DDP
    (``torchrun --nproc-per-node N train.py``).
    """
    # Parse config
    config = PretrainConfig(**hydra_config)

    # DDP setup (no-op when running single-GPU)
    ddp = setup_ddp()
    
    # Force all training logs to stderr so they appear in .err (Hydra may have sent them to stdout)
    root = logging.getLogger()
    root.handlers.clear()
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(stderr_handler)

    # Non-main ranks: suppress verbose logging to keep .err readable
    if not ddp.is_main:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Setup naming: single output folder per script (no project/run subdirs)
    if config.project_name is None:
        config.project_name = "arm-arc-agi"
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('@')[-1]}_{coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = config.output_dir
    
    logger.info(f"Project: {config.project_name}, Run: {config.run_name}")
    logger.info(f"Output dir: {config.checkpoint_path}")
    if ddp.is_ddp:
        logger.info(f"DDP: world_size={ddp.world_size}, per-GPU batch={config.global_batch_size // ddp.world_size}")

    # #region agent log
    if ddp.is_main:
        run_id = config.run_name or "unknown-run"
        primary_summaries = [_inspect_dataset_path(p) for p in config.data_paths]
        _debug_emit(
            run_id=run_id,
            hypothesis_id="H1",
            location="run:dataset_primary_summary",
            message="primary_dataset_summary",
            data={"data_paths": config.data_paths, "summaries": primary_summaries},
        )
        if config.reference_data_paths:
            ref_summaries = [_inspect_dataset_path(p) for p in config.reference_data_paths]
            _debug_emit(
                run_id=run_id,
                hypothesis_id="H2",
                location="run:dataset_reference_summary",
                message="reference_dataset_summary",
                data={"reference_data_paths": config.reference_data_paths, "summaries": ref_summaries},
            )
            _debug_emit(
                run_id=run_id,
                hypothesis_id="H3",
                location="run:dataset_parity_compare",
                message="primary_vs_reference",
                data={
                    "primary_layouts": [s.get("layout") for s in primary_summaries],
                    "reference_layouts": [s.get("layout") for s in ref_summaries],
                    "primary_reuse": [s.get("puzzle_id_reuse_ratio") for s in primary_summaries],
                    "reference_reuse": [s.get("puzzle_id_reuse_ratio") for s in ref_summaries],
                    "primary_keys": [s.get("sample_keys", []) for s in primary_summaries],
                    "reference_keys": [s.get("sample_keys", []) for s in ref_summaries],
                },
            )
    # #endregion
    
    # Set seed (offset by rank for data diversity across GPUs)
    torch.manual_seed(config.seed + ddp.rank)
    logger.info(f"Random seed: {config.seed} (rank offset {ddp.rank})")

    device = torch.device(f"cuda:{ddp.local_rank}" if ddp.is_ddp else "cuda")
    
    # Create train dataloader (one epoch per iteration; we recreate each epoch)
    logger.info("Creating train dataloader (1 epoch per pass)...")
    train_loader, train_metadata = create_dataloader(
        config, "train", ddp=ddp, test_set_mode=False, epochs_per_iter=1
    )
    
    # Resolve eval split: use "test" (unseen tasks).  Only rank 0 needs eval loader.
    # Keep the loader so we can reuse it for sampling a tracked eval example (avoids creating test dataloader twice).
    eval_split = None
    initial_eval_loader: Optional[Any] = None
    if ddp.is_main:
        try:
            initial_eval_loader, _meta = create_dataloader(
                config, "test", test_set_mode=True, epochs_per_iter=1
            )
            eval_split = "test"
            logger.info("Evaluation will use split 'test' (unseen tasks)")
        except FileNotFoundError:
            logger.warning("No test split available (no test/ with meta.yaml)")

    # Deprecated: tracked eval image logging under Eval/* during training steps.
    # Keep disabled to avoid mixing periodic Train-step logs with eval/TTT namespaces.
    tracked_eval_batch: Optional[dict] = None
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, train_metadata, ddp=ddp)

    resume_ckpt_dir: Optional[str] = None
    if config.resume:
        resume_ckpt_dir = _resume_latest_epoch_dir(config.resume)
        if resume_ckpt_dir:
            logger.info(f"Resuming from {resume_ckpt_dir}")
            load_checkpoint(model, resume_ckpt_dir)
        else:
            logger.warning(f"Resume path not found or no checkpoints: {config.resume}")
    elif config.load_checkpoint:
        load_checkpoint(model, config.load_checkpoint)

    # Wrap with DDP after checkpoint loading (weights must be identical across ranks)
    raw_model = model  # keep unwrapped ref for checkpoint saving
    if ddp.is_ddp:
        model = DDP(model, device_ids=[ddp.local_rank])
        logger.info("Model wrapped with DistributedDataParallel")
    
    # Create optimizer (operates on DDP wrapper so grads are synced)
    optimizer = create_optimizer(config, model)
    
    # Steps per epoch:
    # - Sharded JSONL layout: estimate from total_puzzles and per-GPU batch.
    # - Legacy arc-2-aug layout: count batches from one dataloader pass because
    #   grouped legacy sampling does not iterate over all raw examples each epoch.
    if train_metadata.pad_id is not None:
        logger.info("Computing steps per epoch from legacy iterator...")
        steps_per_epoch = max(1, sum(1 for _ in train_loader))
    else:
        per_rank_puzzles = train_metadata.total_puzzles // max(ddp.world_size, 1)
        per_gpu_batch = config.global_batch_size // max(ddp.world_size, 1)
        steps_per_epoch = max(1, (per_rank_puzzles + per_gpu_batch - 1) // per_gpu_batch)
    total_steps = config.epochs * steps_per_epoch
    logger.info(f"Steps per epoch: {steps_per_epoch:,}, Total steps: {total_steps:,} ({config.epochs} epochs)")
    
    # Initialize train state
    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        step=0,
        total_steps=total_steps,
    )

    start_epoch = 0
    wandb_run_id: Optional[str] = None
    if config.resume and resume_ckpt_dir:
        loaded_epoch, loaded_step, wandb_run_id = load_train_state(optimizer, resume_ckpt_dir)
        if loaded_epoch == 0 and loaded_step == 0:
            base = os.path.basename(resume_ckpt_dir.rstrip(os.sep))
            if base.startswith("epoch_"):
                try:
                    loaded_epoch = int(base[6:])
                    loaded_step = loaded_epoch * steps_per_epoch
                    train_state.step = loaded_step
                    logger.info(f"No train_state.pt found; inferred epoch {loaded_epoch}, step ~{loaded_step}")
                except ValueError:
                    pass
            wandb_run_id = None
        else:
            train_state.step = loaded_step
        start_epoch = loaded_epoch
        logger.info(f"Resuming from epoch {loaded_epoch}, step {train_state.step}; will run epochs {start_epoch + 1}-{config.epochs}")
    
    # Initialize WandB -- rank 0 only; other ranks get disabled mode
    if ddp.is_main:
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            wandb_api_key = wandb_api_key.strip("'\"")
            logger.info("Authenticating with WandB")
            wandb.login(key=wandb_api_key)
        
        wandb_init_kw: dict = {
            "project": config.project_name,
            "name": config.run_name,
            "config": config.model_dump(),
            "settings": wandb.Settings(_disable_stats=False),
            "mode": "disabled" if config.wandb_disabled else "online",
        }
        if wandb_run_id:
            wandb_init_kw["id"] = wandb_run_id
            wandb_init_kw["resume"] = "must"
            logger.info(f"WandB resuming existing run id={wandb_run_id}")
        wandb.init(**wandb_init_kw)
        
        num_params = sum(p.numel() for p in model.parameters())
        wandb.log({"num_params": num_params}, step=0)
        logger.info(f"WandB initialized. Model has {num_params:,} parameters")
        
        save_config(config)
    else:
        wandb.init(mode="disabled")
    
    # Epoch loop: one epoch -> validate -> log -> checkpoint -> next epoch
    logger.info(f"Starting training: epochs {start_epoch + 1}-{config.epochs} of {config.epochs} (validate and checkpoint after each epoch)")
    
    for epoch in range(start_epoch, config.epochs):
        if ddp.is_main:
            logger.info("=" * 60)
            logger.info(f"EPOCH {epoch + 1}/{config.epochs}")
            logger.info("=" * 60)
        
        # Fresh train dataloader for this epoch (IterableDataset is consumed after one pass)
        train_loader, _ = create_dataloader(
            config, "train", ddp=ddp, test_set_mode=False, epochs_per_iter=1
        )
        train_state.model.train()
        if ddp.is_main:
            logger.info(f"Training epoch {epoch + 1}: 0/{steps_per_epoch} steps")
        
        epoch_steps = 0
        log_interval = min(50, max(1, steps_per_epoch // 25))
        for set_name, batch, batch_size in train_loader:
            # #region agent log
            if ddp.is_main and train_state.step == 0:
                _debug_emit(
                    run_id=config.run_name or "unknown-run",
                    hypothesis_id="H4",
                    location="run:first_train_batch",
                    message="first_batch_schema_and_ranges",
                    data={
                        "set_name": set_name,
                        "global_batch_from_loader": batch_size,
                        "keys": sorted(list(batch.keys())),
                        "inputs_shape": list(batch["inputs"].shape) if "inputs" in batch else None,
                        "labels_shape": list(batch["labels"].shape) if "labels" in batch else None,
                        "puzzle_ids_shape": list(batch["puzzle_identifiers"].shape) if "puzzle_identifiers" in batch else None,
                        "num_demos_present": "num_demos" in batch,
                        "transform_ids_present": "transform_ids" in batch,
                        "color_maps_present": "color_maps" in batch,
                        "task_identifiers_present": "task_identifiers" in batch,
                    },
                )
            # #endregion
            metrics = train_batch(config, train_state, batch, batch_size)
            if metrics:
                if ddp.is_main:
                    wandb.log(metrics, step=train_state.step)
                epoch_steps += 1

                if (
                    ddp.is_main
                    and config.debug
                    and config.log_train_images
                    and config.train_images_log_interval > 0
                    and (
                        train_state.step == 1
                        or train_state.step % config.train_images_log_interval == 0
                    )
                ):
                    log_train_images_batch(
                        train_state.model, batch, device,
                        step=train_state.step, max_images=config.train_images_max,
                        max_images_per_step=config.max_images_per_step,
                        grid_max_size=config.grid_max_size, grid_pad_id=config.grid_pad_id,
                        grid_eos_id=config.grid_eos_id, grid_vocab_offset=config.grid_vocab_offset,
                    )
                if ddp.is_main and (epoch_steps % log_interval == 0 or epoch_steps == 1):
                    lm_loss = metrics.get("Train/lm_loss", "n/a")
                    q_halt_loss = metrics.get("Train/q_halt_loss", "n/a")
                    q_continue_loss = metrics.get("Train/q_continue_loss", "n/a")
                    act_steps = metrics.get("Train/act_steps", "n/a")
                    exact_acc = metrics.get("Train/exact_accuracy", metrics.get("exact_accuracy", 0.0))
                    acc = metrics.get("Train/accuracy", metrics.get("accuracy", 0.0))
                    logger.info(
                        f"Epoch {epoch + 1} step {epoch_steps}/{steps_per_epoch} "
                        f"(global {train_state.step}), "
                        f"Train/lm_loss={lm_loss}, Train/q_halt_loss={q_halt_loss}, "
                        f"Train/q_continue_loss={q_continue_loss}, Train/act_steps={act_steps}, "
                        f"Train/accuracy={acc:.4f}, Train/exact_accuracy={exact_acc:.4f}"
                    )
            if train_state.step >= total_steps:
                break

        if ddp.is_ddp:
            dist.barrier()
        
        if ddp.is_main:
            logger.info(f"Epoch {epoch + 1} finished: {epoch_steps} steps")
            wandb.log({"Train/epoch": epoch + 1, "Train/epoch_steps": epoch_steps}, step=train_state.step)
        
        # Evaluation every eval_epoch_interval epochs (or on the last epoch) -- rank 0 only
        is_final_epoch = (epoch == config.epochs - 1)
        should_eval = (
            ddp.is_main
            and eval_split is not None
            and epoch >= config.min_eval_interval
            and ((epoch + 1) % config.eval_epoch_interval == 0 or is_final_epoch)
        )
        if should_eval:
            logger.info("Evaluation on unseen test tasks...")
            # Use unwrapped model for eval (DDP wrapper adds overhead and is not needed)
            eval_model = raw_model if ddp.is_ddp else model

            if config.ttt_enabled:
                logger.info(f"TTT enabled: mode={config.ttt_mode}, steps={config.ttt_steps}, lr={config.ttt_lr}")
                ttt_eval_loader, _ = create_dataloader(
                    config, eval_split, test_set_mode=True, epochs_per_iter=1
                )
                ttt_image_namespace = "TTT/final_images" if is_final_epoch else "TTT/val_images"
                ttt_max_images = config.eval_final_max_images if is_final_epoch else config.eval_val_max_images
                ttt_metrics = evaluate_with_ttt(
                    eval_model,
                    ttt_eval_loader,
                    ttt_steps=config.ttt_steps,
                    ttt_lr=config.ttt_lr,
                    ttt_mode=config.ttt_mode,
                    debug=config.debug,
                    max_batches=config.eval_max_batches,
                    epoch=epoch + 1,
                    step=train_state.step,
                    grid_max_size=config.grid_max_size, grid_pad_id=config.grid_pad_id,
                    grid_eos_id=config.grid_eos_id, grid_vocab_offset=config.grid_vocab_offset,
                    ttt_augment=config.ttt_augment,
                    log_images=config.log_eval_images,
                    image_namespace=ttt_image_namespace,
                    max_images=ttt_max_images,
                    max_images_per_step=config.max_images_per_step,
                )
                ttt_metrics["TTT/epoch"] = epoch + 1
                logger.info(f"Epoch {epoch + 1} TTT eval: {ttt_metrics}")
                wandb.log(ttt_metrics, step=train_state.step)
            else:
                eval_loader, _ = create_dataloader(
                    config, eval_split, test_set_mode=True, epochs_per_iter=1
                )
                image_namespace = "Eval/final_images" if is_final_epoch else "Eval/val_images"
                max_images = config.eval_final_max_images if is_final_epoch else config.eval_val_max_images
                eval_metrics = evaluate(
                    eval_model,
                    eval_loader,
                    debug=config.debug,
                    log_images=config.log_eval_images,
                    image_namespace=image_namespace,
                    max_images=max_images,
                    max_images_per_step=config.max_images_per_step,
                    max_batches=config.eval_max_batches,
                    epoch=epoch + 1,
                    step=train_state.step,
                    grid_max_size=config.grid_max_size, grid_pad_id=config.grid_pad_id,
                    grid_eos_id=config.grid_eos_id, grid_vocab_offset=config.grid_vocab_offset,
                )
                eval_metrics["Eval/epoch"] = epoch + 1
                logger.info(f"Epoch {epoch + 1} evaluation: {eval_metrics}")
                wandb.log(eval_metrics, step=train_state.step)

        if ddp.is_ddp:
            dist.barrier()
        
        # Checkpoint: rank 0 only
        should_checkpoint = config.checkpoint_every_eval or (epoch == config.epochs - 1)
        if ddp.is_main and should_checkpoint:
            logger.info("Saving checkpoint...")
            # Save unwrapped model (without DDP wrapper)
            save_train_state = TrainState(
                model=raw_model if ddp.is_ddp else model,
                optimizer=optimizer,
                step=train_state.step,
                total_steps=total_steps,
            )
            save_checkpoint(config, save_train_state, epoch=epoch + 1)

        if ddp.is_ddp:
            dist.barrier()
        
        if train_state.step >= total_steps:
            break
    
    if ddp.is_main:
        wandb.finish()
    if ddp.is_ddp:
        dist.destroy_process_group()
    logger.info("Training completed successfully")


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    """Hydra entry point -- resolves config then delegates to run()."""
    run(hydra_config)


if __name__ == "__main__":
    launch()
