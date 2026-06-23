"""Evaluation module for ARC-style tasks."""

from typing import Dict, Any, List, Optional, Tuple
import logging

import torch
from torch.utils.data import DataLoader

from utils.dataset.common import (
    seq_to_grid,
    pad_grid_to_size,
    build_composite_image,
    GRID_TRANSPARENT_ID,
)

logger = logging.getLogger(__name__)


def evaluate(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device = torch.device("cuda"),
    debug: bool = False,
    log_images: bool = False,
    image_namespace: str = "Eval/val_images",
    max_images: int = 8,
    max_images_per_step: int = 3,
    max_batches: Optional[int] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    grid_max_size: int = 30,
    grid_pad_id: int = 0,
    grid_eos_id: int = 1,
    grid_vocab_offset: int = 2,
) -> Dict[str, float]:
    """
    Run evaluation on a dataset.
    
    Args:
        model: The model to evaluate (with loss head).
        eval_loader: DataLoader yielding (set_name, batch, batch_size) tuples.
        device: Device to run evaluation on.
        debug: Whether to include debug metrics.
        log_images: If True, log composite [input|pred|target] images to wandb.
        image_namespace: WandB key for images, e.g. "Eval/val_images" or "Eval/final_images".
        max_images: Max number of composite images to log (small for val, large for final).
        epoch: Optional epoch for image captions.
        step: Optional step for image captions.
        grid_max_size, grid_pad_id, grid_eos_id, grid_vocab_offset: Grid encoding (from config).
    
    Returns:
        Dictionary of evaluation metrics (all under Eval/*).
    """
    model.eval()
    
    metrics_accum = {}
    total_batches = 0
    eval_composites: List[Any] = []
    eval_task_ids: List[Optional[str]] = []  # external ARC-AGI task id (e.g. 7fe24cdd) for arcprize.org/play?task=
    max_rows = max_cols = grid_max_size
    
    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            if max_batches is not None and total_batches >= max_batches:
                break
            total_batches += 1

            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.device(device):
                carry = model.initial_carry(batch)
            
            max_log = min(max_images_per_step, max_images)
            # Ask for logits so we can compute preds here (model does not emit 'preds' directly)
            return_keys = ["logits"] if log_images and len(eval_composites) < max_log else []
            inference_steps = 0
            while True:
                carry, loss, metrics, detached, all_finish = model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                if all_finish:
                    break
            
            for k, v in metrics.items():
                if k not in metrics_accum:
                    metrics_accum[k] = 0.0
                metrics_accum[k] += v.item() if isinstance(v, torch.Tensor) else v
            
            if log_images and len(eval_composites) < max_log and detached and "logits" in detached:
                # Convert logits -> predicted tokens
                preds = torch.argmax(detached["logits"], dim=-1).cpu().numpy()
                inputs = batch["inputs"].cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                original_task_ids: Optional[List[str]] = batch.get("original_task_ids")
                for i in range(preds.shape[0]):
                    if len(eval_composites) >= max_log:
                        break
                    input_grid = pad_grid_to_size(
                        seq_to_grid(
                            inputs[i], max_rows=max_rows, max_cols=max_cols,
                            pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset,
                        ),
                        grid_max_size, grid_max_size,
                        fill=GRID_TRANSPARENT_ID,
                    )
                    target_grid = pad_grid_to_size(
                        seq_to_grid(
                            labels[i], max_rows=max_rows, max_cols=max_cols,
                            pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset,
                        ),
                        grid_max_size, grid_max_size,
                        fill=GRID_TRANSPARENT_ID,
                    )
                    pred_grid = pad_grid_to_size(
                        seq_to_grid(
                            preds[i], max_rows=max_rows, max_cols=max_cols,
                            pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset,
                        ),
                        grid_max_size, grid_max_size,
                        fill=GRID_TRANSPARENT_ID,
                    )
                    composite = build_composite_image(
                        input_grid, pred_grid, target_grid,
                        transparent_value=GRID_TRANSPARENT_ID,
                    )
                    eval_composites.append(composite)
                    if original_task_ids is not None and i < len(original_task_ids):
                        eval_task_ids.append(original_task_ids[i])
                    else:
                        eval_task_ids.append(None)
            
            if debug and total_batches % 10 == 0:
                logger.debug(f"Eval batch {total_batches}: steps={inference_steps}")
    
    if log_images and eval_composites:
        try:
            import wandb
            images = []
            for idx, composite in enumerate(eval_composites):
                task_id_str: Optional[str] = None
                if idx < len(eval_task_ids):
                    task_id_str = eval_task_ids[idx]
                if task_id_str:
                    caption = f"task={task_id_str}; step {step}; input; target; output"
                else:
                    caption = f"step {step}; input; target; output"
                images.append(wandb.Image(composite, caption=caption))
            if images:
                wandb.log({image_namespace: images}, step=step)
        except ImportError:
            pass
    
    count = metrics_accum.pop("count", 1)
    if count == 0:
        count = 1
    count_all = metrics_accum.pop("count_all", count)
    if count_all == 0:
        count_all = count
    
    results = {}
    for k, v in metrics_accum.items():
        if k.endswith("loss"):
            results[f"Eval/{k}"] = v / max(total_batches, 1)
        else:
            norm_count = count_all if k in ("accuracy", "exact_accuracy") else count
            results[f"Eval/{k}"] = v / norm_count
    
    results["Eval/count"] = count
    results["Eval/batches"] = total_batches
    
    logger.info(f"Evaluation complete: {total_batches} batches, {count} samples")
    return results


def log_train_images_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    step: int,
    max_images: int = 4,
    max_images_per_step: int = 3,
    grid_max_size: int = 30,
    grid_pad_id: int = 0,
    grid_eos_id: int = 1,
    grid_vocab_offset: int = 2,
) -> None:
    """
    Log a few composite [input|pred|target] images from one training batch under Train/images.
    For qualitative sanity checks only; call once per epoch or at fixed step intervals.
    Grid encoding args come from config.
    """
    model.eval()
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.device(device):
        carry = model.initial_carry(batch)
    with torch.inference_mode():
        while True:
            carry, _, _, detached, all_finish = model(
                carry=carry, batch=batch, return_keys=["logits"]
            )
            if all_finish:
                break
    if not detached or "logits" not in detached:
        return
    preds = torch.argmax(detached["logits"], dim=-1).cpu().numpy()
    inputs = batch["inputs"].cpu().numpy()
    labels = batch["labels"].cpu().numpy()
    original_task_ids: Optional[List[str]] = batch.get("original_task_ids")
    try:
        import wandb
        max_rows = max_cols = grid_max_size
        composites = []
        for i in range(min(max_images_per_step, max_images, preds.shape[0])):
            input_grid = pad_grid_to_size(
                seq_to_grid(
                    inputs[i], max_rows=max_rows, max_cols=max_cols,
                    pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset,
                ),
                grid_max_size, grid_max_size,
                fill=GRID_TRANSPARENT_ID,
            )
            target_grid = pad_grid_to_size(
                seq_to_grid(
                    labels[i], max_rows=max_rows, max_cols=max_cols,
                    pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset,
                ),
                grid_max_size, grid_max_size,
                fill=GRID_TRANSPARENT_ID,
            )
            pred_grid = pad_grid_to_size(
                seq_to_grid(
                    preds[i], max_rows=max_rows, max_cols=max_cols,
                    pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset,
                ),
                grid_max_size, grid_max_size,
                fill=GRID_TRANSPARENT_ID,
            )
            composite = build_composite_image(
                input_grid, pred_grid, target_grid,
                transparent_value=GRID_TRANSPARENT_ID,
            )
            if original_task_ids is not None and i < len(original_task_ids) and original_task_ids[i]:
                caption = f"task={original_task_ids[i]}; step {step}; input; target; output"
            else:
                caption = f"step {step}; input; target; output"
            composites.append(wandb.Image(composite, caption=caption))
        if composites:
            wandb.log({"Train/images": composites}, step=step)
    except ImportError:
        pass
    model.train()


def log_tracked_eval_example(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    step: int,
    grid_max_size: int = 30,
    grid_pad_id: int = 0,
    grid_eos_id: int = 1,
    grid_vocab_offset: int = 2,
) -> None:
    """
    Run model on a single example (batch with batch size 1) and log composite
    [input, target, pred] to wandb under Eval/images (same style as Train/images).
    Used to track how the model improves on one fixed eval example over training.
    """
    model.eval()
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.device(device):
        carry = model.initial_carry(batch)
    with torch.inference_mode():
        while True:
            carry, _, _, detached, all_finish = model(
                carry=carry, batch=batch, return_keys=["logits"]
            )
            if all_finish:
                break
    if not detached or "logits" not in detached:
        return
    preds = torch.argmax(detached["logits"], dim=-1).cpu().numpy()
    inputs = batch["inputs"].cpu().numpy()
    labels = batch["labels"].cpu().numpy()
    try:
        import wandb
        max_rows = max_cols = grid_max_size
        input_grid = pad_grid_to_size(
            seq_to_grid(
                inputs[0], max_rows=max_rows, max_cols=max_cols,
                pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset,
            ),
            grid_max_size, grid_max_size,
            fill=GRID_TRANSPARENT_ID,
        )
        target_grid = pad_grid_to_size(
            seq_to_grid(
                labels[0], max_rows=max_rows, max_cols=max_cols,
                pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset,
            ),
            grid_max_size, grid_max_size,
            fill=GRID_TRANSPARENT_ID,
        )
        pred_grid = pad_grid_to_size(
            seq_to_grid(
                preds[0], max_rows=max_rows, max_cols=max_cols,
                pad_id=grid_pad_id, eos_id=grid_eos_id, vocab_offset=grid_vocab_offset,
            ),
            grid_max_size, grid_max_size,
            fill=GRID_TRANSPARENT_ID,
        )
        composite = build_composite_image(
            input_grid, pred_grid, target_grid,
            transparent_value=GRID_TRANSPARENT_ID,
        )
        original_task_ids: Optional[List[str]] = batch.get("original_task_ids")
        task_id_str: Optional[str] = None
        if original_task_ids and len(original_task_ids) > 0:
            task_id_str = original_task_ids[0]
        if task_id_str:
            caption = f"task={task_id_str}; step {step}; input; target; output"
        else:
            caption = f"step {step}; input; target; output"
        wandb.log({"Eval/images": [wandb.Image(composite, caption=caption)]}, step=step)
    except ImportError:
        pass
    model.train()


def compute_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[float, float]:
    """
    Compute token-level and sequence-level accuracy.
    
    Args:
        preds: Predicted tokens (B, seq_len)
        labels: Ground truth tokens (B, seq_len)
        ignore_index: Label value to ignore.
    
    Returns:
        Tuple of (token_accuracy, sequence_accuracy).
    """
    mask = labels != ignore_index
    
    if not mask.any():
        return 0.0, 0.0
    
    correct = (preds == labels) & mask
    
    # Token-level accuracy
    token_acc = correct.sum().float() / mask.sum().float()
    
    # Sequence-level accuracy (all tokens correct)
    seq_correct = (correct.sum(dim=-1) == mask.sum(dim=-1))
    seq_acc = seq_correct.float().mean()
    
    return token_acc.item(), seq_acc.item()
