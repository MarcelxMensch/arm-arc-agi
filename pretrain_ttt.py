"""ARM-aware training entrypoint (fork of pretrain.py, 2026-04-14).

This file is a near-verbatim copy of ``pretrain.py`` with three additions:

1. ``PretrainConfig`` exposes ``arm_episode_mode`` / ``k_demos`` so the hydra
   config can flip ARM's episode iteration on without touching the baseline
   trainer.
2. ``create_dataloader`` forwards those flags into ``PuzzleDatasetConfig`` so
   the new collator in ``utils/data/arm_collator.py`` produces
   ``demo_inputs`` / ``demo_outputs`` batches.
3. ``load_checkpoint`` guards the ``puzzle_emb.weights.shape`` access so ARM
   checkpoints — whose ``puzzle_emb`` property is ``None`` — load cleanly.

The optimizer-group logic already has a ``puzzle_emb_ndim == 0`` branch that
drops the sparse optimizer, which matches ``arm.yaml``'s ``puzzle_emb_ndim: 0``
— so no change is needed there. The Phase 6 ``evaluate_ttt.py`` loop (per-task
LoRA TTT) lives in a separate file; this trainer is only the ARM base-training
entrypoint.

Baseline ``pretrain.py`` is left untouched for existing TRM / trm_abstraction
runs.
"""
from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

# ── Performance: enable TF32 for matmuls and cuDNN auto-tuning ──
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2 as AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from utils.models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from utils.models.ema import EMAHelper
from utils.logging import TrainLogger


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = "arm-arc-agi"
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

    # Training process
    gradient_accumulation_steps: int = 1
    log_interval: int = 100
    debug: bool = False
    skip_eval: bool = False
    checkpoint_interval: int = 0  # save checkpoint every N steps (0 = only at eval)
    autoresearch_max_steps: Optional[int] = None  # if set, stop training after this many steps

    # ARM episode mode — when enabled, the dataloader yields (demo pairs, target)
    # episodes rather than individual puzzle examples, and the model receives
    # batch["demo_inputs"] / batch["demo_outputs"]. See utils/data/arm_collator.py
    # for details.
    arm_episode_mode: bool = False
    k_demos: int = 2
    encoder_lr_mult: float = 1.0

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        arm_episode_mode=config.arm_episode_mode,
        k_demos=config.k_demos,
        **kwargs
    ), split=split)
    num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "1"))
    loader_kw = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kw["prefetch_factor"] = min(8, int(os.environ.get("DATALOADER_PREFETCH_FACTOR", "8")))
        loader_kw["persistent_workers"] = True
    dataloader = DataLoader(dataset, **loader_kw)
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // (world_size * config.gradient_accumulation_steps),
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        num_task_identifiers=getattr(train_metadata, 'num_task_identifiers', None),
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

        # Apply Triton kernel optimizations if enabled
        if "ENABLE_TRITON_KERNELS" in os.environ:
            from utils.models.layers_optimized import optimize_model
            model = optimize_model(model)
            print("[TRITON] Applied optimized kernels: RMSNorm, SwiGLU, RoPE, CastedLinear")

        if "DISABLE_COMPILE" not in os.environ:
            compile_mode = os.environ.get("COMPILE_MODE", "default")
            model = torch.compile(model, mode=compile_mode)  # type: ignore
            print(f"[COMPILE] torch.compile mode={compile_mode}")

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        if config.encoder_lr_mult != 1.0:
            encoder_params = [p for n, p in model.named_parameters() if 'demo_encoder' in n]
            other_params = [p for n, p in model.named_parameters() if 'demo_encoder' not in n]
            optimizers = [
                AdamATan2(
                    [{'params': other_params}, {'params': encoder_params, 'lr': config.lr * config.encoder_lr_mult}],
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
            ]
        else:
            optimizers = [
                AdamATan2(
                    model.parameters(),
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
            ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        sparse_emb_params = [{"params": list(emb.buffers())} for emb in model.model.sparse_embeddings]  # type: ignore
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                sparse_emb_params,
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        sparse_emb_params = [{"params": list(emb.buffers())} for emb in model.model.sparse_embeddings]  # type: ignore
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                sparse_emb_params,
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=config.lr,  # Base lr; scheduler sets param_group['lr'] each step
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState, ema_helper=None):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))

    # Save EMA weights separately so eval can load them without deepcopy
    if ema_helper is not None:
        torch.save(ema_helper.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}_ema"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Check if model uses composite softprompt (task_emb instead of puzzle_emb)
        use_composite = hasattr(model.model.inner, 'task_emb')
        # ARM has no sparse puzzle table at all — `puzzle_emb` is a property
        # that returns None. Guard the shape-patch block below so the loader
        # does not AttributeError on `None.weights`.
        has_puzzle_table = getattr(model.model, 'puzzle_emb', None) is not None

        if not use_composite and has_puzzle_table:
            # Resize and reset puzzle emb if needed
            puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
            expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
            if puzzle_emb_name in state_dict:
                puzzle_emb = state_dict[puzzle_emb_name]
                if puzzle_emb.shape != expected_shape:
                    print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                    # Re-initialize using mean
                    state_dict[puzzle_emb_name] = (
                        torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                    )
            model.load_state_dict(state_dict, assign=True)
        elif not use_composite and not has_puzzle_table:
            # ARM (no sparse table). Load strict=False — the checkpoint may be
            # from a trm_abstraction run with extra puzzle_emb weights that
            # don't map onto ARM's parameter set.
            missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
            if missing:
                print(f"Checkpoint missing keys (expected for ARM fresh init): {missing}")
            if unexpected:
                print(f"Checkpoint unexpected keys (old puzzle_emb / legacy): {unexpected}")
        else:
            # Composite mode: load with strict=False to allow mismatched embedding keys
            missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
            if missing:
                print(f"Checkpoint missing keys (expected for composite softprompt): {missing}")
            if unexpected:
                print(f"Checkpoint unexpected keys (old puzzle_emb): {unexpected}")


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "utils.evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_micro_batch(config: PretrainConfig, train_state: TrainState, batch: Any, effective_global_batch_size: int, rank: int, world_size: int):
    """Run forward + backward for a single micro-batch, accumulating gradients."""
    # To device (non-blocking since pin_memory=True in dataloader)
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward - request extra outputs for logging
    return_keys = ["preds", "q_halt_logits"]
    train_state.carry, loss, metrics, extra_outputs, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=return_keys)

    # Scale loss by effective global batch size (accounts for accumulation)
    ((1 / effective_global_batch_size) * loss).backward()

    return metrics, extra_outputs, batch


def train_step(config: PretrainConfig, train_state: TrainState, micro_batches: list, effective_global_batch_size: int, rank: int, world_size: int, logger: Optional[TrainLogger] = None):
    """Run N micro-batches with gradient accumulation, then optimizer step."""
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return None, None, None, None

    # Accumulate gradients over micro-batches
    all_metrics = []
    last_extra_outputs = None
    last_batch = None
    for batch_data in micro_batches:
        _, batch, _ = batch_data
        metrics, extra_outputs, batch_cuda = train_micro_batch(
            config, train_state, batch, effective_global_batch_size, rank, world_size
        )
        all_metrics.append(metrics)
        last_extra_outputs = extra_outputs
        last_batch = batch_cuda

    # Allreduce (after all micro-batches accumulated)
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Compute gradient metrics after allreduce, before optimizer step (rank 0 only)
    grad_metrics = {}
    if rank == 0 and logger is not None and train_state.step % config.log_interval == 0:
        grad_metrics = logger.compute_grad_metrics()

    # Apply optimizer
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step

        optim.step()
        optim.zero_grad()

    # Aggregate and reduce metrics from all micro-batches
    if all_metrics and len(all_metrics[0]):
        # Sum metrics across micro-batches
        metric_keys = list(sorted(all_metrics[0].keys()))
        combined = torch.stack([
            sum(m[k] for m in all_metrics) for k in metric_keys
        ])

        assert not any(combined[i].requires_grad for i in range(len(metric_keys)))

        if world_size > 1:
            dist.reduce(combined, dst=0)

        if rank == 0:
            metric_values = combined.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            # Extract and process per-ACT-step loss metrics before standard postprocessing
            per_step_metrics = {}
            step_count_keys = sorted([k for k in reduced_metrics if k.startswith("count_step_")])
            for count_key in step_count_keys:
                s = count_key.split("_")[-1]
                loss_key = f"loss_step_{s}"
                c = max(reduced_metrics.pop(count_key), 1)
                loss_val = reduced_metrics.pop(loss_key)
                per_step_metrics[f"train/loss_at_step_{s}"] = loss_val / c

            # Extract precision/recall raw counts before generic postprocessing
            q_attempted = max(reduced_metrics.pop("q_attempted"), 1)
            q_attempted_correct = reduced_metrics.pop("q_attempted_correct")
            q_solvable = max(reduced_metrics.pop("q_solvable"), 1)

            # Standard postprocessing
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (effective_global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            # Precision/recall/F1 from Q-value decisions (debug only)
            if logger is not None:
                precision = q_attempted_correct / q_attempted
                recall = q_attempted_correct / q_solvable
                f1_denom = precision + recall
                reduced_metrics["train/q_precision"] = precision
                reduced_metrics["train/q_recall"] = recall
                reduced_metrics["train/q_f1"] = 2 * precision * recall / f1_denom if f1_denom > 0 else 0.0

            reduced_metrics["train/lr"] = lr_this_step
            reduced_metrics["train/samples_seen"] = train_state.step * effective_global_batch_size
            reduced_metrics.update(grad_metrics)
            return reduced_metrics, last_extra_outputs, per_step_metrics, last_batch

    return None, None, None, None

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    logger: Optional[TrainLogger] = None,
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)
        # Ensure preds and q_halt_logits are available for eval logging
        return_keys.add("preds")
        return_keys.add("q_halt_logits")

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        # Collect samples for eval logging (grid images, confusion matrix)
        eval_log_batches = []
        eval_log_preds = []
        # Collect Q-value data for all batches (precision/recall analysis)
        eval_q_data = []

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            # Save a few batches for eval logging (CPU copies)
            if rank == 0 and logger is not None and len(eval_log_batches) < 3:
                eval_log_batches.append({k: v.cpu() for k, v in batch.items()})
                eval_log_preds.append({k: v.cpu() for k, v in preds.items()})

            # Collect Q-value data for precision/recall analysis (all batches)
            if rank == 0 and logger is not None and "q_halt_logits" in preds:
                eval_q_data.append({
                    "q_halt_logits": preds["q_halt_logits"].cpu(),
                    "preds": preds["preds"].cpu(),
                    "labels": batch["labels"].cpu(),
                    "puzzle_identifiers": batch["puzzle_identifiers"].cpu(),
                })

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    # Remove per-ACT-step loss metrics (training-specific, not useful during eval)
                    for k in list(m.keys()):
                        if k.startswith("count_step_") or k.startswith("loss_step_"):
                            m.pop(k)
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

        # Eval logging (grid images, confusion matrix, embedding PCA, Q-value analysis)
        if rank == 0 and logger is not None and eval_log_batches:
            logger.log_eval(train_state.step, eval_log_batches, eval_log_preds, eval_q_data)

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


def _save_embeddings(config: PretrainConfig, train_state: TrainState):
    """Save raw puzzle embeddings for PCA/clustering analysis in experiment reports."""
    if config.checkpoint_path is None:
        return
    try:
        import numpy as np
        from utils.logging import _unwrap_to_inner, load_identifier_maps, get_arc_puzzle_id

        inner = _unwrap_to_inner(train_state.model)
        if inner is None:
            return

        if hasattr(inner, 'task_emb'):
            weights = inner.task_emb.weights.detach().cpu().float().numpy()
        elif hasattr(inner, 'puzzle_emb'):
            weights = inner.puzzle_emb.weights.detach().cpu().float().numpy()
        else:
            return
        # Resolve puzzle IDs
        id_maps = load_identifier_maps(config.data_paths)
        puzzle_ids = []
        for idx in range(weights.shape[0]):
            puzzle_ids.append(get_arc_puzzle_id(id_maps, idx))

        emb_dir = os.path.join(config.checkpoint_path, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        np.savez(
            os.path.join(emb_dir, f"step_{train_state.step}.npz"),
            embeddings=weights,
            puzzle_ids=np.array(puzzle_ids, dtype=object),
            step=train_state.step,
        )

        # Log to W&B as Table for the Embeddings panel
        if wandb.run is not None:
            # Deduplicate augmented variants
            seen = {}
            for idx in range(1, weights.shape[0]):
                pid = puzzle_ids[idx]
                if pid != "<blank>" and pid not in seen:
                    seen[pid] = weights[idx]
            if seen:
                columns = ["puzzle_id", "step"] + [f"d{i}" for i in range(weights.shape[1])]
                rows = []
                for pid, emb in seen.items():
                    rows.append([pid, train_state.step] + emb.tolist())
                table = wandb.Table(columns=columns, data=rows)
                wandb.log({"embeddings/puzzle_embeddings": table}, step=train_state.step)
    except Exception as e:
        print(f"Embedding save failed: {e}")


def _log_external_eval_results(config: PretrainConfig, train_state: TrainState, logged_results: set):
    """Pick up eval results written by standalone eval script and log to W&B."""
    import json as json_mod
    eval_dir = os.path.join(config.checkpoint_path, "eval_results")
    if not os.path.isdir(eval_dir):
        return
    for fname in sorted(os.listdir(eval_dir)):
        if not fname.endswith(".json") or fname in logged_results:
            continue
        fpath = os.path.join(eval_dir, fname)
        try:
            with open(fpath) as f:
                result = json_mod.load(f)
            step = result.pop("_step", train_state.step)
            if step > train_state.step:
                continue  # skip eval results from a previous run that are ahead of current training
            wandb.log(result, step=step)
            logged_results.add(fname)
            print(f"Logged external eval: {fname} at step {step}")
        except Exception as e:
            print(f"Failed to load eval result {fname}: {e}")


def _print_autoresearch_summary(metrics: dict, step: int):
    """Print a machine-readable summary line for autoresearch experiment extraction."""
    lm_loss = metrics.get("train/lm_loss", "N/A")
    accuracy = metrics.get("train/accuracy", "N/A")
    exact_accuracy = metrics.get("train/exact_accuracy", "N/A")
    pics_per_sec = metrics.get("train/pics_per_sec", "N/A")
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"\nRESULT: lm_loss={lm_loss} accuracy={accuracy} exact_accuracy={exact_accuracy} pics_per_sec={pics_per_sec} steps={step} peak_vram_mb={peak_vram_mb:.1f}")


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    # Enable TF32 for float32 matmuls (e.g. loss computation, grad norm)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    micro_batch_size = config.global_batch_size // config.gradient_accumulation_steps
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=micro_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=micro_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    train_logger = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump())  # type: ignore
        if config.debug:
            pass  # wandb.log() handles metric logging; wandb.watch() removed (fragile with None grads)
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
        # Persist W&B run ID so standalone eval can resume this run
        if config.checkpoint_path and wandb.run is not None:
            with open(os.path.join(config.checkpoint_path, "wandb_run_id.txt"), "w") as f:
                f.write(wandb.run.id)
        if config.debug:
            train_logger = TrainLogger(train_state.model, config, log_interval_medium=config.log_interval, log_interval_heavy=config.log_interval * 10)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Track which eval results we've already logged
    logged_eval_results = set()

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Pick up external eval results (from standalone eval script)
        if RANK == 0 and config.checkpoint_path is not None:
            _log_external_eval_results(config, train_state, logged_eval_results)

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        accum_steps = config.gradient_accumulation_steps
        micro_batch_buffer = []
        for batch_data in train_loader:
            micro_batch_buffer.append(batch_data)
            if len(micro_batch_buffer) < accum_steps:
                continue

            if RANK == 0 and train_logger is not None:
                train_logger.step_start()

            metrics, extra_outputs, per_step_metrics, last_batch_cuda = train_step(
                config, train_state, micro_batch_buffer, config.global_batch_size,
                rank=RANK, world_size=WORLD_SIZE, logger=train_logger
            )
            micro_batch_buffer = []

            if RANK == 0 and train_logger is not None:
                train_logger.mark_step_compute_done()

            if RANK == 0 and metrics is not None:
                should_log = train_state.step % config.log_interval == 0
                if train_logger is not None and should_log:
                    train_logger.log_train_step(
                        train_state.step, metrics, last_batch_cuda,
                        train_state.carry, extra_outputs, config.global_batch_size
                    )
                if should_log:
                    wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)

            # autoresearch early stop
            if config.autoresearch_max_steps is not None and train_state.step >= config.autoresearch_max_steps:
                if RANK == 0 and metrics is not None:
                    _print_autoresearch_summary(metrics, train_state.step)
                break

            # Periodic checkpoint (independent of eval)
            if config.checkpoint_interval > 0 and train_state.step % config.checkpoint_interval == 0:
                if RANK == 0:
                    print(f"PERIODIC CHECKPOINT at step {train_state.step}")
                    save_train_state(config, train_state, ema_helper=ema_helper)
                    _save_embeddings(config, train_state)

        # Break outer loop if autoresearch step limit reached
        if config.autoresearch_max_steps is not None and train_state.step >= config.autoresearch_max_steps:
            break

        ############ Evaluation (skip if decoupled)
        if not config.skip_eval and _iter_id >= config.min_eval_interval:
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config,
                train_state_eval,
                eval_loader,
                eval_metadata,
                evaluators,
                rank=RANK,
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP,
                logger=train_logger)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)

            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval, ema_helper=ema_helper)
                _save_embeddings(config, train_state_eval)

            if config.ema:
                del train_state_eval
        elif config.skip_eval:
            # Still save checkpoint at eval boundaries when eval is skipped
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                print(f"SAVE CHECKPOINT (eval skipped) at step {train_state.step}")
                save_train_state(config, train_state, ema_helper=ema_helper)
                _save_embeddings(config, train_state)

    # Export W&B history to experiment directory for experiment reports
    if RANK == 0 and wandb.run is not None and config.checkpoint_path is not None:
        try:
            from utils.wandb_export import export_run
            exp_dir = os.path.dirname(config.checkpoint_path)
            export_run(
                entity=wandb.run.entity,
                project=wandb.run.project,
                run_id=wandb.run.id,
                out_dir=exp_dir,
            )
        except Exception as e:
            print(f"W&B export failed: {e}")

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
