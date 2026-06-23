#!/usr/bin/env python3
"""Support-TTT LoRA evaluation on episode datasets.

This is a true per-episode adaptation harness for the support-TTT datasets:

1. Load a frozen support-TTT checkpoint.
2. Add fresh low-rank adapters to the frozen model.
3. For each episode with enough support examples, leave one support pair out
   for validation and adapt LoRA on the remaining known support pairs.
4. Select the best fold by validation loss.
5. Predict the query with the selected adapter state.

The query label is used only for reporting metrics, never for adaptation or
adapter selection.

Datasets such as ``trm-abstraction-support-ttt-arc1-aug-1000`` use
``ignore_label_id=0`` for padded canvas positions; those positions are mapped
to ``-100`` before CE and accuracy, matching ``PuzzleDataset`` / training.
Each row is one (base task, augmentation plan, query) episode. Use
``--sample-tasks`` / ``--episodes-per-task`` to sample a small set of base tasks
and multiple rows per task; summaries include ``task_level/*`` (mean of per-task
max metrics). Use ``--no-ttt`` for a zero-shot baseline without adaptation.

Use ``--query-act-steps`` to align query inference with meta-training
(``arch.meta.query_act_steps``; default from config, else ``halt_max_steps``).
Training ``exact_accuracy`` uses the same CE mask as this script: only
non-``ignore`` label positions count; padding can look arbitrary in W&B
RGB grids because ignored targets are plotted as black after ``clip(0,9)``.
``--dump-raw-tokens`` writes per-episode ``pred`` / ``labels`` / ``mismatches``
into the JSON for auditing.

Also reports **ARC-cropped** exact (same ``_crop`` + ``inverse_aug`` + ``grid_hash`` as
``utils/evaluators/arc.py``) and **pass@K** over episode groups
``(task_identifier, query_sources, query_indices)`` when multiple rows share a query.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from evaluate_checkpoint import extract_step, find_checkpoints
from pretrain import PretrainConfig, create_dataloader
from utils.dataset.build_arc_dataset import grid_hash, inverse_aug
from utils.functions import load_model_class
from utils.evaluators.arc import _crop
from utils.models.losses import IGNORE_LABEL_ID


class LowRankAdapter(nn.Module):
    """Small residual adapter with zero-init output projection."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / (dim**0.5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x.float())).to(x.dtype)


class DeltaReadLoRA(nn.Module):
    def __init__(self, base: nn.Module, dim: int, rank: int) -> None:
        super().__init__()
        self.base = base
        self.lora = LowRankAdapter(dim, rank)

    def forward(self, z_H: torch.Tensor, delta_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        # Forward extra kwargs (e.g. key_padding_mask) to the wrapped base block.
        return self.base(z_H, delta_tokens, **kwargs) + self.lora(z_H)


class BroadcastLoRA(nn.Module):
    def __init__(self, base: nn.Module, dim: int, rank: int) -> None:
        super().__init__()
        self.base = base
        self.lora = LowRankAdapter(dim, rank)

    def forward(self, z_L: torch.Tensor, z_H: torch.Tensor) -> torch.Tensor:
        y = self.base(z_L, z_H)
        return y + self.lora(y)


class HLevelLoRA(nn.Module):
    """Wrap H-level with LoRA on perceive input and each latent block input."""

    def __init__(self, base: nn.Module, dim: int, rank: int) -> None:
        super().__init__()
        self.base = base
        self.lora_perceive = LowRankAdapter(dim, rank)
        self.block_loras = nn.ModuleList([LowRankAdapter(dim, rank) for _ in base.layers])

    def forward(self, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        # Mirrors _HReasoningModule.forward with parallel low-rank residuals.
        perceive = self.base.cross_attn_perceive(query=z_H, kv=z_L) + self.lora_perceive(z_H)
        from utils.models.layers import rms_norm

        z_H = rms_norm(z_H + perceive, variance_epsilon=self.base.norm_eps)
        for layer, lora in zip(self.base.layers, self.block_loras):
            z_in = z_H
            z_H = layer(z_H) + lora(z_in)
        return z_H


def _unwrap_inner(loss_head: nn.Module):
    return loss_head.model.inner  # ACTLossHead -> support-TTT model -> inner


def install_lora(loss_head: nn.Module, rank: int, include_h: bool) -> List[nn.Parameter]:
    """Freeze checkpoint weights and insert trainable LoRA modules."""

    device = next(loss_head.parameters()).device
    for p in loss_head.parameters():
        p.requires_grad_(False)

    inner = _unwrap_inner(loss_head)
    inner.delta_read_block = DeltaReadLoRA(inner.delta_read_block, inner.config.H_hidden_size, rank).to(device)
    inner.broadcast = BroadcastLoRA(inner.broadcast, inner.config.hidden_size, rank).to(device)
    if include_h:
        inner.H_level = HLevelLoRA(inner.H_level, inner.config.H_hidden_size, rank).to(device)

    params: List[nn.Parameter] = []
    for name, p in loss_head.named_parameters():
        if ".lora" in name or "lora_" in name or "block_loras" in name:
            p.requires_grad_(True)
            params.append(p)
    if not params:
        raise RuntimeError("No LoRA parameters were installed")
    return params


def adapter_state_dict(loss_head: nn.Module, adapt_params: str) -> Dict[str, torch.Tensor]:
    if adapt_params == "full_weight":
        return {
            k: v.detach().clone()
            for k, v in loss_head.state_dict().items()
        }
    return {
        k: v.detach().clone()
        for k, v in loss_head.state_dict().items()
        if ".lora" in k or "lora_" in k or "block_loras" in k
    }


def load_adapter_state(loss_head: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    own = loss_head.state_dict()
    for k, v in state.items():
        own[k].copy_(v)


def latest_checkpoint(ckpt_dir: Path) -> Path:
    checkpoints = find_checkpoints(str(ckpt_dir))
    if not checkpoints:
        raise FileNotFoundError(f"No step_* checkpoints found in {ckpt_dir}")
    return Path(checkpoints[-1])


def load_support_config(
    checkpoint_path: Path,
    config_override: Optional[Path] = None,
    wandb_run: Optional[str] = None,
) -> PretrainConfig:
    """Load training config for rebuilding the model. Prefers local YAML, else W&B."""
    candidates: List[Path] = []
    if config_override is not None:
        candidates.append(config_override)
    candidates.append(checkpoint_path.parent / "all_config.yaml")

    raw: Optional[Dict] = None
    for config_path in candidates:
        if config_path.is_file():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            break

    if raw is None and wandb_run:
        import wandb

        run = wandb.Api(timeout=120).run(wandb_run)
        raw = dict(run.config)

    if raw is None:
        raise FileNotFoundError(
            f"No all_config.yaml next to {checkpoint_path} (tried {candidates[-1]}). "
            f"Pass --config PATH or --wandb-run entity/project/runid."
        )

    # W&B and Hydra may store extra keys; keep only PretrainConfig fields.
    keep = set(PretrainConfig.model_fields.keys())
    raw = {k: v for k, v in raw.items() if k in keep}

    if "evaluators" not in raw or not raw["evaluators"]:
        raw["evaluators"] = [{"name": "arc@ARC"}]
    raw["support_ttt_mode"] = True
    return PretrainConfig(**raw)


def strip_orig_mod(state_dict: dict) -> dict:
    return {
        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
        for k, v in state_dict.items()
    }


def meta_lora_rank_from_arch(config: PretrainConfig) -> int:
    extra = dict(getattr(config.arch, "__pydantic_extra__", {}) or {})
    meta = extra.get("maml") or extra.get("meta") or {}
    if isinstance(meta, dict):
        return int(meta.get("lora_rank", 8))
    return int(getattr(meta, "lora_rank", 8))


def query_act_steps_default_from_arch(config: PretrainConfig, halt_max_steps: int) -> int:
    """Match meta-training ``forward_loss_and_outputs(..., act_steps=query_act_steps)`` when unset."""
    extra = dict(getattr(config.arch, "__pydantic_extra__", {}) or {})
    meta = extra.get("maml") or extra.get("meta") or {}
    if isinstance(meta, dict) and meta.get("query_act_steps") is not None:
        return max(1, int(meta["query_act_steps"]))
    if not isinstance(meta, dict) and getattr(meta, "query_act_steps", None) is not None:
        return max(1, int(getattr(meta, "query_act_steps")))
    return max(1, int(halt_max_steps))


def state_dict_has_meta_lora_layout(sd: dict) -> bool:
    """True if checkpoint was saved from meta-training after ``install_lora(..., include_h=True)``."""
    return any("model.inner.H_level.base." in k for k in sd)


def create_support_model_for_eval(
    config: PretrainConfig,
    train_metadata,
    checkpoint_path: Path,
    use_ema: bool,
    device: torch.device,
) -> nn.Module:
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        num_task_identifiers=getattr(train_metadata, "num_task_identifiers", None),
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    with torch.device(device):
        model = model_cls(model_cfg)
        loss_head = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

    state_dict = strip_orig_mod(torch.load(checkpoint_path, map_location=device))
    if state_dict_has_meta_lora_layout(state_dict):
        install_lora(
            loss_head,
            rank=meta_lora_rank_from_arch(config),
            include_h=True,
        )
    loss_head.load_state_dict(state_dict, assign=True)
    if use_ema and checkpoint_path.with_name(checkpoint_path.name + "_ema").exists():
        ema = torch.load(checkpoint_path.with_name(checkpoint_path.name + "_ema"), map_location=device)
        ema = strip_orig_mod(ema)
        for name, param in loss_head.named_parameters():
            if name in ema:
                param.data.copy_(ema[name])
        print(f"Loaded EMA weights from {checkpoint_path}_ema")
    elif use_ema:
        print(f"Warning: --ema requested but no EMA file found for {checkpoint_path}; using base weights")
    return loss_head.to(device)


def load_episode_arrays(data_path: Path, split: str) -> Dict[str, np.ndarray]:
    split_dir = data_path / split
    keys = (
        "inputs",
        "labels",
        "support_inputs",
        "support_outputs",
        "support_mask",
        "puzzle_identifiers",
        "query_indices",
        "query_sources",
    )
    arrays = {}
    for key in keys:
        path = split_dir / f"all__{key}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays[key] = np.load(path, mmap_mode="r")
    task_path = split_dir / "all__task_identifiers.npy"
    if task_path.is_file():
        arrays["task_identifiers"] = np.load(task_path, mmap_mode="r")
    else:
        arrays["task_identifiers"] = None
    return arrays


def load_identifier_list(data_path: Path, split: str) -> List[str]:
    """Load puzzle string ids for ARC-hash exact.

    Support-TTT builds ``identifiers.json`` as ``{"train": ["<blank>", ...], "test": [...]}``;
    ``all__puzzle_identifiers.npy`` indexes into the list for that split only. Legacy datasets
    may use a single top-level JSON array instead.
    """
    path = data_path / "identifiers.json"
    if not path.is_file():
        return []
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        key = split if split in raw else None
        if key is None:
            for fallback in ("test", "train", "all"):
                if fallback in raw and isinstance(raw[fallback], list):
                    key = fallback
                    break
        if key is not None:
            lst = raw[key]
            return lst if isinstance(lst, list) else []
    return []


def arc_cropped_grid_match(
    pred_flat: np.ndarray,
    label_flat: np.ndarray,
    puzzle_identifier: int,
    identifier_list: Sequence[str],
) -> Optional[float]:
    """ARC-style exact: ``inverse_aug`` + ``_crop`` on pred/label grids, compare ``grid_hash``."""
    if puzzle_identifier < 0 or puzzle_identifier >= len(identifier_list):
        return None
    name = identifier_list[puzzle_identifier]
    if name == "<blank>":
        return None
    try:
        _o, inv = inverse_aug(name)
        pred_grid = inv(_crop(np.asarray(pred_flat, dtype=np.float64)))
        label_grid = inv(_crop(np.asarray(label_flat, dtype=np.float64)))
        pred_grid = np.asarray(pred_grid, dtype=np.uint8)
        label_grid = np.asarray(label_grid, dtype=np.uint8)
    except Exception:
        return None
    if pred_grid.shape != label_grid.shape:
        return 0.0
    return 1.0 if grid_hash(pred_grid) == grid_hash(label_grid) else 0.0


def parse_pass_at_ks(spec: str) -> Tuple[int, ...]:
    out = [int(x.strip()) for x in spec.split(",") if x.strip()]
    return tuple(sorted(set(k for k in out if k > 0)))


def pass_at_k_metrics(
    episode_records: List[Tuple[int, Dict[str, float]]],
    arrays: Dict[str, np.ndarray],
    pass_ks: Sequence[int],
    metric_key: str,
    name_prefix: str,
) -> Dict[str, float]:
    """For each (task, query_source, query_index) group, treat rows as up to K attempts (row order)."""

    if not pass_ks or not episode_records:
        return {}
    task_identifiers = arrays.get("task_identifiers")
    if task_identifiers is None:
        return {}
    qsrc = arrays["query_sources"]
    qidx = arrays["query_indices"]
    groups: Dict[Tuple[int, int, int], List[Tuple[int, Dict[str, float]]]] = defaultdict(list)
    for row, m in episode_records:
        key = (int(task_identifiers[int(row)]), int(qsrc[row]), int(qidx[row]))
        groups[key].append((row, m))

    results: Dict[str, float] = {}
    n_groups = len(groups)
    for k in pass_ks:
        hits = 0
        for _gk, lst in groups.items():
            lst_sorted = sorted(lst, key=lambda x: x[0])
            take = lst_sorted[: min(k, len(lst_sorted))]
            ok = False
            for _r, mm in take:
                v = mm.get(metric_key)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                if float(v) >= 0.999:
                    ok = True
                    break
            hits += int(ok)
        results[f"{name_prefix}/pass@{k}"] = hits / max(n_groups, 1)
    results[f"{name_prefix}/num_query_groups"] = float(n_groups)
    return results


def _apply_ignore_label_id(
    batch_labels: torch.Tensor, ignore_label_id: Optional[int]
) -> torch.Tensor:
    """Match ``PuzzleDataset._collate_*``: padded canvas cells must not be scored or CE targets."""
    if ignore_label_id is None:
        return batch_labels
    out = batch_labels.clone()
    out[out == ignore_label_id] = IGNORE_LABEL_ID
    return out


def make_batch(
    input_seq: np.ndarray,
    label_seq: np.ndarray,
    support_inputs: np.ndarray,
    support_outputs: np.ndarray,
    puzzle_identifier: int,
    max_support_examples: int,
    device: torch.device,
    ignore_label_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Build a 1-item batch. ``ignore_label_id`` applies only to ``labels`` (CE targets);

    ``support_*`` tensors must stay in 0.. for delta / token embeddings (same as
    ``PuzzleDataset._collate_support_ttt_batch``, which only remaps top-level ``labels``).
    """
    n_support = min(len(support_inputs), max_support_examples)
    seq_len = int(input_seq.shape[0])
    s_in = np.zeros((max_support_examples, seq_len), dtype=np.int32)
    s_out = np.zeros((max_support_examples, seq_len), dtype=np.int32)
    s_mask = np.zeros((max_support_examples,), dtype=np.bool_)
    if n_support:
        s_in[:n_support] = support_inputs[:n_support].astype(np.int32)
        s_out[:n_support] = support_outputs[:n_support].astype(np.int32)
        s_mask[:n_support] = True

    labels = torch.from_numpy(label_seq.astype(np.int32)[None]).to(device)
    labels = _apply_ignore_label_id(labels, ignore_label_id)

    return {
        "inputs": torch.from_numpy(input_seq.astype(np.int32)[None]).to(device),
        "labels": labels,
        "support_inputs": torch.from_numpy(s_in[None]).to(device),
        "support_outputs": torch.from_numpy(s_out[None]).to(device),
        "support_mask": torch.from_numpy(s_mask[None]).to(device),
        "puzzle_identifiers": torch.tensor([puzzle_identifier], dtype=torch.int32, device=device),
        "query_indices": torch.zeros((1,), dtype=torch.int32, device=device),
        "query_sources": torch.zeros((1,), dtype=torch.int32, device=device),
    }


def ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1).long(),
        ignore_index=IGNORE_LABEL_ID,
    )


def forward_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    act_steps: int,
    train_mode: bool,
) -> torch.Tensor:
    model.train(train_mode)
    with torch.device(batch["inputs"].device):
        carry = model.initial_carry(batch)
    total = None
    for _ in range(act_steps):
        carry, outputs = model(carry, batch)
        loss = ce_loss_from_logits(outputs["logits"], batch["labels"])
        total = loss if total is None else total + loss
        if not train_mode and bool(carry.halted.all()):
            break
    assert total is not None
    return total / act_steps


@torch.no_grad()
def predict_query(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    query_act_steps: int,
    *,
    return_raw: bool = False,
    return_logits: bool = False,
) -> Tuple[float, float, int, float, int, np.ndarray, Optional[Dict[str, object]], Optional[np.ndarray]]:
    """Query forward aligned with ``pretrain_support_ttt_supervised_reptile.forward_loss_and_outputs``:

    ``for step in range(1, query_act_steps + 1):`` forward; in ``eval()`` early exit when ``carry.halted.all()``.

    This is **not** the same as always running ``halt_max_steps`` unless you pass ``query_act_steps=halt_max_steps``.
    """
    model.eval()
    with torch.device(batch["inputs"].device):
        carry = model.initial_carry(batch)
    outputs = None
    steps = 0
    for steps in range(1, max(1, query_act_steps) + 1):
        carry, outputs = model(carry, batch)
        if bool(carry.halted.all()):
            break
    assert outputs is not None
    logits = outputs["logits"]
    pred = logits.argmax(dim=-1)
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    correct = (pred == labels) & mask
    n_sup = int(mask.sum().item())
    token_acc = correct.float().sum().item() / max(n_sup, 1)
    exact = float(bool((correct.sum(dim=-1) == mask.sum(dim=-1)).all()))
    q_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1).long(),
        ignore_index=IGNORE_LABEL_ID,
        reduction="sum",
    )
    query_ce_mean = (q_ce / max(n_sup, 1)).item()

    pred_np = pred[0].detach().cpu().numpy().astype(np.int32)
    logits_np: Optional[np.ndarray] = None
    if return_logits:
        logits_np = logits[0].detach().float().cpu().numpy()

    raw: Optional[Dict[str, object]] = None
    if return_raw:
        seq_len = int(labels.shape[1])
        p0 = pred[0].cpu().tolist()
        y0 = labels[0].cpu().tolist()
        mismatches: List[List[int]] = []
        for i in range(seq_len):
            yi = int(y0[i])
            if yi == IGNORE_LABEL_ID:
                continue
            pi = int(p0[i])
            if pi != yi:
                mismatches.append([i, pi, yi])
        raw = {
            "seq_len": seq_len,
            "n_supervised": n_sup,
            "pred": p0,
            "labels": y0,
            "input": batch["inputs"][0].cpu().tolist(),
            "supervised_mask": [int(y0[i] != IGNORE_LABEL_ID) for i in range(seq_len)],
            "mismatches": mismatches,
            "last_support_index": int(outputs["support_indices"][0].item()) if "support_indices" in outputs else -1,
        }
    return token_acc, exact, steps, query_ce_mean, n_sup, pred_np, raw, logits_np


def episode_indices(arrays: Dict[str, np.ndarray], min_support: int, max_episodes: int, seed: int) -> List[int]:
    support_counts = arrays["support_mask"].sum(axis=1)
    eligible = np.flatnonzero(support_counts >= min_support)
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    if max_episodes > 0:
        eligible = eligible[:max_episodes]
    return eligible.tolist()


def sample_episodes_per_base_task(
    arrays: Dict[str, np.ndarray],
    min_support: int,
    n_tasks: int,
    episodes_per_task: int,
    seed: int,
) -> List[int]:
    """Sample ``n_tasks`` base ARC tasks (``task_identifiers``), up to ``episodes_per_task`` rows each.

    Lets you report e.g. 10 tasks with 4 episodes each (partial pass@4 over sampled augs/queries).
    """
    if arrays.get("task_identifiers") is None:
        raise FileNotFoundError(
            "Missing all__task_identifiers.npy in this data split; "
            "use flat --max-episodes sampling or rebuild the support-TTT dataset."
        )
    support_counts = arrays["support_mask"].sum(axis=1)
    eligible = np.flatnonzero(support_counts >= min_support)
    rng = np.random.default_rng(seed)
    by_task: Dict[int, List[int]] = {}
    for row in eligible:
        tid = int(arrays["task_identifiers"][int(row)])
        by_task.setdefault(tid, []).append(int(row))
    task_ids = [t for t, rows in by_task.items() if rows]
    rng.shuffle(task_ids)
    task_ids = task_ids[: max(0, n_tasks)]
    rows_out: List[int] = []
    for tid in task_ids:
        rlist = by_task[tid][:]
        rng.shuffle(rlist)
        rows_out.extend(rlist[: max(1, episodes_per_task)])
    return rows_out


def task_level_aggregates(
    episode_records: List[Tuple[int, Dict[str, float]]],
    task_identifiers: Optional[np.ndarray],
) -> Dict[str, float]:
    """Per base task (``task_identifiers``): max / mean over that task's episode rows in this run."""
    if task_identifiers is None or not episode_records:
        return {}
    by_task: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for row, m in episode_records:
        by_task[int(task_identifiers[int(row)])].append(m)
    max_exact: List[float] = []
    max_tok: List[float] = []
    mean_tok: List[float] = []
    best_query_ce: List[float] = []
    max_exact_arc: List[float] = []
    for _tid, lst in by_task.items():
        max_exact.append(max(x["exact"] for x in lst))
        max_tok.append(max(x["token_acc"] for x in lst))
        mean_tok.append(sum(x["token_acc"] for x in lst) / len(lst))
        best_query_ce.append(min(x["query_ce"] for x in lst))
        arc_vals = [
            float(x["exact_arc"])
            for x in lst
            if "exact_arc" in x and isinstance(x["exact_arc"], (int, float)) and math.isfinite(float(x["exact_arc"]))
        ]
        max_exact_arc.append(max(arc_vals) if arc_vals else float("nan"))
    n = len(by_task)
    out: Dict[str, float] = {
        "task_level/num_base_tasks": float(n),
        "task_level/mean_max_exact": sum(max_exact) / max(n, 1),
        "task_level/mean_max_token_acc": sum(max_tok) / max(n, 1),
        "task_level/mean_mean_token_acc": sum(mean_tok) / max(n, 1),
        "task_level/mean_best_query_ce": sum(best_query_ce) / max(n, 1),
        "task_level/fraction_tasks_any_exact": sum(1 for x in max_exact if x >= 0.999) / max(n, 1),
    }
    finite_arc = [x for x in max_exact_arc if math.isfinite(x)]
    if finite_arc:
        out["task_level/mean_max_exact_arc"] = sum(finite_arc) / len(finite_arc)
        out["task_level/fraction_tasks_any_exact_arc"] = (
            sum(1 for x in max_exact_arc if math.isfinite(x) and x >= 0.999) / max(n, 1)
        )
    return out


def adapt_one_episode(
    loss_head: nn.Module,
    adapt_params: Sequence[nn.Parameter],
    initial_state: Dict[str, torch.Tensor],
    row: int,
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    model = loss_head.model
    load_adapter_state(loss_head, initial_state)
    model_cfg = model.config
    max_support = int(model_cfg.max_support_examples)

    support_mask = arrays["support_mask"][row].astype(bool)
    valid = np.flatnonzero(support_mask)
    support_inputs_all = arrays["support_inputs"][row, valid]
    support_outputs_all = arrays["support_outputs"][row, valid]
    n_support = len(valid)

    want_logits = bool(getattr(args, "poe", False))
    if args.no_ttt:
        query_batch = make_batch(
            arrays["inputs"][row],
            arrays["labels"][row],
            support_inputs_all,
            support_outputs_all,
            int(arrays["puzzle_identifiers"][row]),
            max_support,
            device,
            args.ignore_label_id,
        )
        token_acc, exact, infer_steps, query_ce_mean, n_sup, pred_np, raw, logits_np = predict_query(
            model,
            query_batch,
            args.query_act_steps,
            return_raw=bool(getattr(args, "dump_raw_tokens", False)),
            return_logits=want_logits,
        )
        exact_arc = float("nan")
        ids = getattr(args, "identifiers", []) or []
        if ids:
            r = arc_cropped_grid_match(
                pred_np,
                np.asarray(arrays["labels"][row]),
                int(arrays["puzzle_identifiers"][row]),
                ids,
            )
            if r is not None:
                exact_arc = float(r)
        core = {
            "token_acc": token_acc,
            "exact": exact,
            "exact_arc": exact_arc,
            "val_loss": float("nan"),
            "support_count": float(n_support),
            "infer_steps": float(infer_steps),
            "query_ce": query_ce_mean,
            "supervised_tokens": float(n_sup),
            "pred_np": pred_np,
        }
        if logits_np is not None:
            core["logits_np"] = logits_np
        extra = _episode_raw_payload(row, arrays, raw) if raw is not None else None
        return core, extra

    fold_order = list(range(n_support))
    if args.max_folds > 0:
        fold_order = fold_order[: args.max_folds]

    best_val = float("inf")
    best_state = copy.deepcopy(initial_state)

    for val_local_idx in fold_order:
        load_adapter_state(loss_head, initial_state)
        train_local = [i for i in range(n_support) if i != val_local_idx]
        # Plain SGD (momentum=0) to match the meta-training inner loop
        # (_shared_batch_inner_adapt does param -= inner_lr * grad). The Reptile
        # meta-init is optimised for an SGD inner loop; AdamW adapts off-trajectory.
        optim = torch.optim.SGD(adapt_params, lr=args.ttt_lr, momentum=0.0, weight_decay=args.ttt_weight_decay)

        for step in range(args.ttt_steps):
            target_local_idx = train_local[step % len(train_local)]
            context = [i for i in train_local if i != target_local_idx]
            if not context:
                context = train_local
            batch = make_batch(
                support_inputs_all[target_local_idx],
                support_outputs_all[target_local_idx],
                support_inputs_all[context],
                support_outputs_all[context],
                int(arrays["puzzle_identifiers"][row]),
                max_support,
                device,
                args.ignore_label_id,
            )
            optim.zero_grad(set_to_none=True)
            loss = forward_loss(model, batch, act_steps=args.adapt_act_steps, train_mode=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapt_params, args.grad_clip)
            optim.step()

        val_batch = make_batch(
            support_inputs_all[val_local_idx],
            support_outputs_all[val_local_idx],
            support_inputs_all[train_local],
            support_outputs_all[train_local],
            int(arrays["puzzle_identifiers"][row]),
            max_support,
            device,
            args.ignore_label_id,
        )
        with torch.no_grad():
            val_loss = float(forward_loss(model, val_batch, act_steps=args.eval_act_steps, train_mode=False).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = adapter_state_dict(loss_head, args.adapt_params)

    load_adapter_state(loss_head, best_state)
    query_batch = make_batch(
        arrays["inputs"][row],
        arrays["labels"][row],
        support_inputs_all,
        support_outputs_all,
        int(arrays["puzzle_identifiers"][row]),
        max_support,
        device,
        args.ignore_label_id,
    )
    token_acc, exact, infer_steps, query_ce_mean, n_sup, pred_np, raw, logits_np = predict_query(
        model,
        query_batch,
        args.query_act_steps,
        return_raw=bool(getattr(args, "dump_raw_tokens", False)),
        return_logits=want_logits,
    )
    exact_arc = float("nan")
    ids = getattr(args, "identifiers", []) or []
    if ids:
        r = arc_cropped_grid_match(
            pred_np,
            np.asarray(arrays["labels"][row]),
            int(arrays["puzzle_identifiers"][row]),
            ids,
        )
        if r is not None:
            exact_arc = float(r)
    core = {
        "token_acc": token_acc,
        "exact": exact,
        "exact_arc": exact_arc,
        "val_loss": best_val,
        "support_count": float(n_support),
        "infer_steps": float(infer_steps),
        "query_ce": query_ce_mean,
        "supervised_tokens": float(n_sup),
        "pred_np": pred_np,
    }
    if logits_np is not None:
        core["logits_np"] = logits_np
    extra = _episode_raw_payload(row, arrays, raw) if raw is not None else None
    return core, extra


def adapt_one_episode_d8(
    loss_head: nn.Module,
    adapt_params: Sequence[nn.Parameter],
    initial_state: Dict[str, torch.Tensor],
    row: int,
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    """D8 ensemble: run adapt_one_episode under each of 8 dihedral transforms, majority-vote.

    Each rotation has its own LoRA adaptation (rotated supports) and prediction
    (rotated query). Predictions are inverse-rotated to the original frame and
    voted per-cell. Final metrics are computed against original labels.
    """
    from scripts.experiments.evaluate.d8_helpers import (
        _content_bbox,
        color_perm_logits,
        color_perm_seq,
        color_perm_seq_batch,
        dihedral_logits_to_top_left,
        dihedral_seq,
        dihedral_seq_batch,
        inverse_pred_to_original_frame,
        invert_color_perm,
        label_content_position,
        majority_vote_per_position,
        poe_aggregate,
        repack_to_position,
        sample_color_perm,
        top_k_candidates_per_position,
    )

    use_poe = bool(getattr(args, "poe", False))
    logits_stack: List[np.ndarray] = []

    preds_top_left: List[np.ndarray] = []
    rotated_metrics: List[Dict[str, float]] = []
    n_support = int(arrays["support_mask"][row].astype(bool).sum())
    orig_label_pad_r, orig_label_pad_c = label_content_position(
        arrays["labels"][row].astype(np.int32)
    )

    n_color_perms = max(1, int(getattr(args, "color_perms", 1)))
    color_perm_seeds = [0] + list(range(1, n_color_perms))  # 0 = identity
    skip_rotations = bool(getattr(args, "no_d8_rotations", False))
    tid_range = (0,) if skip_rotations else tuple(range(8))

    for cp_idx, cp_seed in enumerate(color_perm_seeds):
        cp = sample_color_perm(cp_seed)
        cp_inv = invert_color_perm(cp)
        for tid in tid_range:
            arrays_rot = dict(arrays)
            arrays_rot["inputs"] = arrays["inputs"].copy()
            arrays_rot["labels"] = arrays["labels"].copy()
            arrays_rot["support_inputs"] = arrays["support_inputs"].copy()
            arrays_rot["support_outputs"] = arrays["support_outputs"].copy()
            inp = color_perm_seq(arrays["inputs"][row].astype(np.int32), cp)
            lbl = color_perm_seq(arrays["labels"][row].astype(np.int32), cp)
            sup_in = color_perm_seq_batch(arrays["support_inputs"][row].astype(np.int32), cp)
            sup_out = color_perm_seq_batch(arrays["support_outputs"][row].astype(np.int32), cp)
            if tid != 0:
                inp = dihedral_seq(inp, tid)
                lbl = dihedral_seq(lbl, tid)
                sup_in = dihedral_seq_batch(sup_in, tid)
                sup_out = dihedral_seq_batch(sup_out, tid)
            arrays_rot["inputs"][row] = inp
            arrays_rot["labels"][row] = lbl
            arrays_rot["support_inputs"][row] = sup_in
            arrays_rot["support_outputs"][row] = sup_out

            m, _ = adapt_one_episode(loss_head, adapt_params, initial_state, row, arrays_rot, args, device)
            pred_rot = m["pred_np"]
            pred_top_left = inverse_pred_to_original_frame(pred_rot, tid)
            pred_top_left = color_perm_seq(pred_top_left, cp_inv)
            preds_top_left.append(pred_top_left.astype(np.int32))
            if use_poe and m.get("logits_np") is not None:
                rotated_input = arrays_rot["inputs"][row].astype(np.int32).reshape(30, 30)
                bbox = _content_bbox(rotated_input)
                rot_logits = dihedral_logits_to_top_left(m["logits_np"].astype(np.float32), tid, bbox)
                rot_logits = color_perm_logits(rot_logits, cp_inv)
                logits_stack.append(rot_logits)
            rotated_metrics.append({k: v for k, v in m.items() if k not in ("pred_np", "logits_np")})
            print(f"  [d8 row={row} cp={cp_idx} tid={tid}] exact={m.get('exact', 0.0):.0f} arc={m.get('exact_arc', float('nan')):.2f}", flush=True)

    preds_stack = np.stack(preds_top_left, axis=0)
    top_k = max(1, int(getattr(args, "top_k_candidates", 1)))
    if use_poe and logits_stack:
        candidates_top_left = poe_aggregate(np.stack(logits_stack, axis=0), top_k=top_k)
    else:
        candidates_top_left = top_k_candidates_per_position(preds_stack, k=top_k)

    orig_labels = arrays["labels"][row].astype(np.int32)
    if args.ignore_label_id is not None:
        mask = orig_labels != int(args.ignore_label_id)
    else:
        mask = orig_labels != IGNORE_LABEL_ID
    n_sup_total = int(mask.sum())
    ids = getattr(args, "identifiers", []) or []

    val_losses = [m["val_loss"] for m in rotated_metrics if not math.isnan(m["val_loss"])]
    val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
    query_ces = [m["query_ce"] for m in rotated_metrics if math.isfinite(m["query_ce"])]
    query_ce = sum(query_ces) / len(query_ces) if query_ces else float("nan")
    infer_steps = sum(m["infer_steps"] for m in rotated_metrics) / len(rotated_metrics)
    n_rot = max(len(rotated_metrics), 1)
    d8_single_exact_mean = sum(m["exact"] for m in rotated_metrics) / n_rot
    d8_single_arc_mean = sum(
        float(m["exact_arc"]) for m in rotated_metrics if math.isfinite(float(m["exact_arc"]))
    ) / max(sum(1 for m in rotated_metrics if math.isfinite(float(m["exact_arc"]))), 1)

    cand_metrics_list: List[Dict[str, float]] = []
    for ci in range(top_k):
        voted = repack_to_position(
            candidates_top_left[ci], orig_label_pad_r, orig_label_pad_c
        ).astype(np.int32)
        correct = (voted == orig_labels) & mask
        token_acc = float(correct.sum()) / max(n_sup_total, 1)
        exact = float(correct.sum() == mask.sum())
        exact_arc = float("nan")
        if ids:
            r = arc_cropped_grid_match(
                voted,
                orig_labels,
                int(arrays["puzzle_identifiers"][row]),
                ids,
            )
            if r is not None:
                exact_arc = float(r)
        cand_metrics_list.append(
            {
                "token_acc": token_acc,
                "exact": exact,
                "exact_arc": exact_arc,
                "val_loss": val_loss,
                "support_count": float(n_support),
                "infer_steps": float(infer_steps),
                "query_ce": query_ce,
                "supervised_tokens": float(n_sup_total),
                "d8/single_pass_exact_mean": d8_single_exact_mean,
                "d8/single_pass_arc_mean": d8_single_arc_mean,
                "d8/n_rotations": float(n_rot),
                "d8/candidate_index": float(ci),
            }
        )

    core = cand_metrics_list[0]
    if top_k > 1:
        core["_alt_metrics"] = cand_metrics_list[1:]
    return core, None


def _episode_raw_payload(row: int, arrays: Dict[str, np.ndarray], raw: Dict[str, object]) -> Dict[str, Any]:
    tid = None
    if arrays.get("task_identifiers") is not None:
        tid = int(arrays["task_identifiers"][row])
    return {
        "row": int(row),
        "puzzle_identifier": int(arrays["puzzle_identifiers"][row]),
        "task_identifier": tid,
        **raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Support-TTT leave-one-out LoRA evaluation")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to latest in --checkpoint-dir.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to all_config.yaml (default: <checkpoint-dir>/all_config.yaml).",
    )
    parser.add_argument(
        "--wandb-run",
        default=None,
        help="If local all_config.yaml is missing: W&B run path entity/project/run_id (e.g. entity/arm-arc-agi/run_id).",
    )
    parser.add_argument("--checkpoint-dir", default="experiments/train-trm-abstraction-support-ttt-emp64-l40s/checkpoints")
    parser.add_argument("--ema", action="store_true", help="Use EMA shadow if available.")
    parser.add_argument("--data-path", default="data/trm-abstraction-support-ttt-arc1-aug-1000")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--mode", required=True, choices=["last_cycle", "full_h_cycles", "all_act_steps"])
    parser.add_argument("--adapt-params", default="lora", choices=["lora", "full_weight"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--ttt-steps", type=int, default=20)
    parser.add_argument("--ttt-lr", type=float, default=1e-4)
    parser.add_argument("--ttt-weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--max-folds", type=int, default=0, help="0 means all leave-one-out folds.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=256,
        help="Cap on shuffled eligible rows; use 0 for all eligible (test split can be ~1e6+ rows, min_support permitting).",
    )
    parser.add_argument(
        "--result-suffix",
        default=None,
        help="If set, append _SUFFIX to the output JSON filename (safe for batch / meta-validation runs).",
    )
    parser.add_argument(
        "--sample-tasks",
        type=int,
        default=0,
        help="If >0, sample this many distinct base tasks (requires all__task_identifiers.npy); "
        "ignores flat --max-episodes for row selection.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=4,
        help="With --sample-tasks, up to this many episode rows per base task.",
    )
    parser.add_argument(
        "--no-ttt",
        action="store_true",
        help="Skip adaptation; evaluate query prediction at initial adapter / loaded weights only.",
    )
    parser.add_argument(
        "--query-act-steps",
        type=int,
        default=-1,
        help="Outer ACT forward passes on the query batch (matches meta-training query when set to "
        "arch.meta.query_act_steps). Default -1: read from config meta, else model halt_max_steps.",
    )
    parser.add_argument(
        "--dump-raw-tokens",
        action="store_true",
        help="Include full pred/labels token lists and mismatch indices per episode in the JSON output.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--pass-at-ks",
        type=str,
        default="1,2,4,8",
        help="Comma-separated K for pass@K: groups rows by (task_id, query_sources, query_indices).",
    )
    parser.add_argument(
        "--d8-ensemble",
        action="store_true",
        help="Test-time D8 (dihedral group) ensemble: 8 rotations, separate adapt+predict each, majority-vote output. ~8x cost.",
    )
    parser.add_argument(
        "--color-perms",
        type=int,
        default=1,
        help="When --d8-ensemble is on, also compose K color permutations (K=1 means identity only). Total ensemble = 8*K (or K if --no-d8-rotations).",
    )
    parser.add_argument(
        "--no-d8-rotations",
        action="store_true",
        help="When --d8-ensemble is on, skip the 8 dihedral rotations and use only identity transform. Combine with --color-perms K to ensemble over K color permutations only.",
    )
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=1,
        help="When --d8-ensemble is on, emit top-K per-cell vote candidates as alternate predictions for pass@K (Kaggle's k=2 setting).",
    )
    parser.add_argument(
        "--poe",
        action="store_true",
        help="Product-of-experts aggregation: sum log-softmax across rotations instead of majority-vote on argmax. Top-K candidates are ranked by summed log-prob.",
    )
    parser.add_argument(
        "--ablate-emp",
        choices=["none", "zero", "pad"],
        default="none",
        help="EMP token ablation. 'zero': zero out delta_embedder_emp row for emp_token_id (no EMP signal). 'pad': remap emp_token_id to 0 so 'same' positions use the pad embedding.",
    )
    parser.add_argument(
        "--emp-id-override",
        type=int,
        default=-1,
        help="Phase-1 C4 ablation: remap emp_token_id to N at eval time (overrides --ablate-emp pad). "
             "0=pad, 1=EOS, 2..11=color tokens. Default -1 = no override.",
    )
    parser.add_argument(
        "--zero-delta-read-block",
        action="store_true",
        help="Phase-1 C2 ablation: zero out DeltaReadBlock weights so H_level cannot read support. "
             "Tests whether cross-attn to support is doing the work or if TTT alone explains the gain.",
    )
    parser.add_argument(
        "--measure-flops",
        action="store_true",
        help="Probe FLOPs per forward via torch.utils.flop_counter.FlopCounterMode on the first episode. "
             "Records flops_per_forward in the result JSON. Adds ~1s startup overhead.",
    )
    parser.add_argument(
        "--identity-aug-only",
        action="store_true",
        help="Restrict evaluation to the IDENTITY augmentation rows (aug0, transform t0, identity color perm 0123456789). "
             "For arc1-aug-1000/test this yields the canonical ARC-AGI-1 evaluation pairs as published by Chollet — "
             "the apples-to-apples comparison vs literature TRM ~45% pass@1.",
    )
    parser.add_argument(
        "--h-cycles-override",
        type=int,
        default=-1,
        help="At inference, override model.config.H_cycles. Default -1 = keep training value. "
             "TRM-style: more H-cycles at inference = deeper recursive reasoning at the cost of "
             "linear extra compute per forward.",
    )
    parser.add_argument(
        "--l-cycles-override",
        type=int,
        default=-1,
        help="At inference, override model.config.L_cycles. Default -1 = keep training value.",
    )
    args = parser.parse_args()

    args.data_path = str(Path(args.data_path).expanduser().resolve())

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    pass_ks = parse_pass_at_ks(args.pass_at_ks)

    os.environ.setdefault("DISABLE_COMPILE", "1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(Path(args.checkpoint_dir))
    ckpt_dir = checkpoint.parent
    step = extract_step(str(checkpoint))
    print(f"Using checkpoint: {checkpoint} (step {step})", flush=True)

    config = load_support_config(
        checkpoint,
        config_override=Path(args.config) if args.config else None,
        wandb_run=args.wandb_run,
    )
    config.global_batch_size = 1
    config.support_ttt_mode = True
    config.data_paths = [args.data_path]

    _, train_metadata = create_dataloader(
        config, "train", rank=0, world_size=1,
        test_set_mode=False, epochs_per_iter=1, global_batch_size=1,
    )
    args.ignore_label_id = train_metadata.ignore_label_id
    print(
        f"Label masking: ignore_label_id={args.ignore_label_id!r} "
        "(padded canvas cells excluded from token_acc, exact, and CE; matches training collate).",
        flush=True,
    )
    print(
        "Logged ttt/token_acc, ttt/exact, etc. are running means over *episodes* (each row is a new puzzle): "
        "the [n/N] lines show convergence to the split mean, not improvement over inner TTT SGD steps. "
        "Query metrics are computed once after LoRA adaptation finishes (see --ttt-steps). "
        "Compare with --no-ttt on the same rows to measure adaptation gain.",
        flush=True,
    )
    print(
        "Aggregate ttt/token_acc weights each episode equally (mean of per-episode token fractions); "
        "ttt/supervised_tokens is the mean supervised cell count per query.",
        flush=True,
    )
    print(
        "With --sample-tasks, also see task_level/* (per-base-task aggregation).",
        flush=True,
    )
    loss_head = create_support_model_for_eval(config, train_metadata, checkpoint, use_ema=args.ema, device=device)
    model = loss_head.model.to(device)
    model.eval()

    flops_per_forward: Optional[int] = None
    if getattr(args, "measure_flops", False):
        try:
            from torch.utils.flop_counter import FlopCounterMode
            print("[measure-flops] probing FLOPs per forward...", flush=True)
        except ImportError:
            print("[measure-flops] FlopCounterMode unavailable; skipping", flush=True)
            FlopCounterMode = None  # type: ignore[assignment]
        if FlopCounterMode is not None:
            # Build a representative batch using the same shape as the actual eval.
            # Use the first row from the eval data later via a fresh dummy forward.
            # We construct a dummy zero batch matching expected shapes.
            seq_len = int(model.config.seq_len)
            max_sup = int(model.config.max_support_examples)
            dummy_inputs = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
            dummy_labels = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
            dummy_sup_in = torch.zeros(1, max_sup, seq_len, dtype=torch.int64, device=device)
            dummy_sup_out = torch.zeros(1, max_sup, seq_len, dtype=torch.int64, device=device)
            dummy_sup_mask = torch.zeros(1, max_sup, dtype=torch.bool, device=device)
            dummy_sup_mask[:, :3] = True  # 3 valid supports
            dummy_pid = torch.zeros(1, dtype=torch.int64, device=device)
            dummy_batch = {
                "inputs": dummy_inputs,
                "labels": dummy_labels,
                "support_inputs": dummy_sup_in,
                "support_outputs": dummy_sup_out,
                "support_mask": dummy_sup_mask,
                "puzzle_identifiers": dummy_pid,
            }
            with torch.no_grad():
                with FlopCounterMode(display=False) as fc:
                    with torch.device(device):
                        carry = model.initial_carry(dummy_batch)
                    _, _ = model(carry, dummy_batch)
                flops_per_forward = int(fc.get_total_flops())
            params_total = sum(p.numel() for p in model.parameters())
            print(f"[measure-flops] flops/forward = {flops_per_forward:,}  params = {params_total:,}", flush=True)
            args._flops_per_forward = flops_per_forward
            args._params_total = params_total

    h_override = int(getattr(args, "h_cycles_override", -1))
    l_override = int(getattr(args, "l_cycles_override", -1))
    if h_override > 0 or l_override > 0:
        inner = getattr(model, "inner", model)
        if h_override > 0:
            old_h = int(model.config.H_cycles)
            model.config.H_cycles = h_override
            if hasattr(inner, "config"):
                inner.config.H_cycles = h_override
            print(f"[cycles-override] H_cycles {old_h} -> {h_override}", flush=True)
        if l_override > 0:
            old_l = int(model.config.L_cycles)
            model.config.L_cycles = l_override
            if hasattr(inner, "config"):
                inner.config.L_cycles = l_override
            print(f"[cycles-override] L_cycles {old_l} -> {l_override}", flush=True)

    ablate_mode = getattr(args, "ablate_emp", "none")
    if ablate_mode and ablate_mode != "none":
        emp_id = int(getattr(model.config, "emp_token_id", 12))
        if getattr(model.config, "support_evidence_mode", "") != "emp_token":
            print(f"[ablate-emp] skip: support_evidence_mode != emp_token", flush=True)
        elif ablate_mode == "zero":
            inner = getattr(model, "inner", model)
            with torch.no_grad():
                w = inner.delta_embedder_emp.token_emb.embedding_weight  # type: ignore[attr-defined]
                w[emp_id].zero_()
            print(f"[ablate-emp] zeroed inner.delta_embedder_emp.token_emb row {emp_id}", flush=True)
        elif ablate_mode == "pad":
            inner = getattr(model, "inner", model)
            model.config.emp_token_id = 0  # type: ignore[attr-defined]
            if hasattr(inner, "config"):
                inner.config.emp_token_id = 0  # type: ignore[attr-defined]
            print(f"[ablate-emp] remapped emp_token_id 12 -> 0 (pad embedding at 'same' positions)", flush=True)
        else:
            raise ValueError(f"unknown --ablate-emp {ablate_mode!r}")

    emp_id_override = int(getattr(args, "emp_id_override", -1))
    if emp_id_override >= 0:
        inner = getattr(model, "inner", model)
        if getattr(model.config, "support_evidence_mode", "") != "emp_token":
            print(f"[emp-id-override] skip: support_evidence_mode != emp_token", flush=True)
        else:
            vocab_max = int(getattr(model.config, "delta_vocab_size", 13)) - 1
            if emp_id_override > vocab_max:
                raise ValueError(f"--emp-id-override {emp_id_override} > delta_vocab_size-1 ({vocab_max})")
            old_id = int(getattr(model.config, "emp_token_id", 12))
            model.config.emp_token_id = emp_id_override  # type: ignore[attr-defined]
            if hasattr(inner, "config"):
                inner.config.emp_token_id = emp_id_override  # type: ignore[attr-defined]
            print(f"[emp-id-override] remapped emp_token_id {old_id} -> {emp_id_override}", flush=True)

    if getattr(args, "zero_delta_read_block", False):
        inner = getattr(model, "inner", model)
        if not hasattr(inner, "delta_read_block"):
            print(f"[zero-delta-read-block] skip: model has no delta_read_block", flush=True)
        else:
            n_params = 0
            with torch.no_grad():
                for p in inner.delta_read_block.parameters():  # type: ignore[attr-defined]
                    p.zero_()
                    n_params += p.numel()
            print(f"[zero-delta-read-block] zeroed {n_params:,} params in DeltaReadBlock — H_level cannot read support", flush=True)

    if args.query_act_steps < 0:
        args.query_act_steps = query_act_steps_default_from_arch(config, int(model.config.halt_max_steps))
    else:
        args.query_act_steps = max(1, int(args.query_act_steps))
    print(
        f"Query inference: query_act_steps={args.query_act_steps} "
        f"(meta-training uses the same cap in forward_loss_and_outputs on the query batch).",
        flush=True,
    )

    if args.mode == "last_cycle":
        model.config.grad_cycles = 1
        model.inner.config.grad_cycles = 1
        adapt_act_steps = 1
        include_h = False
    elif args.mode == "full_h_cycles":
        model.config.grad_cycles = model.config.H_cycles
        model.inner.config.grad_cycles = model.inner.config.H_cycles
        adapt_act_steps = 1
        include_h = True
    else:
        model.config.grad_cycles = model.config.H_cycles
        model.inner.config.grad_cycles = model.inner.config.H_cycles
        adapt_act_steps = model.config.halt_max_steps
        include_h = True
    args.adapt_act_steps = adapt_act_steps
    args.eval_act_steps = min(model.config.halt_max_steps, max(1, adapt_act_steps))

    inner = _unwrap_inner(loss_head)
    h_is_meta_lora = isinstance(inner.H_level, HLevelLoRA)
    if h_is_meta_lora and include_h:
        print(
            "Note: checkpoint includes meta-training H-level LoRA; adding a second H-level LoRA "
            "wrapper is not supported. Adapting delta_read + broadcast LoRA only (TTT include_h=False).",
            flush=True,
        )
        include_h = False

    if args.adapt_params == "lora":
        install_lora(loss_head, rank=args.lora_rank, include_h=include_h)
        for name, p in loss_head.named_parameters():
            if ".delta_read_block.base." in name or ".broadcast.base." in name:
                p.requires_grad_(False)
            if ".H_level." in name:
                p.requires_grad_(False)
        selected_params = [p for p in loss_head.parameters() if p.requires_grad]
    else:
        for p in loss_head.parameters():
            p.requires_grad_(True)
        selected_params = [p for p in loss_head.parameters() if p.requires_grad]
    initial_state = adapter_state_dict(loss_head, args.adapt_params)
    print(
        f"Adapt params: {args.adapt_params} | params={sum(p.numel() for p in selected_params):,} "
        f"| mode={args.mode} | adapt_act_steps={adapt_act_steps}"
        f"{' | no_ttt (zero-shot)' if args.no_ttt else ''}",
        flush=True,
    )

    arrays = load_episode_arrays(Path(args.data_path), args.split)

    identity_row_set: Optional[set] = None
    if getattr(args, "identity_aug_only", False):
        # Filter to rows whose identifier ends with "aug0|||t0|||0123456789"
        # (identity augmentation = canonical ARC eval pair).
        ident_path = Path(args.data_path) / "identifiers.json"
        if not ident_path.is_file():
            raise FileNotFoundError(f"--identity-aug-only requires {ident_path}")
        with open(ident_path) as f:
            ident_raw = json.load(f)
        if isinstance(ident_raw, dict):
            id_table = ident_raw.get(args.split, [])
        else:
            id_table = ident_raw
        identity_suffix = "|||aug0|||t0|||0123456789"
        pids = arrays["puzzle_identifiers"]
        identity_row_set = set()
        for i in range(len(pids)):
            p = int(pids[i])
            if 0 <= p < len(id_table):
                s = id_table[p]
                if isinstance(s, str) and s.endswith(identity_suffix):
                    identity_row_set.add(i)
        print(
            f"[identity-aug-only] filtering to identity rows (suffix='{identity_suffix}'): "
            f"{len(identity_row_set):,} of {len(pids):,} rows match",
            flush=True,
        )
        # CRITICAL: apply the filter BEFORE sampling, by marking non-identity
        # rows as ineligible (support_mask=0). Otherwise --sample-tasks would
        # draw mostly non-identity rows (1 in ~4455) and then drop them.
        valid_row = np.zeros(len(arrays["support_mask"]), dtype=bool)
        for r in identity_row_set:
            valid_row[r] = True
        arrays = dict(arrays)
        arrays["support_mask"] = arrays["support_mask"].copy()
        arrays["support_mask"][~valid_row] = 0
        post_eligible = int((arrays["support_mask"].sum(axis=1) >= args.min_support).sum())
        print(
            f"[identity-aug-only] eligible identity rows after min_support>={args.min_support}: {post_eligible}",
            flush=True,
        )

    if args.sample_tasks > 0:
        rows = sample_episodes_per_base_task(
            arrays,
            min_support=args.min_support,
            n_tasks=args.sample_tasks,
            episodes_per_task=args.episodes_per_task,
            seed=args.seed,
        )
        print(
            f"Sampled base tasks: {args.sample_tasks} (cap), {args.episodes_per_task} episodes/task cap "
            f"-> {len(rows)} episode rows",
            flush=True,
        )
    else:
        rows = episode_indices(arrays, min_support=args.min_support, max_episodes=args.max_episodes, seed=args.seed)

    # NOTE: identity_aug_only is now applied via support_mask pre-filter above,
    # so post-filter is a no-op (kept as a safety net).
    if identity_row_set is not None:
        before = len(rows)
        rows = [r for r in rows if r in identity_row_set]
        if len(rows) != before:
            print(f"[identity-aug-only] safety-net post-filter: {len(rows)} kept of {before}", flush=True)

    print(f"Eligible episodes selected: {len(rows)} (split={args.split}, min_support={args.min_support})", flush=True)

    args.identifiers = load_identifier_list(Path(args.data_path), args.split)
    ident_path = Path(args.data_path) / "identifiers.json"
    if args.identifiers:
        print(
            f"ARC eval alignment: loaded {len(args.identifiers)} identifiers for split={args.split!r} "
            f"from {ident_path} (exact_arc + pass@K); pass@K={pass_ks}",
            flush=True,
        )
    else:
        if not ident_path.is_file():
            print(
                f"ARC eval alignment: no file at {ident_path.resolve()} — "
                "exact_arc and arc pass@K skipped (ensure identifiers.json is synced next to the dataset).",
                flush=True,
            )
        else:
            print(
                f"ARC eval alignment: {ident_path} exists but yielded no identifier strings for "
                f"split={args.split!r}; exact_arc and arc pass@K skipped.",
                flush=True,
            )

    wb = None
    if args.wandb:
        import wandb

        wb_cfg = {**vars(args), "checkpoint": str(checkpoint), "checkpoint_step": step}
        ids = wb_cfg.get("identifiers")
        if isinstance(ids, list) and ids:
            wb_cfg["identifiers"] = f"<omitted {len(ids)} strings>"
            wb_cfg["identifiers_count"] = len(ids)

        wb = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "arm-arc-agi"),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=args.run_name or f"support-ttt-lora-{args.mode}-step-{step}",
            config=wb_cfg,
        )

    totals = {
        "token_acc": 0.0,
        "exact": 0.0,
        "exact_arc": 0.0,
        "support_count": 0.0,
        "infer_steps": 0.0,
        "query_ce": 0.0,
        "supervised_tokens": 0.0,
    }
    exact_arc_n = 0
    val_loss_sum = 0.0
    val_loss_n = 0
    episode_records: List[Tuple[int, Dict[str, float]]] = []
    episodes_raw: List[Dict[str, Any]] = []
    _adapt_fn = adapt_one_episode_d8 if getattr(args, "d8_ensemble", False) else adapt_one_episode
    for n, row in enumerate(rows, start=1):
        metrics, raw_ep = _adapt_fn(loss_head, selected_params, initial_state, row, arrays, args, device)
        alt_metrics = metrics.pop("_alt_metrics", None) if isinstance(metrics, dict) else None
        episode_records.append((row, metrics))
        if alt_metrics:
            for am in alt_metrics:
                episode_records.append((row, am))
        if raw_ep is not None:
            episodes_raw.append(raw_ep)
        for k in totals:
            if k == "exact_arc":
                continue
            totals[k] += metrics[k]
        if math.isfinite(metrics.get("exact_arc", float("nan"))):
            exact_arc_n += 1
            totals["exact_arc"] += float(metrics["exact_arc"])
        if not math.isnan(metrics["val_loss"]):
            val_loss_sum += metrics["val_loss"]
            val_loss_n += 1
        if n == 1 or n % 10 == 0 or n == len(rows):
            agg: Dict[str, float] = {}
            for k, v in totals.items():
                if k == "exact_arc":
                    continue
                agg[f"ttt/{k}"] = v / n
            agg["ttt/exact_arc"] = (totals["exact_arc"] / max(exact_arc_n, 1)) if exact_arc_n else float("nan")
            agg["ttt/episodes"] = float(n)
            agg["ttt/val_loss"] = val_loss_sum / max(val_loss_n, 1) if val_loss_n else float("nan")
            print(
                f"[{n}/{len(rows)}] exact={agg['ttt/exact']:.4f} exact_arc={agg['ttt/exact_arc']:.4f} "
                f"token_acc={agg['ttt/token_acc']:.4f} "
                f"query_ce={agg['ttt/query_ce']:.4f} "
                f"sup_tok={agg['ttt/supervised_tokens']:.1f} "
                f"val_loss={agg['ttt/val_loss']:.4f}",
                flush=True,
            )
            if wb is not None:
                wb.log(agg, step=n)

    final: Dict[str, Any] = {}
    for k, v in totals.items():
        if k == "exact_arc":
            continue
        final[f"ttt/{k}"] = v / max(len(rows), 1)
    final["ttt/exact_arc"] = (
        (totals["exact_arc"] / max(exact_arc_n, 1)) if exact_arc_n else float("nan")
    )
    final["ttt/exact_arc_eval_rows"] = float(exact_arc_n)
    final["ttt/val_loss"] = val_loss_sum / max(val_loss_n, 1) if val_loss_n else float("nan")
    final.update(
        task_level_aggregates(episode_records, arrays.get("task_identifiers"))
    )
    if pass_ks and arrays.get("task_identifiers") is not None:
        final.update(pass_at_k_metrics(episode_records, arrays, pass_ks, "exact", "token_masked"))
        if args.identifiers:
            final.update(pass_at_k_metrics(episode_records, arrays, pass_ks, "exact_arc", "arc_cropped"))
    final["pass_at_Ks_configured"] = [int(x) for x in pass_ks]
    final.update({
        "ttt/episodes": len(rows),
        "checkpoint_step": step,
        "mode": args.mode,
        "checkpoint": str(checkpoint),
        "no_ttt": bool(args.no_ttt),
        "sample_tasks": int(args.sample_tasks),
        "episodes_per_task": int(args.episodes_per_task),
        "query_act_steps": int(args.query_act_steps),
        "split": args.split,
    })
    if episodes_raw:
        final["episodes_raw"] = episodes_raw

    def _json_safe(o):
        if isinstance(o, dict):
            return {k: _json_safe(v) for k, v in o.items()}
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o

    out_dir = ckpt_dir / ("support_ttt_full_weight_eval" if args.adapt_params == "full_weight" else "support_ttt_lora_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_parts: List[str] = []
    tag_parts.append(f"q{args.query_act_steps}")
    if args.dump_raw_tokens:
        tag_parts.append("raw")
    if args.no_ttt:
        tag_parts.append("nottt")
    if args.sample_tasks > 0:
        tag_parts.append(f"t{args.sample_tasks}x{args.episodes_per_task}")
    tag = ("_" + "_".join(tag_parts)) if tag_parts else ""
    rsfx = ""
    if getattr(args, "result_suffix", None):
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(args.result_suffix))
        rsfx = f"_{safe}" if safe else ""
    out_path = out_dir / f"step_{step}_{args.adapt_params}_{args.mode}{tag}_episodes_{len(rows)}{rsfx}.json"
    flops_pf = getattr(args, "_flops_per_forward", None)
    params_tot = getattr(args, "_params_total", None)
    if flops_pf is not None:
        if getattr(args, "d8_ensemble", False):
            tids_per_cp = 1 if getattr(args, "no_d8_rotations", False) else 8
            n_rotations = max(1, tids_per_cp * int(getattr(args, "color_perms", 1)))
        else:
            n_rotations = 1
        ttt_steps = int(args.ttt_steps) if not args.no_ttt else 0
        # forward+backward ~ 3× forward; per rotation: ttt_steps × 3 + 1 (final inference) forwards
        flops_per_rot = ttt_steps * 3 * flops_pf + flops_pf
        flops_per_task = n_rotations * flops_per_rot
        final["compute/flops_per_forward"] = int(flops_pf)
        final["compute/flops_per_task_est"] = int(flops_per_task)
        final["compute/flops_per_task_petaflops"] = flops_per_task / 1e15
        final["compute/params_total"] = int(params_tot) if params_tot is not None else None
        final["compute/n_rotations_per_task"] = n_rotations
        # $/task derivations (community-cloud price floor — community 4090, retail A100, retail H100)
        # eff_tflops = peak × 0.70 utilization
        final["compute/dollars_per_task_4090_community"] = (flops_per_task / (165e12 * 0.70)) * (0.40 / 3600)
        final["compute/dollars_per_task_a100_80gb"] =        (flops_per_task / (312e12 * 0.70)) * (1.30 / 3600)
        final["compute/dollars_per_task_h100_sxm5"] =        (flops_per_task / (989e12 * 0.70)) * (2.50 / 3600)
    with open(out_path, "w") as f:
        json.dump(_json_safe(final), f, indent=2)
    print(f"Final metrics: {final}", flush=True)
    print(f"Wrote {out_path}", flush=True)
    if wb is not None:
        wb.summary.update(final)
        wb.finish()


if __name__ == "__main__":
    main()
