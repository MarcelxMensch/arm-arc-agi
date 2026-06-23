"""Support-TTT meta-training: FOMAML, Reptile, OpenAI Reptile/FOML (LoRA + optional backbone).

Config: ``arch.meta`` (legacy ``arch.maml``). See ``arch.meta.meta_algorithm``.
"""


from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import coolname
import hydra
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn

from utils.logging import TrainLogger

from scripts.experiments.evaluate.evaluate_support_ttt_lora import (
    adapter_state_dict,
    install_lora,
    load_adapter_state,
)
from pretrain import (
    PretrainConfig,
    TrainState,
    create_dataloader,
    create_model,
    save_train_state,
)


IGNORE_LABEL_ID = -100

# Module-level q_halt regularizer weight. Set once in launch() from PretrainConfig.
# Read by forward_query_ce_with_act_metrics if the caller doesn't override it.
_Q_HALT_WEIGHT_DEFAULT: float = 0.0

# A8 Stage 1: supervised contrastive aux loss weight + temperature. Set once in launch().
_CONTRASTIVE_AUX_WEIGHT: float = 0.0
_CONTRASTIVE_TEMPERATURE: float = 0.1


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020), task-identity grouped.

    features: [B, D] (will be L2-normalized).
    labels:   [B] integer task IDs. Same id ⇒ positive, different ⇒ negative.

    Returns scalar mean over anchors that have at least one positive in the batch.
    Returns 0.0 (with a no-op grad path) if no anchor has positives.
    """
    B = features.shape[0]
    if B < 2:
        return features.sum() * 0.0
    f = F.normalize(features.float(), dim=-1)
    sim = (f @ f.T) / max(temperature, 1e-6)  # [B, B]
    self_mask = torch.eye(B, device=features.device, dtype=torch.bool)
    # Subtract max for numerical stability; large negative on diagonal excludes self
    # without producing -inf (which would NaN through `0 * -inf` later).
    LARGE_NEG = -1e4
    sim = sim.masked_fill(self_mask, LARGE_NEG)
    sim = sim - sim.detach().max(dim=1, keepdim=True).values  # stabilize
    sim = sim.masked_fill(self_mask, LARGE_NEG)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_mask = (labels.view(-1, 1) == labels.view(1, -1)) & ~self_mask
    n_positives = pos_mask.sum(dim=1)
    has_pos = n_positives > 0
    if not has_pos.any():
        return features.sum() * 0.0
    # Mask out non-positive entries explicitly with `where` (avoids 0 * -inf = NaN).
    masked_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    pos_log_prob = masked_log_prob.sum(dim=1) / n_positives.clamp_min(1).float()
    return -pos_log_prob[has_pos].mean()


def maybe_add_contrastive_aux(
    query_loss: torch.Tensor,
    q_outputs: Dict[str, torch.Tensor],
    active: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[float]]:
    """Add supervised contrastive aux on z_H (mean-pooled) when enabled.

    Returns (possibly augmented) query_loss and the raw contrastive scalar (or None).
    """
    if _CONTRASTIVE_AUX_WEIGHT <= 0.0:
        return query_loss, None
    z_H = q_outputs.get("z_H_with_grad")
    if z_H is None:
        return query_loss, None
    pids = active.get("puzzle_identifiers")
    if pids is None or pids.numel() < 2:
        return query_loss, None
    pooled = z_H.mean(dim=1)  # [B, D_H] — task representation per episode
    con = supervised_contrastive_loss(pooled, pids.long(), _CONTRASTIVE_TEMPERATURE)
    return query_loss + _CONTRASTIVE_AUX_WEIGHT * con, float(con.detach().item())


def ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1).long(),
        ignore_index=IGNORE_LABEL_ID,
    )


def forward_query_ce_with_act_metrics(
    loss_head: nn.Module,
    batch: Dict[str, torch.Tensor],
    act_steps: int,
    train_mode: bool,
    q_halt_weight: Optional[float] = None,
) -> Tuple[torch.Tensor, Any, Dict[str, torch.Tensor], int]:
    """Query forward: mean CE (meta loss) for backward; carry/outputs for logging.

    When ``q_halt_weight > 0``, also adds BCE(q_halt_logits, seq_is_correct) to the
    objective. This propagates gradient through ``q_head`` and (transitively) through
    ``H_level`` even at ``grad_cycles=1``, partially fixing the dead-H_level-LoRA bug.
    """

    if q_halt_weight is None:
        q_halt_weight = _Q_HALT_WEIGHT_DEFAULT
    model = loss_head.model
    model.train(train_mode)
    with torch.device(batch["inputs"].device):
        carry = model.initial_carry(batch)
    total_ce: Optional[torch.Tensor] = None
    total_qhalt: Optional[torch.Tensor] = None
    n_ce_steps = 0
    final_carry, final_outputs = carry, {}  # type: ignore

    for _ in range(1, act_steps + 1):
        carry, outputs = model(carry, batch)
        loss = ce_loss_from_logits(outputs["logits"], batch["labels"])
        total_ce = loss if total_ce is None else total_ce + loss
        if q_halt_weight > 0.0 and "q_halt_logits" in outputs:
            with torch.no_grad():
                pred = outputs["logits"].argmax(dim=-1)
                labels = batch["labels"]
                mask = labels != IGNORE_LABEL_ID
                loss_counts = mask.sum(dim=-1)
                is_correct_mask = mask & (pred == labels)
                seq_is_correct = (is_correct_mask.sum(dim=-1) == loss_counts).to(
                    outputs["q_halt_logits"].dtype
                )
            q_loss = F.binary_cross_entropy_with_logits(
                outputs["q_halt_logits"], seq_is_correct, reduction="mean"
            )
            total_qhalt = q_loss if total_qhalt is None else total_qhalt + q_loss
        n_ce_steps += 1
        final_carry, final_outputs = carry, outputs
        if not train_mode and bool(carry.halted.all()):
            break

    assert total_ce is not None
    query_loss = total_ce / max(n_ce_steps, 1)
    if total_qhalt is not None:
        query_loss = query_loss + q_halt_weight * (total_qhalt / max(n_ce_steps, 1))
    return query_loss, final_carry, final_outputs, n_ce_steps


def forward_loss_and_outputs(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    act_steps: int,
    train_mode: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], int, Any]:
    model.train(train_mode)
    with torch.device(batch["inputs"].device):
        carry = model.initial_carry(batch)
    total = None
    outputs = None
    steps = 0
    for steps in range(1, act_steps + 1):
        carry, outputs = model(carry, batch)
        loss = ce_loss_from_logits(outputs["logits"], batch["labels"])
        total = loss if total is None else total + loss
        if not train_mode and bool(carry.halted.all()):
            break
    assert total is not None
    assert outputs is not None
    return total / max(steps, 1), outputs, steps, carry


@torch.no_grad()
def q_halt_bce_sum(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
    """Same target and reduction as ``ACTLossHead`` (`losses.py`): sum over batch of BCE(q_halt, seq_is_correct)."""
    if "q_halt_logits" not in outputs:
        return 0.0
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    loss_counts = mask.sum(dim=-1)
    pred = outputs["logits"].argmax(dim=-1)
    is_correct = mask & (pred == labels)
    seq_is_correct = is_correct.sum(dim=-1) == loss_counts
    q = outputs["q_halt_logits"]
    tgt = seq_is_correct.to(q.dtype)
    return float(F.binary_cross_entropy_with_logits(q, tgt, reduction="sum").item())


@torch.no_grad()
def metrics_from_outputs(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    steps: int,
) -> Dict[str, float]:
    pred = outputs["logits"].argmax(dim=-1)
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    correct = (pred == labels) & mask
    token_acc = correct.float().sum().item() / max(mask.float().sum().item(), 1.0)
    # Per-sequence exact match, then mean over batch (shared batches have B>1).
    per_seq_exact = correct.sum(dim=-1) == mask.sum(dim=-1)
    exact = per_seq_exact.float().mean().item()
    return {"accuracy": token_acc, "exact_accuracy": exact, "steps": float(steps)}


@torch.no_grad()
def q_halt_decision_counts(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    carry: Any,
) -> Dict[str, float]:
    """Match ``ACTLossHead`` Q-halt precision/recall raw counts (halted rows only)."""
    if "q_halt_logits" not in outputs:
        return {"q_attempted": 0.0, "q_attempted_correct": 0.0, "q_solvable": 0.0}
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    loss_counts = mask.sum(dim=-1)
    pred = outputs["logits"].argmax(dim=-1)
    is_correct = mask & (pred == labels)
    seq_is_correct = is_correct.sum(dim=-1) == loss_counts
    valid_metrics = carry.halted & (loss_counts > 0)
    q_says_halt = outputs["q_halt_logits"] >= 0
    q_attempted = valid_metrics & q_says_halt
    q_attempted_correct = q_attempted & seq_is_correct
    q_solvable = valid_metrics & seq_is_correct
    return {
        "q_attempted": float(q_attempted.sum().item()),
        "q_attempted_correct": float(q_attempted_correct.sum().item()),
        "q_solvable": float(q_solvable.sum().item()),
    }


def q_precision_recall_f1_from_totals(
    q_attempted: float,
    q_attempted_correct: float,
    q_solvable: float,
    *,
    key_prefix: str,
) -> Dict[str, float]:
    qa = max(q_attempted, 1.0)
    qs = max(q_solvable, 1.0)
    qc = q_attempted_correct
    precision = qc / qa
    recall = qc / qs
    f1_denom = precision + recall
    f1 = 2.0 * precision * recall / f1_denom if f1_denom > 0.0 else 0.0
    if key_prefix == "train":
        return {"train/q_precision": precision, "train/q_recall": recall, "train/q_f1": f1}
    if key_prefix == "train/full_act":
        return {
            "train/full_act_q_precision": precision,
            "train/full_act_q_recall": recall,
            "train/full_act_q_f1": f1,
        }
    raise ValueError(f"unknown key_prefix {key_prefix!r}")


def query_step_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    carry: Any,
    steps: float,
) -> Dict[str, float]:
    m = metrics_from_outputs(outputs, batch, steps=steps)
    m.update(q_halt_decision_counts(outputs, batch, carry))
    m["q_halt_loss"] = q_halt_bce_sum(outputs, batch)
    return m


@torch.no_grad()
def run_val_zero_shot(
    loss_head: nn.Module,
    val_arrays: Dict[str, Any],
    val_rows: Sequence[int],
    max_support: int,
    query_act_steps: int,
    ignore_label_id: int,
) -> Dict[str, float]:
    """Zero-shot val eval — full-ACT inference at the current meta-trained state, no TTT.

    Mimics the eval-time inference protocol so the metric is comparable to test-set numbers.
    Doesn't run per-episode TTT (kept fast — ~10-30s for 32 episodes).

    Returns a dict of `val/*` aggregated metrics plus the count of evaluated episodes.
    """
    import numpy as _np
    model = loss_head.model
    was_training = model.training
    model.eval()
    device = next(loss_head.parameters()).device

    n = 0
    sum_exact = 0.0
    sum_token_acc = 0.0
    sum_query_ce = 0.0
    sum_act_steps = 0.0
    for row in val_rows:
        sm = _np.asarray(val_arrays["support_mask"][row]).astype(bool)
        n_sup_pairs = int(sm.sum())
        if n_sup_pairs < 1:
            continue
        seq_len = int(val_arrays["inputs"][row].shape[0])
        s_in = torch.zeros((max_support, seq_len), dtype=torch.int32, device=device)
        s_out = torch.zeros((max_support, seq_len), dtype=torch.int32, device=device)
        s_mask = torch.zeros((max_support,), dtype=torch.bool, device=device)
        take = min(n_sup_pairs, max_support)
        s_in[:take] = torch.from_numpy(_np.asarray(val_arrays["support_inputs"][row, :take]).astype(_np.int32)).to(device)
        s_out[:take] = torch.from_numpy(_np.asarray(val_arrays["support_outputs"][row, :take]).astype(_np.int32)).to(device)
        s_mask[:take] = True
        labels_np = _np.asarray(val_arrays["labels"][row]).astype(_np.int32)
        labels_t = torch.from_numpy(labels_np).to(device).clone()
        labels_t[labels_t == ignore_label_id] = IGNORE_LABEL_ID
        batch = {
            "inputs": torch.from_numpy(_np.asarray(val_arrays["inputs"][row]).astype(_np.int32)).to(device).unsqueeze(0),
            "labels": labels_t.unsqueeze(0),
            "support_inputs": s_in.unsqueeze(0),
            "support_outputs": s_out.unsqueeze(0),
            "support_mask": s_mask.unsqueeze(0),
            "puzzle_identifiers": torch.tensor([int(val_arrays["puzzle_identifiers"][row])], dtype=torch.int32, device=device),
            "query_indices": torch.zeros((1,), dtype=torch.int32, device=device),
            "query_sources": torch.zeros((1,), dtype=torch.int32, device=device),
        }
        # Run full-ACT inference up to query_act_steps; halt early if all halted (eval semantics).
        with torch.device(device):
            carry = model.initial_carry(batch)
        outputs = None
        steps = 0
        for steps in range(1, max(1, query_act_steps) + 1):
            carry, outputs = model(carry, batch)
            if bool(carry.halted.all()):
                break
        if outputs is None:
            continue
        pred = outputs["logits"].argmax(dim=-1)
        labels = batch["labels"]
        mask = labels != IGNORE_LABEL_ID
        correct = (pred == labels) & mask
        n_supervised = int(mask.sum().item())
        if n_supervised == 0:
            continue
        token_acc = correct.float().sum().item() / n_supervised
        per_seq_exact = (correct.sum(dim=-1) == mask.sum(dim=-1)).float()
        exact = float(per_seq_exact.mean().item())
        ce = F.cross_entropy(
            outputs["logits"].reshape(-1, outputs["logits"].shape[-1]).float(),
            labels.reshape(-1).long(),
            ignore_index=IGNORE_LABEL_ID,
            reduction="mean",
        ).item()
        sum_exact += exact
        sum_token_acc += token_acc
        sum_query_ce += ce
        sum_act_steps += float(steps)
        n += 1
    if was_training:
        model.train(True)
    if n == 0:
        return {"val/episodes": 0.0}
    return {
        "val/episodes": float(n),
        "val/exact": sum_exact / n,
        "val/token_acc": sum_token_acc / n,
        "val/query_ce": sum_query_ce / n,
        "val/act_steps": sum_act_steps / n,
    }


def make_single_episode_batch(
    input_seq: torch.Tensor,
    label_seq: torch.Tensor,
    support_inputs: torch.Tensor,
    support_outputs: torch.Tensor,
    puzzle_identifier: torch.Tensor,
    max_support_examples: int,
) -> Dict[str, torch.Tensor]:
    """Build a batch-size-1 episode directly on GPU without CPU round-trips."""
    seq_len = int(input_seq.shape[0])
    n_support = min(int(support_inputs.shape[0]), max_support_examples)
    s_in = torch.zeros((max_support_examples, seq_len), dtype=support_inputs.dtype, device=support_inputs.device)
    s_out = torch.zeros((max_support_examples, seq_len), dtype=support_outputs.dtype, device=support_outputs.device)
    s_mask = torch.zeros((max_support_examples,), dtype=torch.bool, device=support_inputs.device)
    if n_support:
        s_in[:n_support] = support_inputs[:n_support]
        s_out[:n_support] = support_outputs[:n_support]
        s_mask[:n_support] = True

    return {
        "inputs": input_seq[None],
        "labels": label_seq[None],
        "support_inputs": s_in[None],
        "support_outputs": s_out[None],
        "support_mask": s_mask[None],
        "puzzle_identifiers": puzzle_identifier.reshape(1).to(torch.int32),
        "query_indices": torch.zeros((1,), dtype=torch.int32, device=input_seq.device),
        "query_sources": torch.zeros((1,), dtype=torch.int32, device=input_seq.device),
    }


def _episode_support(batch: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
    return torch.nonzero(batch["support_mask"][idx].to(torch.bool), as_tuple=False).flatten()


def _episode_inner_adapt(
    loss_head: nn.Module,
    fast_params: Sequence[nn.Parameter],
    initial_fast_state: Dict[str, torch.Tensor],
    episode_batch: Dict[str, torch.Tensor],
    episode_idx: int,
    inner_steps: int,
    inner_lr: float,
    adapt_act_steps: int,
    grad_clip: float,
    return_last_step_delta: bool = False,
) -> tuple[Dict[str, torch.Tensor] | None, float, Optional[Dict[str, torch.Tensor]]]:
    """Inner SGD on support for one episode; optional OpenAI-FOML last-step delta.

    When ``return_last_step_delta`` is True, returns the LoRA weight change
    induced by the final inner SGD step only (weights after last step minus
    weights immediately before it), matching ``FOML.train_step`` in OpenAI's
    supervised-reptile codebase.
    """

    if return_last_step_delta and inner_steps < 1:
        raise ValueError("return_last_step_delta requires inner_steps >= 1")

    model = loss_head.model
    valid = _episode_support(episode_batch, episode_idx)
    if len(valid) < 2:
        return None, 0.0, None

    load_adapter_state(loss_head, initial_fast_state)
    max_support = int(model.config.max_support_examples)
    support_inputs = episode_batch["support_inputs"][episode_idx, valid]
    support_outputs = episode_batch["support_outputs"][episode_idx, valid]
    puzzle_identifier = episode_batch["puzzle_identifiers"][episode_idx]

    last_backup: Optional[Dict[str, torch.Tensor]] = None
    for step in range(inner_steps):
        if return_last_step_delta and step == inner_steps - 1:
            last_backup = {
                k: v.detach().clone() for k, v in adapter_state_dict(loss_head, "lora").items()
            }
        target_local_idx = step % len(valid)
        context = [i for i in range(len(valid)) if i != target_local_idx]
        if not context:
            context = list(range(len(valid)))
        adapt_batch = make_single_episode_batch(
            support_inputs[target_local_idx],
            support_outputs[target_local_idx],
            support_inputs[context],
            support_outputs[context],
            puzzle_identifier,
            max_support,
        )
        support_loss, _, _, _ = forward_loss_and_outputs(model, adapt_batch, act_steps=adapt_act_steps, train_mode=True)
        grads = torch.autograd.grad(
            support_loss,
            fast_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        if grad_clip > 0:
            finite_grads = [g for g in grads if g is not None]
            if finite_grads:
                total_norm = torch.sqrt(sum(g.detach().float().pow(2).sum() for g in finite_grads))
                clip_coef = min(1.0, float(grad_clip / (total_norm.item() + 1e-6)))
                grads = tuple(None if g is None else g * clip_coef for g in grads)
        with torch.no_grad():
            for param, grad in zip(fast_params, grads):
                if grad is not None:
                    param.add_(grad, alpha=-inner_lr)

    query_batch = make_single_episode_batch(
        episode_batch["inputs"][episode_idx],
        episode_batch["labels"][episode_idx],
        support_inputs,
        support_outputs,
        puzzle_identifier,
        max_support,
    )
    last_delta: Optional[Dict[str, torch.Tensor]] = None
    if return_last_step_delta:
        assert last_backup is not None
        phi = adapter_state_dict(loss_head, "lora")
        last_delta = {k: phi[k] - last_backup[k] for k in phi}
    return query_batch, float(len(valid)), last_delta


def meta_train_episode_fomaml(
    loss_head: nn.Module,
    fast_params: Sequence[nn.Parameter],
    initial_fast_state: Dict[str, torch.Tensor],
    episode_batch: Dict[str, torch.Tensor],
    episode_idx: int,
    inner_steps: int,
    inner_lr: float,
    adapt_act_steps: int,
    query_act_steps: int,
    grad_clip: float,
    meta_batch_size: int,
) -> Dict[str, float]:
    """FOMAML: inner adapt, query CE backward at adapted LoRA, reset LoRA before outer step."""

    query_batch, support_count, _ = _episode_inner_adapt(
        loss_head,
        fast_params,
        initial_fast_state,
        episode_batch,
        episode_idx,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        adapt_act_steps=adapt_act_steps,
        grad_clip=grad_clip,
        return_last_step_delta=False,
    )
    if query_batch is None:
        return {"skipped": 1.0}

    query_loss, q_carry, q_outputs, n_ce = forward_query_ce_with_act_metrics(
        loss_head,
        query_batch,
        act_steps=query_act_steps,
        train_mode=True,
    )
    (query_loss / meta_batch_size).backward()
    metrics = query_step_metrics(q_outputs, query_batch, q_carry, float(n_ce))
    load_adapter_state(loss_head, initial_fast_state)
    return {
        "skipped": 0.0,
        "query_loss": float(query_loss.detach().item()),
        "accuracy": metrics["accuracy"],
        "exact_accuracy": metrics["exact_accuracy"],
        "steps": metrics["steps"],
        "q_attempted": metrics["q_attempted"],
        "q_attempted_correct": metrics["q_attempted_correct"],
        "q_solvable": metrics["q_solvable"],
        "q_halt_loss": metrics["q_halt_loss"],
        "support_count": support_count,
        "_query_carry": q_carry,
        "_query_outputs": q_outputs,
    }


def meta_train_episode_reptile_step(
    loss_head: nn.Module,
    fast_params: Sequence[nn.Parameter],
    initial_fast_state: Dict[str, torch.Tensor],
    episode_batch: Dict[str, torch.Tensor],
    episode_idx: int,
    inner_steps: int,
    inner_lr: float,
    adapt_act_steps: int,
    query_act_steps: int,
    grad_clip: float,
    light_query: bool = False,
) -> Dict[str, float]:
    """One episode for Reptile: inner adapt, snapshot phi, reset LoRA; query metrics in no_grad."""

    query_batch, support_count, _ = _episode_inner_adapt(
        loss_head,
        fast_params,
        initial_fast_state,
        episode_batch,
        episode_idx,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        adapt_act_steps=adapt_act_steps,
        grad_clip=grad_clip,
        return_last_step_delta=False,
    )
    if query_batch is None:
        return {"skipped": 1.0}

    phi = adapter_state_dict(loss_head, "lora")
    delta = {k: phi[k] - initial_fast_state[k] for k in phi}
    model = loss_head.model
    if light_query:
        with torch.no_grad():
            query_loss_t, q_outputs, n_steps, q_carry = forward_loss_and_outputs(
                model, query_batch, act_steps=query_act_steps, train_mode=False
            )
        metrics = query_step_metrics(q_outputs, query_batch, q_carry, float(n_steps))
        load_adapter_state(loss_head, initial_fast_state)
        return {
            "skipped": 0.0,
            "delta": delta,
            "query_loss": float(query_loss_t.item()),
            "accuracy": metrics["accuracy"],
            "exact_accuracy": metrics["exact_accuracy"],
            "steps": metrics["steps"],
            "q_attempted": metrics["q_attempted"],
            "q_attempted_correct": metrics["q_attempted_correct"],
            "q_solvable": metrics["q_solvable"],
            "q_halt_loss": metrics["q_halt_loss"],
            "support_count": support_count,
            "_query_carry": q_carry,
            "_query_outputs": q_outputs,
        }

    with torch.no_grad():
        query_loss, q_carry, q_outputs, n_ce = forward_query_ce_with_act_metrics(
            loss_head,
            query_batch,
            act_steps=query_act_steps,
            train_mode=False,
        )
    metrics = query_step_metrics(q_outputs, query_batch, q_carry, float(n_ce))
    load_adapter_state(loss_head, initial_fast_state)
    return {
        "skipped": 0.0,
        "delta": delta,
        "query_loss": float(query_loss.item()),
        "accuracy": metrics["accuracy"],
        "exact_accuracy": metrics["exact_accuracy"],
        "steps": metrics["steps"],
        "q_attempted": metrics["q_attempted"],
        "q_attempted_correct": metrics["q_attempted_correct"],
        "q_solvable": metrics["q_solvable"],
        "q_halt_loss": metrics["q_halt_loss"],
        "support_count": support_count,
        "_query_carry": q_carry,
        "_query_outputs": q_outputs,
    }


def meta_train_episode_openai_foml_step(
    loss_head: nn.Module,
    fast_params: Sequence[nn.Parameter],
    initial_fast_state: Dict[str, torch.Tensor],
    episode_batch: Dict[str, torch.Tensor],
    episode_idx: int,
    inner_steps: int,
    inner_lr: float,
    adapt_act_steps: int,
    query_act_steps: int,
    grad_clip: float,
    light_query: bool = True,
) -> Dict[str, float]:
    """OpenAI FOML: average last inner-step LoRA deltas across episodes; outer add (not interpolate)."""

    query_batch, support_count, last_delta = _episode_inner_adapt(
        loss_head,
        fast_params,
        initial_fast_state,
        episode_batch,
        episode_idx,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        adapt_act_steps=adapt_act_steps,
        grad_clip=grad_clip,
        return_last_step_delta=True,
    )
    if query_batch is None:
        return {"skipped": 1.0}
    assert last_delta is not None

    model = loss_head.model
    if light_query:
        with torch.no_grad():
            query_loss_t, q_outputs, n_steps, q_carry = forward_loss_and_outputs(
                model, query_batch, act_steps=query_act_steps, train_mode=False
            )
        metrics = query_step_metrics(q_outputs, query_batch, q_carry, float(n_steps))
        load_adapter_state(loss_head, initial_fast_state)
        return {
            "skipped": 0.0,
            "last_delta": last_delta,
            "query_loss": float(query_loss_t.item()),
            "accuracy": metrics["accuracy"],
            "exact_accuracy": metrics["exact_accuracy"],
            "steps": metrics["steps"],
            "q_attempted": metrics["q_attempted"],
            "q_attempted_correct": metrics["q_attempted_correct"],
            "q_solvable": metrics["q_solvable"],
            "q_halt_loss": metrics["q_halt_loss"],
            "support_count": support_count,
            "_query_carry": q_carry,
            "_query_outputs": q_outputs,
        }

    query_loss, q_carry, q_outputs, n_ce = forward_query_ce_with_act_metrics(
        loss_head,
        query_batch,
        act_steps=query_act_steps,
        train_mode=False,
    )
    metrics = query_step_metrics(q_outputs, query_batch, q_carry, float(n_ce))
    load_adapter_state(loss_head, initial_fast_state)
    return {
        "skipped": 0.0,
        "last_delta": last_delta,
        "query_loss": float(query_loss.item()),
        "accuracy": metrics["accuracy"],
        "exact_accuracy": metrics["exact_accuracy"],
        "steps": metrics["steps"],
        "q_attempted": metrics["q_attempted"],
        "q_attempted_correct": metrics["q_attempted_correct"],
        "q_solvable": metrics["q_solvable"],
        "q_halt_loss": metrics["q_halt_loss"],
        "support_count": support_count,
        "_query_carry": q_carry,
        "_query_outputs": q_outputs,
    }


def _shared_batch_inner_adapt(
    loss_head: nn.Module,
    model: nn.Module,
    fast_params: Sequence[nn.Parameter],
    initial_fast_state: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    inner_steps: int,
    inner_lr: float,
    adapt_act_steps: int,
    grad_clip: float,
    return_last_step_delta: bool = False,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, float, Optional[Dict[str, torch.Tensor]]]:
    """Shared inner loop: restore LoRA to meta init, adapt on support, return active slice and skip count.

    When ``return_last_step_delta`` is True, the last return value is the LoRA delta from the
    final inner SGD step only (OpenAI supervised-reptile ``FOML`` semantics).
    """

    if return_last_step_delta and inner_steps < 1:
        raise ValueError("return_last_step_delta requires inner_steps >= 1")

    support_mask = batch["support_mask"].to(torch.bool)
    valid_rows = torch.nonzero(support_mask.sum(dim=1) >= 2, as_tuple=False).flatten()
    skipped = int(batch["inputs"].shape[0] - valid_rows.shape[0])
    if valid_rows.numel() == 0:
        return {}, valid_rows, float(skipped), None

    load_adapter_state(loss_head, initial_fast_state)
    active = {k: v[valid_rows] for k, v in batch.items()}
    row_idx = torch.arange(valid_rows.shape[0], device=batch["inputs"].device)

    last_backup: Optional[Dict[str, torch.Tensor]] = None
    for step in range(inner_steps):
        if return_last_step_delta and step == inner_steps - 1:
            last_backup = {
                k: v.detach().clone() for k, v in adapter_state_dict(loss_head, "lora").items()
            }
        counts = active["support_mask"].sum(dim=1).clamp_min(1).to(torch.long)
        target_idx = (torch.full_like(counts, step) % counts).to(torch.long)

        adapt_support_mask = active["support_mask"].clone()
        adapt_support_mask[row_idx, target_idx] = False
        adapt_batch = {
            "inputs": active["support_inputs"][row_idx, target_idx],
            "labels": active["support_outputs"][row_idx, target_idx],
            "support_inputs": active["support_inputs"],
            "support_outputs": active["support_outputs"],
            "support_mask": adapt_support_mask,
            "puzzle_identifiers": active["puzzle_identifiers"],
            "query_indices": active["query_indices"],
            "query_sources": active["query_sources"],
        }
        support_loss, _, _, _ = forward_loss_and_outputs(model, adapt_batch, act_steps=adapt_act_steps, train_mode=True)
        grads = torch.autograd.grad(
            support_loss,
            fast_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        if grad_clip > 0:
            finite_grads = [g for g in grads if g is not None]
            if finite_grads:
                total_norm = torch.sqrt(sum(g.detach().float().pow(2).sum() for g in finite_grads))
                clip_coef = min(1.0, float(grad_clip / (total_norm.item() + 1e-6)))
                grads = tuple(None if g is None else g * clip_coef for g in grads)
        with torch.no_grad():
            for param, grad in zip(fast_params, grads):
                if grad is not None:
                    param.add_(grad, alpha=-inner_lr)

    last_delta: Optional[Dict[str, torch.Tensor]] = None
    if return_last_step_delta:
        assert last_backup is not None
        phi = adapter_state_dict(loss_head, "lora")
        last_delta = {k: phi[k] - last_backup[k] for k in phi}
    return active, valid_rows, float(skipped), last_delta


def meta_train_shared_batch_fomaml(
    loss_head: nn.Module,
    fast_params: Sequence[nn.Parameter],
    initial_fast_state: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    inner_steps: int,
    inner_lr: float,
    adapt_act_steps: int,
    query_act_steps: int,
    grad_clip: float,
) -> Dict[str, float]:
    """FOMAML: shared inner adapt, query backward, reset LoRA before ``optim.step``."""

    model = loss_head.model
    active, valid_rows, skipped, _ = _shared_batch_inner_adapt(
        loss_head,
        model,
        fast_params,
        initial_fast_state,
        batch,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        adapt_act_steps=adapt_act_steps,
        grad_clip=grad_clip,
        return_last_step_delta=False,
    )
    if valid_rows.numel() == 0:
        return {"skipped": skipped}

    query_loss, q_carry, q_outputs, n_ce = forward_query_ce_with_act_metrics(
        loss_head,
        active,
        act_steps=query_act_steps,
        train_mode=True,
    )
    query_loss, contrastive_loss_value = maybe_add_contrastive_aux(query_loss, q_outputs, active)
    query_loss.backward()
    metrics = query_step_metrics(q_outputs, active, q_carry, float(n_ce))
    load_adapter_state(loss_head, initial_fast_state)

    denom = float(valid_rows.shape[0])
    out: Dict[str, Any] = {
        "skipped": skipped,
        "query_loss": float(query_loss.detach().item()) * denom,
        "accuracy": metrics["accuracy"] * denom,
        "exact_accuracy": metrics["exact_accuracy"] * denom,
        "steps": metrics["steps"] * denom,
        "q_attempted": metrics["q_attempted"],
        "q_attempted_correct": metrics["q_attempted_correct"],
        "q_solvable": metrics["q_solvable"],
        "q_halt_loss": metrics["q_halt_loss"],
        "support_count": float(active["support_mask"].sum(dim=1).float().mean().item()) * denom,
        "_query_carry": q_carry,
        "_query_outputs": q_outputs,
    }
    if contrastive_loss_value is not None:
        out["contrastive_loss"] = contrastive_loss_value * denom
    return out


def meta_train_shared_batch_reptile(
    loss_head: nn.Module,
    fast_params: Sequence[nn.Parameter],
    initial_fast_state: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    inner_steps: int,
    inner_lr: float,
    adapt_act_steps: int,
    query_act_steps: int,
    grad_clip: float,
    meta_outer_lr: float,
    train_backbone: bool,
    light_query: bool = False,
) -> Dict[str, float]:
    """Reptile: inner SGD on support; outer step interpolates LoRA toward adapted ``phi``."""

    model = loss_head.model
    active, valid_rows, skipped, _ = _shared_batch_inner_adapt(
        loss_head,
        model,
        fast_params,
        initial_fast_state,
        batch,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        adapt_act_steps=adapt_act_steps,
        grad_clip=grad_clip,
        return_last_step_delta=False,
    )
    if valid_rows.numel() == 0:
        return {"skipped": skipped}

    phi_state = adapter_state_dict(loss_head, "lora")
    load_adapter_state(loss_head, initial_fast_state)
    eps = float(meta_outer_lr)
    own = loss_head.state_dict()
    for key, phi in phi_state.items():
        t0 = initial_fast_state[key]
        own[key].copy_(t0 + eps * (phi - t0))

    q_carry = None
    contrastive_loss_value: Optional[float] = None
    if train_backbone:
        if light_query:
            query_loss, q_outputs, n_ce, q_carry = forward_loss_and_outputs(
                model, active, act_steps=query_act_steps, train_mode=True
            )
            query_loss, contrastive_loss_value = maybe_add_contrastive_aux(query_loss, q_outputs, active)
            query_loss.backward()
            for p in fast_params:
                p.grad = None
            metrics = query_step_metrics(q_outputs, active, q_carry, float(n_ce))
        else:
            query_loss, q_carry, q_outputs, n_ce = forward_query_ce_with_act_metrics(
                loss_head,
                active,
                act_steps=query_act_steps,
                train_mode=True,
            )
            query_loss, contrastive_loss_value = maybe_add_contrastive_aux(query_loss, q_outputs, active)
            query_loss.backward()
            for p in fast_params:
                p.grad = None
            metrics = query_step_metrics(q_outputs, active, q_carry, float(n_ce))
    else:
        if light_query:
            with torch.no_grad():
                query_loss, q_outputs, n_ce, q_carry = forward_loss_and_outputs(
                    model, active, act_steps=query_act_steps, train_mode=False
                )
            metrics = query_step_metrics(q_outputs, active, q_carry, float(n_ce))
        else:
            with torch.no_grad():
                query_loss, q_carry, q_outputs, n_ce = forward_query_ce_with_act_metrics(
                    loss_head,
                    active,
                    act_steps=query_act_steps,
                    train_mode=False,
                )
            metrics = query_step_metrics(q_outputs, active, q_carry, float(n_ce))

    denom = float(valid_rows.shape[0])
    qv = float(query_loss.detach().item()) if train_backbone else float(query_loss.item())
    out: Dict[str, Any] = {
        "skipped": skipped,
        "query_loss": qv * denom,
        "accuracy": metrics["accuracy"] * denom,
        "exact_accuracy": metrics["exact_accuracy"] * denom,
        "steps": metrics["steps"] * denom,
        "q_attempted": metrics["q_attempted"],
        "q_attempted_correct": metrics["q_attempted_correct"],
        "q_solvable": metrics["q_solvable"],
        "q_halt_loss": metrics["q_halt_loss"],
        "support_count": float(active["support_mask"].sum(dim=1).float().mean().item()) * denom,
        "_query_outputs": q_outputs,
    }
    if q_carry is not None:
        out["_query_carry"] = q_carry
    if contrastive_loss_value is not None:
        out["contrastive_loss"] = contrastive_loss_value * denom
    return out


def meta_train_shared_batch_openai_foml(
    loss_head: nn.Module,
    fast_params: Sequence[nn.Parameter],
    initial_fast_state: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    inner_steps: int,
    inner_lr: float,
    adapt_act_steps: int,
    query_act_steps: int,
    grad_clip: float,
    meta_outer_lr: float,
    train_backbone: bool,
    light_query: bool = True,
) -> Dict[str, float]:
    """OpenAI FOML on a shared inner loop: ``theta <- theta + eps * last_inner_step_delta``."""

    model = loss_head.model
    active, valid_rows, skipped, last_delta = _shared_batch_inner_adapt(
        loss_head,
        model,
        fast_params,
        initial_fast_state,
        batch,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        adapt_act_steps=adapt_act_steps,
        grad_clip=grad_clip,
        return_last_step_delta=True,
    )
    if valid_rows.numel() == 0:
        return {"skipped": skipped}
    assert last_delta is not None

    load_adapter_state(loss_head, initial_fast_state)
    eps = float(meta_outer_lr)
    own = loss_head.state_dict()
    for k, d in last_delta.items():
        own[k].copy_(initial_fast_state[k] + eps * d)

    q_carry = None
    contrastive_loss_value: Optional[float] = None
    if train_backbone:
        if light_query:
            query_loss, q_outputs, n_ce, q_carry = forward_loss_and_outputs(
                model, active, act_steps=query_act_steps, train_mode=True
            )
            query_loss, contrastive_loss_value = maybe_add_contrastive_aux(query_loss, q_outputs, active)
            query_loss.backward()
            for p in fast_params:
                p.grad = None
            metrics = query_step_metrics(q_outputs, active, q_carry, float(n_ce))
        else:
            query_loss, q_carry, q_outputs, n_ce = forward_query_ce_with_act_metrics(
                loss_head,
                active,
                act_steps=query_act_steps,
                train_mode=True,
            )
            query_loss, contrastive_loss_value = maybe_add_contrastive_aux(query_loss, q_outputs, active)
            query_loss.backward()
            for p in fast_params:
                p.grad = None
            metrics = query_step_metrics(q_outputs, active, q_carry, float(n_ce))
    else:
        if light_query:
            with torch.no_grad():
                query_loss, q_outputs, n_ce, q_carry = forward_loss_and_outputs(
                    model, active, act_steps=query_act_steps, train_mode=False
                )
            metrics = query_step_metrics(q_outputs, active, q_carry, float(n_ce))
        else:
            with torch.no_grad():
                query_loss, q_carry, q_outputs, n_ce = forward_query_ce_with_act_metrics(
                    loss_head,
                    active,
                    act_steps=query_act_steps,
                    train_mode=False,
                )
            metrics = query_step_metrics(q_outputs, active, q_carry, float(n_ce))

    denom = float(valid_rows.shape[0])
    qv = float(query_loss.detach().item()) if train_backbone else float(query_loss.item())
    out: Dict[str, Any] = {
        "skipped": skipped,
        "query_loss": qv * denom,
        "accuracy": metrics["accuracy"] * denom,
        "exact_accuracy": metrics["exact_accuracy"] * denom,
        "steps": metrics["steps"] * denom,
        "q_attempted": metrics["q_attempted"],
        "q_attempted_correct": metrics["q_attempted_correct"],
        "q_solvable": metrics["q_solvable"],
        "q_halt_loss": metrics["q_halt_loss"],
        "support_count": float(active["support_mask"].sum(dim=1).float().mean().item()) * denom,
        "_query_outputs": q_outputs,
    }
    if q_carry is not None:
        out["_query_carry"] = q_carry
    if contrastive_loss_value is not None:
        out["contrastive_loss"] = contrastive_loss_value * denom
    return out


def full_act_probe_metrics(
    loss_head: nn.Module,
    fast_params: Sequence[nn.Parameter],
    initial_fast_state: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    inner_steps: int,
    inner_lr: float,
    adapt_act_steps: int,
    act_steps: int,
    grad_clip: float,
) -> Dict[str, float]:
    """Measure full-ACT query metrics after the same shared fast adaptation."""
    model = loss_head.model
    support_mask = batch["support_mask"].to(torch.bool)
    valid_rows = torch.nonzero(support_mask.sum(dim=1) >= 2, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        return {}

    load_adapter_state(loss_head, initial_fast_state)
    active = {k: v[valid_rows] for k, v in batch.items()}
    row_idx = torch.arange(valid_rows.shape[0], device=batch["inputs"].device)

    for step in range(inner_steps):
        counts = active["support_mask"].sum(dim=1).clamp_min(1).to(torch.long)
        target_idx = (torch.full_like(counts, step) % counts).to(torch.long)
        adapt_support_mask = active["support_mask"].clone()
        adapt_support_mask[row_idx, target_idx] = False
        adapt_batch = {
            "inputs": active["support_inputs"][row_idx, target_idx],
            "labels": active["support_outputs"][row_idx, target_idx],
            "support_inputs": active["support_inputs"],
            "support_outputs": active["support_outputs"],
            "support_mask": adapt_support_mask,
            "puzzle_identifiers": active["puzzle_identifiers"],
            "query_indices": active["query_indices"],
            "query_sources": active["query_sources"],
        }
        support_loss, _, _, _ = forward_loss_and_outputs(model, adapt_batch, act_steps=adapt_act_steps, train_mode=True)
        grads = torch.autograd.grad(
            support_loss,
            fast_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        if grad_clip > 0:
            finite_grads = [g for g in grads if g is not None]
            if finite_grads:
                total_norm = torch.sqrt(sum(g.detach().float().pow(2).sum() for g in finite_grads))
                clip_coef = min(1.0, float(grad_clip / (total_norm.item() + 1e-6)))
                grads = tuple(None if g is None else g * clip_coef for g in grads)
        with torch.no_grad():
            for param, grad in zip(fast_params, grads):
                if grad is not None:
                    param.add_(grad, alpha=-inner_lr)

    model.eval()
    with torch.no_grad():
        with torch.device(active["inputs"].device):
            carry = model.initial_carry(active)
        outputs = None
        steps = 0
        for steps in range(1, act_steps + 1):
            carry, outputs = model(carry, active)
            if bool(carry.halted.all()):
                break
        assert outputs is not None
        metrics = metrics_from_outputs(outputs, active, steps=float(steps))
        q_ct = q_halt_decision_counts(outputs, active, carry)
        q_extra = q_precision_recall_f1_from_totals(
            q_ct["q_attempted"],
            q_ct["q_attempted_correct"],
            q_ct["q_solvable"],
            key_prefix="train/full_act",
        )
    load_adapter_state(loss_head, initial_fast_state)
    out = {
        "train/full_act_accuracy": metrics["accuracy"],
        "train/full_act_exact_accuracy": metrics["exact_accuracy"],
        "train/full_act_steps": metrics["steps"],
    }
    out.update(q_extra)
    return out


def load_config(hydra_config: DictConfig) -> PretrainConfig:
    config = PretrainConfig(**hydra_config)  # type: ignore
    if config.project_name is None:
        config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-SupervisedReptile"
    if config.run_name is None:
        config.run_name = f"support-ttt-supervised-reptile {coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
    return config


def grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.sqrt(sum(g.pow(2).sum() for g in grads)).item())


def enable_wandb_debug_logging(model: nn.Module) -> None:
    """Match pretrain.py's debug=True behavior while tolerating None grads."""
    try:
        import wandb.integration.torch.wandb_torch as _wt
        from wandb.integration.torch.wandb_torch import log_track_update as _ltu

        def _safe_hook_variable_gradient_stats(self, var, name, log_track):
            if not isinstance(var, torch.autograd.Variable):
                cls = type(var)
                raise TypeError(f"Expected torch.Variable, not {cls.__module__}.{cls.__name__}")
            handle = self._hook_handles.get(name)
            if handle is not None and self._torch_hook_handle_is_valid(handle):
                raise ValueError(f'A hook has already been set under name "{name}"')

            def _callback(grad, log_track):
                if grad is None or not _ltu(log_track):
                    return
                self.log_tensor_stats(grad.data, name)

            handle = var.register_hook(lambda grad: _callback(grad, log_track))
            self._hook_handles[name] = handle
            return handle

        _wt.TorchHistory._hook_variable_gradient_stats = _safe_hook_variable_gradient_stats
        wandb.watch(model, log="all", log_freq=500, log_graph=False)
    except Exception as e:
        print(f"[wandb.watch] skipped due to error: {e}")


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    # LoRA adapters are inserted into named submodules after construction.
    os.environ.setdefault("DISABLE_COMPILE", "1")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    config = load_config(hydra_config)
    torch.manual_seed(config.seed)

    meta_cfg = getattr(config.arch, "meta", None) or getattr(config.arch, "maml", None)
    if meta_cfg is None:
        raise ValueError("Define arch.meta on the architecture config (legacy arch.maml is accepted).")
    inner_steps = int(getattr(meta_cfg, "inner_steps", 1))
    inner_lr = float(getattr(meta_cfg, "inner_lr", 1e-4))
    adapt_act_steps = int(getattr(meta_cfg, "adapt_act_steps", 1))
    query_act_steps = int(getattr(meta_cfg, "query_act_steps", 1))
    lora_rank = int(getattr(meta_cfg, "lora_rank", 8))
    grad_clip = float(getattr(meta_cfg, "grad_clip", 1.0))
    max_steps = int(getattr(meta_cfg, "max_steps", 100000))
    train_backbone = bool(getattr(meta_cfg, "train_backbone", True))
    adaptation_mode = str(getattr(meta_cfg, "adaptation_mode", "shared_batch"))
    full_act_probe_interval = int(getattr(meta_cfg, "full_act_probe_interval", 100))
    full_act_probe_steps = int(getattr(meta_cfg, "full_act_probe_steps", 16))
    meta_algorithm = str(getattr(meta_cfg, "meta_algorithm", "openai_reptile")).lower()
    meta_outer_lr = float(
        getattr(meta_cfg, "meta_outer_lr", getattr(meta_cfg, "reptile_outer_lr", 0.1))
    )

    if meta_algorithm not in ("fomaml", "reptile", "openai_reptile", "openai_foml"):
        raise ValueError(
            "arch.meta.meta_algorithm must be one of "
            "'fomaml', 'reptile', 'openai_reptile', 'openai_foml', "
            f"got {meta_algorithm!r}"
        )
    if meta_algorithm in ("reptile", "openai_reptile") and adaptation_mode == "per_episode" and train_backbone:
        raise ValueError(
            "Reptile / openai_reptile with adaptation_mode='per_episode' does not support train_backbone=True; "
            "use shared_batch or set train_backbone=False."
        )

    meta_batch_size = config.global_batch_size // config.gradient_accumulation_steps
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=config.epochs,
        global_batch_size=meta_batch_size,
        rank=0,
        world_size=1,
    )

    loss_head, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=0, world_size=1)
    loss_head.train()
    fast_params = install_lora(loss_head, rank=lora_rank, include_h=True)

    # Load pretrained backbone if specified. Frozen-backbone meta-learning needs a non-random
    # init or there are no useful features to adapt. Shape-mismatched LoRA keys (different rank)
    # are skipped so install_lora's fresh init stands.
    # Uses `pretrained_backbone` (not `load_checkpoint`) because pretrain.py:create_model auto-calls
    # its own broken-for-support-TTT load_checkpoint() when load_checkpoint is set.
    if getattr(config, "pretrained_backbone", None):
        print(f"[support-ttt] loading pretrained checkpoint: {config.pretrained_backbone}")
        ckpt_sd = torch.load(config.pretrained_backbone, map_location="cpu")
        if any(k.startswith("_orig_mod.") for k in ckpt_sd):
            ckpt_sd = {
                (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                for k, v in ckpt_sd.items()
            }
        own_sd = loss_head.state_dict()
        reset_lora = bool(getattr(config, "reset_lora_on_load", True))
        def _is_lora_key(k: str) -> bool:
            return (".lora" in k) or ("lora_" in k) or ("block_loras" in k)
        accepted: Dict[str, torch.Tensor] = {}
        skipped_shape = 0
        skipped_lora = 0
        for k, v in ckpt_sd.items():
            if reset_lora and _is_lora_key(k):
                skipped_lora += 1
                continue
            if k in own_sd and own_sd[k].shape == v.shape:
                accepted[k] = v
            elif k in own_sd:
                skipped_shape += 1
        loss_head.load_state_dict(accepted, strict=False)
        print(
            f"  loaded {len(accepted)}/{len(ckpt_sd)} ckpt tensors "
            f"(skipped {skipped_shape} shape-mismatched, "
            f"{skipped_lora} LoRA keys{' [reset]' if reset_lora else ''}, "
            f"{len(ckpt_sd) - len(accepted) - skipped_shape - skipped_lora} not-in-model)"
        )

    if train_backbone:
        for param in loss_head.parameters():
            param.requires_grad_(True)

    # Tier 2 #6: split weight_decay between backbone (existing groups) and LoRA (fast_params).
    # Recommended: backbone_weight_decay=0.5 (combat memorization), lora_weight_decay=0.0
    # (LoRA should be free to grow away from zero-init).
    backbone_wd = getattr(config, "backbone_weight_decay", None)
    lora_wd = getattr(config, "lora_weight_decay", None)
    if backbone_wd is None:
        backbone_wd = float(config.weight_decay)
    if lora_wd is None:
        lora_wd = float(config.weight_decay)
    for optim in optimizers:
        for group in optim.param_groups:
            group["weight_decay"] = float(backbone_wd)
        optim.add_param_group({"params": list(fast_params), "lr": config.lr, "weight_decay": float(lora_wd)})

    # Tier 2 #7 wiring: read q_halt_weight once and stash module-level so all
    # forward_query_ce_with_act_metrics call sites pick it up.
    global _Q_HALT_WEIGHT_DEFAULT
    _Q_HALT_WEIGHT_DEFAULT = float(getattr(config, "q_halt_weight", 0.0) or 0.0)
    if _Q_HALT_WEIGHT_DEFAULT > 0.0:
        print(f"[support-ttt] q_halt_weight={_Q_HALT_WEIGHT_DEFAULT} added to query loss")
    # A8 Stage 1: contrastive aux loss wiring.
    global _CONTRASTIVE_AUX_WEIGHT, _CONTRASTIVE_TEMPERATURE
    _CONTRASTIVE_AUX_WEIGHT = float(getattr(config, "contrastive_aux_weight", 0.0) or 0.0)
    _CONTRASTIVE_TEMPERATURE = float(getattr(config, "contrastive_temperature", 0.1) or 0.1)
    if _CONTRASTIVE_AUX_WEIGHT > 0.0:
        print(f"[support-ttt] A8 contrastive aux λ={_CONTRASTIVE_AUX_WEIGHT} τ={_CONTRASTIVE_TEMPERATURE}")
    print(f"[support-ttt] weight_decay backbone={backbone_wd} lora={lora_wd}")

    # In-training val eval setup: load `meta_val/` arrays from a separate dataset path if
    # configured. This is the metric that mimics ARC test-eval behaviour during training.
    val_interval = int(getattr(config, "val_interval", 0) or 0)
    val_episodes = int(getattr(config, "val_episodes", 0) or 0)
    val_data_path = getattr(config, "meta_val_data_path", None)
    val_arrays: Optional[Dict[str, Any]] = None
    val_row_pool: Optional[List[int]] = None
    # FF1 N-shot meta-val (paired with the existing zero-shot for delta).
    meta_val_n_shot = bool(getattr(config, "meta_val_n_shot", False))
    meta_val_evaluator = None  # research.lib.MetaValEvaluator | None
    meta_val_split = None      # research.lib.MetaValSplit | None
    meta_val_difficulty_path = getattr(config, "meta_val_difficulty_json", None)
    meta_val_episodes = int(getattr(config, "meta_val_episodes", 0) or 0) or val_episodes
    meta_val_include_cost = bool(getattr(config, "meta_val_include_cost", True))
    if val_interval > 0 and val_episodes > 0 and val_data_path:
        import numpy as _np
        vroot = os.path.join(val_data_path, "meta_val")
        if not os.path.isdir(vroot):
            print(f"[support-ttt] WARNING: meta_val dir missing at {vroot}; val eval disabled")
        else:
            keys = (
                "inputs", "labels", "support_inputs", "support_outputs", "support_mask",
                "puzzle_identifiers", "query_indices", "query_sources",
            )
            val_arrays = {}
            for k in keys:
                val_arrays[k] = _np.load(os.path.join(vroot, f"all__{k}.npy"), mmap_mode="r")
            ti_path = os.path.join(vroot, "all__task_identifiers.npy")
            if os.path.isfile(ti_path):
                val_arrays["task_identifiers"] = _np.load(ti_path, mmap_mode="r")
            # Pre-pick eligible rows (support_count >= 3 mimicking eval default).
            sm = val_arrays["support_mask"]
            sc = sm.sum(axis=1)
            eligible = _np.flatnonzero(sc >= 3)
            print(f"[support-ttt] val arrays loaded: {sm.shape[0]} rows total, {len(eligible)} eligible (>=3 support)")
            rng = _np.random.default_rng(0)
            rng.shuffle(eligible)
            val_row_pool = [int(r) for r in eligible[: max(val_episodes * 4, val_episodes)]]

            if meta_val_n_shot:
                # FF1 setup. Reuses the same arrays + eligible pool as the zero-shot val.
                from research.lib import EpisodeArgs as _EpisodeArgs
                from research.lib import MetaValEvaluator as _MetaValEvaluator
                from research.lib import MetaValSplit as _MetaValSplit
                difficulty_buckets: Dict[int, str] = {}
                if meta_val_difficulty_path and os.path.isfile(meta_val_difficulty_path):
                    with open(meta_val_difficulty_path) as f:
                        _diff = json.load(f)
                    difficulty_buckets = {int(k): v["bucket"] for k, v in _diff.get("by_task", {}).items()}
                    print(f"[support-ttt] FF1 difficulty buckets loaded from {meta_val_difficulty_path}: "
                          f"{sum(1 for b in difficulty_buckets.values() if b=='easy')}E/"
                          f"{sum(1 for b in difficulty_buckets.values() if b=='medium')}M/"
                          f"{sum(1 for b in difficulty_buckets.values() if b=='hard')}H")
                else:
                    print("[support-ttt] FF1: no difficulty JSON; per-bucket stratification disabled")
                if val_arrays.get("task_identifiers") is None:
                    print("[support-ttt] WARNING: meta_val has no task_identifiers; FF1 stratification + per-task aggregation disabled. "
                          "Falling back to N-shot mean only.")
                    meta_val_n_shot = False
                else:
                    meta_val_split = _MetaValSplit(
                        arrays=val_arrays,
                        eligible_rows=_np.asarray(eligible, dtype=_np.int64),
                        task_ids_by_row=_np.asarray(val_arrays["task_identifiers"], dtype=_np.int64),
                        difficulty_buckets=difficulty_buckets,
                    )
                    # Load ARC puzzle identifiers so adapt_one_episode can compute exact_arc
                    # (the ARC-cropped grid hash match — what mimics submission scoring).
                    # Without this, meta_val/pass_arc@1 and zero_shot/pass_arc@1 are all NaN.
                    from research.lib._eval_bridge import load_identifier_list as _load_ids
                    _identifiers = _load_ids(Path(val_data_path), "meta_val")
                    if not _identifiers:
                        print(f"[support-ttt] FF1 WARNING: no identifiers.json found for meta_val at {val_data_path}; "
                              f"meta_val/pass_arc@1 will stay NaN. Verify identifiers.json exists in dataset root.")
                    else:
                        print(f"[support-ttt] FF1 loaded {len(_identifiers)} ARC identifiers for cropped-grid scoring")
                    ep_args = _EpisodeArgs(
                        adapt_act_steps=adapt_act_steps,
                        eval_act_steps=adapt_act_steps,
                        query_act_steps=query_act_steps,
                        max_folds=1,
                        adapt_params="lora",
                        ignore_label_id=int(getattr(train_metadata, "ignore_label_id", 0) or 0),
                        identifiers=_identifiers,
                    )
                    meta_val_evaluator = _MetaValEvaluator(meta_val_split, ep_args)
                    print(f"[support-ttt] FF1 meta-val enabled: episodes={meta_val_episodes} include_cost={meta_val_include_cost}")
    print(f"[support-ttt] val_interval={val_interval} val_episodes={val_episodes} val_pool={len(val_row_pool) if val_row_pool else 0}")

    if config.checkpoint_path is not None:
        os.makedirs(config.checkpoint_path, exist_ok=True)

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "arm-arc-agi"),
        entity=os.environ.get("WANDB_ENTITY", None),
        name=config.run_name,
        config=OmegaConf.to_container(hydra_config, resolve=True),
    )
    if config.debug:
        enable_wandb_debug_logging(loss_head)

    train_logger = TrainLogger(
        loss_head,
        config,
        log_interval_medium=config.log_interval,
        log_interval_heavy=config.log_interval * 10,
    )

    train_state = TrainState(
        model=loss_head,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        step=0,
        total_steps=max_steps,
    )
    all_params = list(loss_head.parameters())
    num_params = sum(p.numel() for p in all_params)
    num_trainable_params = sum(p.numel() for p in all_params if p.requires_grad)
    num_fast_params = sum(p.numel() for p in fast_params)
    wandb.log(
        {
            "num_params": num_params,
            "num_trainable_params": num_trainable_params,
            "num_fast_params": num_fast_params,
        },
        step=train_state.step,
    )

    for _, batch_cpu, _ in train_loader:
        train_state.step += 1
        if train_state.step > max_steps:
            break

        train_logger.step_start()
        batch = {k: v.cuda(non_blocking=True) for k, v in batch_cpu.items()}
        initial_fast_state = adapter_state_dict(loss_head, "lora")
        for optim in optimizers:
            optim.zero_grad(set_to_none=True)

        carry_log: Any = None
        outputs_log: Optional[Dict[str, torch.Tensor]] = None

        totals: Dict[str, float] = {
            "query_loss": 0.0,
            "accuracy": 0.0,
            "exact_accuracy": 0.0,
            "steps": 0.0,
            "support_count": 0.0,
            "skipped": 0.0,
            "q_attempted": 0.0,
            "q_attempted_correct": 0.0,
            "q_solvable": 0.0,
            "q_halt_loss": 0.0,
        }

        def ingest(step_metrics: Dict[str, Any]) -> None:
            nonlocal carry_log, outputs_log
            c = step_metrics.pop("_query_carry", None)
            o = step_metrics.pop("_query_outputs", None)
            if c is not None:
                carry_log, outputs_log = c, o
            for k, v in step_metrics.items():
                if str(k).startswith("_"):
                    continue
                if isinstance(v, dict):
                    continue
                totals[k] = totals.get(k, 0.0) + float(v)

        if adaptation_mode == "shared_batch":
            if meta_algorithm == "fomaml":
                metrics = meta_train_shared_batch_fomaml(
                    loss_head,
                    fast_params,
                    initial_fast_state,
                    batch,
                    inner_steps=inner_steps,
                    inner_lr=inner_lr,
                    adapt_act_steps=adapt_act_steps,
                    query_act_steps=query_act_steps,
                    grad_clip=grad_clip,
                )
            elif meta_algorithm == "reptile":
                metrics = meta_train_shared_batch_reptile(
                    loss_head,
                    fast_params,
                    initial_fast_state,
                    batch,
                    inner_steps=inner_steps,
                    inner_lr=inner_lr,
                    adapt_act_steps=adapt_act_steps,
                    query_act_steps=query_act_steps,
                    grad_clip=grad_clip,
                    meta_outer_lr=meta_outer_lr,
                    train_backbone=train_backbone,
                    light_query=False,
                )
            elif meta_algorithm == "openai_reptile":
                metrics = meta_train_shared_batch_reptile(
                    loss_head,
                    fast_params,
                    initial_fast_state,
                    batch,
                    inner_steps=inner_steps,
                    inner_lr=inner_lr,
                    adapt_act_steps=adapt_act_steps,
                    query_act_steps=query_act_steps,
                    grad_clip=grad_clip,
                    meta_outer_lr=meta_outer_lr,
                    train_backbone=train_backbone,
                    light_query=True,
                )
            elif meta_algorithm == "openai_foml":
                metrics = meta_train_shared_batch_openai_foml(
                    loss_head,
                    fast_params,
                    initial_fast_state,
                    batch,
                    inner_steps=inner_steps,
                    inner_lr=inner_lr,
                    adapt_act_steps=adapt_act_steps,
                    query_act_steps=query_act_steps,
                    grad_clip=grad_clip,
                    meta_outer_lr=meta_outer_lr,
                    train_backbone=train_backbone,
                    light_query=True,
                )
            else:
                raise ValueError(f"Unknown meta_algorithm={meta_algorithm!r} for adaptation_mode='shared_batch'.")
            ingest(metrics)
        elif adaptation_mode == "per_episode":
            if meta_algorithm in ("reptile", "openai_reptile"):
                light_q = meta_algorithm == "openai_reptile"
                sum_deltas = {k: torch.zeros_like(v) for k, v in initial_fast_state.items()}
                n_reptile = 0
                for i in range(batch["inputs"].shape[0]):
                    metrics = meta_train_episode_reptile_step(
                        loss_head,
                        fast_params,
                        initial_fast_state,
                        batch,
                        i,
                        inner_steps=inner_steps,
                        inner_lr=inner_lr,
                        adapt_act_steps=adapt_act_steps,
                        query_act_steps=query_act_steps,
                        grad_clip=grad_clip,
                        light_query=light_q,
                    )
                    sk = float(metrics.get("skipped", 0.0))
                    if sk >= 1.0:
                        ingest(metrics)
                        continue
                    n_reptile += 1
                    delta = metrics["delta"]
                    for k in sum_deltas:
                        sum_deltas[k] = sum_deltas[k] + delta[k]
                    ingest(metrics)
                if n_reptile > 0:
                    own = loss_head.state_dict()
                    inv_eps = meta_outer_lr / float(n_reptile)
                    for k in initial_fast_state:
                        own[k].copy_(initial_fast_state[k] + inv_eps * sum_deltas[k])
            elif meta_algorithm == "openai_foml":
                sum_last = {k: torch.zeros_like(v) for k, v in initial_fast_state.items()}
                n_foml = 0
                for i in range(batch["inputs"].shape[0]):
                    metrics = meta_train_episode_openai_foml_step(
                        loss_head,
                        fast_params,
                        initial_fast_state,
                        batch,
                        i,
                        inner_steps=inner_steps,
                        inner_lr=inner_lr,
                        adapt_act_steps=adapt_act_steps,
                        query_act_steps=query_act_steps,
                        grad_clip=grad_clip,
                        light_query=True,
                    )
                    sk = float(metrics.get("skipped", 0.0))
                    if sk >= 1.0:
                        ingest(metrics)
                        continue
                    n_foml += 1
                    ld = metrics["last_delta"]
                    for k in sum_last:
                        sum_last[k] = sum_last[k] + ld[k]
                    ingest(metrics)
                if n_foml > 0:
                    own = loss_head.state_dict()
                    inv_eps = meta_outer_lr / float(n_foml)
                    for k in initial_fast_state:
                        own[k].copy_(initial_fast_state[k] + inv_eps * sum_last[k])
            elif meta_algorithm == "fomaml":
                for i in range(batch["inputs"].shape[0]):
                    metrics = meta_train_episode_fomaml(
                        loss_head,
                        fast_params,
                        initial_fast_state,
                        batch,
                        i,
                        inner_steps=inner_steps,
                        inner_lr=inner_lr,
                        adapt_act_steps=adapt_act_steps,
                        query_act_steps=query_act_steps,
                        grad_clip=grad_clip,
                        meta_batch_size=batch["inputs"].shape[0],
                    )
                    ingest(metrics)
            else:
                raise ValueError(
                    f"meta_algorithm {meta_algorithm!r} is not supported with adaptation_mode='per_episode'."
                )
        else:
            raise ValueError(f"Unknown adaptation_mode={adaptation_mode!r}")

        current_grad_norm = grad_norm(all_params)
        do_optimizer_step = (meta_algorithm == "fomaml") or train_backbone
        if do_optimizer_step:
            # Apply cosine LR schedule (was a bug: previously locked to base_lr -> constant LR).
            # compute_lr respects warmup + lr_min_ratio. Using train_state.total_steps as horizon.
            from pretrain import compute_lr as _compute_lr
            for optim, base_lr in zip(optimizers, optimizer_lrs):
                lr_this_step = _compute_lr(base_lr, config, train_state)
                for group in optim.param_groups:
                    group["lr"] = lr_this_step
                optim.step()

        train_logger.mark_step_compute_done()

        denom = max(batch["inputs"].shape[0] - totals["skipped"], 1.0)
        query_loss = totals["query_loss"] / denom
        if train_state.step % config.log_interval == 0:
            q_prf = q_precision_recall_f1_from_totals(
                totals["q_attempted"],
                totals["q_attempted_correct"],
                totals["q_solvable"],
                key_prefix="train",
            )
            log = {
                "train/lm_loss": query_loss,
                "train/query_loss": query_loss,
                "train/accuracy": totals["accuracy"] / denom,
                "train/exact_accuracy": totals["exact_accuracy"] / denom,
                "train/count": denom,
                "train/grad_norm": current_grad_norm,
                "train/meta_outer_lr": meta_outer_lr,
                "train/steps": totals["steps"] / denom,
                "train/support_count": totals["support_count"] / denom,
                "train/skipped": totals["skipped"],
                "train/lr": optimizer_lrs[0],
                "train/samples_seen": float(train_state.step * config.global_batch_size),
                "train/q_halt_loss": totals["q_halt_loss"] / denom,
                **q_prf,
            }
            if "contrastive_loss" in totals:
                log["train/contrastive_loss"] = totals["contrastive_loss"] / denom
            # Full-ACT probe is expensive; only run when we will actually log (avoids throwaway work).
            if (
                full_act_probe_interval > 0
                and train_state.step % full_act_probe_interval == 0
            ):
                log.update(
                    full_act_probe_metrics(
                        loss_head,
                        fast_params,
                        adapter_state_dict(loss_head, "lora"),
                        batch,
                        inner_steps=inner_steps,
                        inner_lr=inner_lr,
                        adapt_act_steps=adapt_act_steps,
                        act_steps=full_act_probe_steps,
                        grad_clip=grad_clip,
                    )
                )
            log_extras: Dict[str, Any] = {}
            if outputs_log is not None:
                log_extras["preds"] = torch.argmax(outputs_log["logits"], dim=-1).detach()
            log_batch: Any = batch
            if carry_log is not None and hasattr(carry_log, "current_data"):
                log_batch = carry_log.current_data
            train_logger.log_train_step(
                train_state.step,
                log,
                log_batch,
                carry_log,
                log_extras,
                config.global_batch_size,
            )
            print(f"step={train_state.step} {log}")
            wandb.log(log, step=train_state.step)

        # Periodic in-training val eval — same inference protocol as the test eval.
        if (
            val_arrays is not None
            and val_row_pool
            and val_interval > 0
            and train_state.step > 0
            and train_state.step % val_interval == 0
        ):
            # Sample val_episodes from the pre-shuffled eligible pool.
            sampled = val_row_pool[: val_episodes]
            # Rotate the pool so successive probes see different rows.
            val_row_pool = val_row_pool[val_episodes:] + val_row_pool[:val_episodes]
            val_metrics = run_val_zero_shot(
                loss_head,
                val_arrays,
                sampled,
                max_support=int(loss_head.model.config.max_support_examples),
                query_act_steps=int(loss_head.model.config.halt_max_steps),
                ignore_label_id=int(getattr(train_metadata, "ignore_label_id", 0) or 0),
            )
            print(f"step={train_state.step} VAL {val_metrics}")
            wandb.log(val_metrics, step=train_state.step)

            # FF1 — paired N-shot meta-val + delta + cost (when enabled).
            if meta_val_evaluator is not None and meta_val_n_shot:
                try:
                    ff1_metrics = meta_val_evaluator.run(
                        loss_head, fast_params,
                        inner_steps=inner_steps, inner_lr=inner_lr,
                        max_episodes=meta_val_episodes,
                        include_cost=meta_val_include_cost,
                        checkpoint_step=train_state.step,
                    )
                    ff1_log = ff1_metrics.as_wandb_dict()
                    print(f"step={train_state.step} META_VAL pass={ff1_log['meta_val/pass@1']:.4f} "
                          f"delta={ff1_log['meta_val/delta']:.4f} "
                          f"cost_p50={ff1_log.get('meta_val/cost_p50_ms', float('nan')):.1f}ms "
                          f"n_tasks={ff1_log['meta_val/n_tasks_evaluated']}")
                    wandb.log(ff1_log, step=train_state.step)
                except Exception as _ff1_err:
                    print(f"[support-ttt] FF1 meta-val failed at step {train_state.step}: {_ff1_err}")
                    import traceback as _tb
                    _tb.print_exc()

        if config.checkpoint_interval > 0 and train_state.step % config.checkpoint_interval == 0:
            save_train_state(config, train_state)

    save_train_state(config, train_state)
    wandb.finish()


if __name__ == "__main__":
    launch()
