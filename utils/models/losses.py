from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            # Q-value decision metrics for precision/recall
            q_says_halt = outputs["q_halt_logits"] >= 0  # agent is confident
            q_attempted = valid_metrics & q_says_halt
            q_attempted_correct = q_attempted & seq_is_correct
            q_solvable = valid_metrics & seq_is_correct

            metrics = {
                "count": valid_metrics.sum(),

                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & (q_says_halt == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),

                # Raw counts for precision/recall/F1 (extracted before generic postprocessing)
                "q_attempted": q_attempted.sum(),
                "q_attempted_correct": q_attempted_correct.sum(),
                "q_solvable": q_solvable.sum(),
            }

        # Losses
        per_position_loss = self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor
        lm_loss = per_position_loss.sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Per-ACT-step loss tracking: group lm_loss by the step number at which items halted
        with torch.no_grad():
            per_item_loss = per_position_loss.detach().sum(-1)  # [batch]
            halt_max = self.model.config.halt_max_steps
            for act_s in range(1, halt_max + 1):
                mask_s = valid_metrics & (new_carry.steps == act_s)
                metrics[f"loss_step_{act_s}"] = torch.where(mask_s, per_item_loss, torch.zeros_like(per_item_loss)).sum()
                metrics[f"count_step_{act_s}"] = mask_s.float().sum()

        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Consistency loss: penalize divergence between penultimate and final z_H
        consistency_loss = 0
        if "z_H_penultimate" in outputs and "z_H_final" in outputs:
            cos_sim = F.cosine_similarity(outputs["z_H_final"], outputs["z_H_penultimate"], dim=-1)  # (B, S)
            consistency_loss = (1.0 - cos_sim).mean()
            consistency_weight = self.model.config.consistency_loss_weight
            metrics["consistency_loss"] = consistency_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        if "z_H_penultimate" in outputs:
            total_loss = total_loss + consistency_weight * consistency_loss

        # Canvas-family aux term (task-consistency). Scaled + emitted by the
        # model; ACTLossHead just adds it in. Guarded so it's a no-op for
        # models that don't emit it.
        if "aux_consistency_loss" in outputs:
            total_loss = total_loss + outputs["aux_consistency_loss"]
            metrics["aux_consistency_loss"] = outputs["aux_consistency_loss"].detach()

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


# =============================================================================
# CMLM Loss Head (mask-predict + shape head)
# =============================================================================

def _extract_shape_from_labels(labels: torch.Tensor, eos_id: int, grid_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recover (H_out, W_out) for each item in batch from the labels' EOS L-shape.

    Returns 1-indexed sizes in [1, grid_size]. If no EOS marker is found (grid
    fills the full canvas), returns grid_size.
    """
    B = labels.shape[0]
    grid = labels.view(B, grid_size, grid_size)
    col0_eos = (grid[:, :, 0] == eos_id)  # (B, grid_size) - is col 0 EOS in each row?
    has_h = col0_eos.any(dim=1)
    h_idx = col0_eos.int().argmax(dim=1)  # first True row idx; H_out = idx (since idx=H_out means row H_out is the EOS row)
    h_out = torch.where(has_h, h_idx, torch.full_like(h_idx, grid_size))
    row0_eos = (grid[:, 0, :] == eos_id)
    has_w = row0_eos.any(dim=1)
    w_idx = row0_eos.int().argmax(dim=1)
    w_out = torch.where(has_w, w_idx, torch.full_like(w_idx, grid_size))
    return h_out.clamp(min=1), w_out.clamp(min=1)


class CMLMLossHead(nn.Module):
    """Conditional masked LM loss head with optional shape-prediction head.

    Per training step:
      1. Sample a mask fraction r ~ U[min_mask_frac, max_mask_frac].
      2. Build `decoded_so_far` from labels: replace random r% of non-pad positions
         with MASK; pad positions stay at PAD_ID (NOT IGNORE).
      3. Inject `decoded_so_far` into model_kwargs["batch"] before calling model.
      4. Compute CE loss over MASKED non-pad positions only (the unmasked ones
         are trivially correct since the model is conditioned on them).
      5. Add ACT halt-head BCE (same as ACTLossHead) on full seq_is_correct.
      6. Add shape-head CE on (H_out, W_out) extracted from labels.

    The standard "predict from scratch" objective is the corner case mask=1.0.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_type: str = "stablemax_cross_entropy",
        mask_token_id: int = 12,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        grid_size: int = 30,
        min_mask_frac: float = 0.15,
        max_mask_frac: float = 1.0,
        shape_head_weight: float = 0.5,
        q_halt_weight: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.grid_size = grid_size
        self.min_mask_frac = min_mask_frac
        self.max_mask_frac = max_mask_frac
        self.shape_head_weight = shape_head_weight
        self.q_halt_weight = q_halt_weight

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(self, return_keys: Sequence[str], **model_kwargs):
        batch = model_kwargs.get("batch")
        if batch is None:
            raise ValueError("CMLMLossHead requires `batch` in model_kwargs")
        labels = batch["labels"]
        device = labels.device
        valid = (labels != IGNORE_LABEL_ID)  # non-pad positions

        # Sample mask fraction. During training only; at eval the caller is
        # responsible for setting `decoded_so_far` directly and self.training=False.
        if self.training:
            r = float(torch.empty(()).uniform_(self.min_mask_frac, self.max_mask_frac))
            rand = torch.rand_like(labels, dtype=torch.float32)
            mask_decision = (rand < r) & valid
            # decoded_so_far: PAD at non-valid, label at unmasked-valid, MASK at masked-valid.
            decoded = labels.clamp(min=0).clone()  # -100 -> 0 (PAD)
            decoded = torch.where(mask_decision, torch.full_like(decoded, self.mask_token_id), decoded)
            new_batch = dict(batch)
            new_batch["decoded_so_far"] = decoded
            model_kwargs = dict(model_kwargs)
            model_kwargs["batch"] = new_batch
            loss_mask = mask_decision  # only graded on masked positions
        else:
            # At eval, caller has already set decoded_so_far; grade all non-pad.
            loss_mask = valid

        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        # Recompute loss_mask in eval mode against the (possibly carry-updated) labels.
        if not self.training:
            loss_mask = (labels != IGNORE_LABEL_ID)

        logits = outputs["logits"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(logits, dim=-1)
            valid_full = (labels != IGNORE_LABEL_ID)
            counts_full = valid_full.sum(-1)
            divisor_full = counts_full.clamp_min(1).unsqueeze(-1)
            is_correct = valid_full & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == counts_full
            valid_metrics = new_carry.halted & (counts_full > 0)
            q_says_halt = outputs["q_halt_logits"] >= 0
            q_attempted = valid_metrics & q_says_halt
            q_attempted_correct = q_attempted & seq_is_correct
            q_solvable = valid_metrics & seq_is_correct
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / divisor_full).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & (q_says_halt == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
                "q_attempted": q_attempted.sum(),
                "q_attempted_correct": q_attempted_correct.sum(),
                "q_solvable": q_solvable.sum(),
            }

        # Mask-aware CE: per-position loss, but reduce only over `loss_mask`.
        per_position_loss = self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=loss_mask)
        # Normalize by mask count per sequence so different mask fractions are comparable.
        mask_counts = loss_mask.sum(-1).clamp_min(1).unsqueeze(-1)
        per_position_loss = per_position_loss / mask_counts
        lm_loss = per_position_loss.sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum",
        )

        # Shape head loss (from labels). Only compute on items where labels yield a valid shape.
        shape_loss = torch.tensor(0.0, device=device)
        if "shape_h_logits" in outputs and "shape_w_logits" in outputs:
            h_out, w_out = _extract_shape_from_labels(labels, self.eos_token_id, self.grid_size)
            # Target indexing: shape values in [1, grid_size] → indices [0, grid_size-1].
            tgt_h = (h_out - 1).clamp(0, self.grid_size - 1).to(torch.long)
            tgt_w = (w_out - 1).clamp(0, self.grid_size - 1).to(torch.long)
            shape_loss = (
                F.cross_entropy(outputs["shape_h_logits"], tgt_h, reduction="sum")
                + F.cross_entropy(outputs["shape_w_logits"], tgt_w, reduction="sum")
            )
            metrics["shape_loss"] = shape_loss.detach()
            with torch.no_grad():
                pred_h = outputs["shape_h_logits"].argmax(-1) + 1
                pred_w = outputs["shape_w_logits"].argmax(-1) + 1
                metrics["shape_accuracy"] = ((pred_h == h_out) & (pred_w == w_out)).sum()

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q-continue (kept for compatibility with ACT machinery).
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        total_loss = lm_loss + self.q_halt_weight * (q_halt_loss + q_continue_loss) + self.shape_head_weight * shape_loss

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
