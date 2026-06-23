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

