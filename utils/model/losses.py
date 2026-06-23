from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


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


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
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

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics: halted-only for count/q_halt; all valid sequences for accuracy so we get signal from step 1
            valid_metrics = new_carry.halted & (loss_counts > 0)
            valid_metrics_all = loss_counts > 0
            metrics = {
                "count": valid_metrics.sum(),
                "count_all": valid_metrics_all.sum(),
                "accuracy": torch.where(valid_metrics_all, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics_all & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss) – always log a scalar every step
        q_continue_loss = torch.tensor(0.0, device=labels.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )
        metrics["q_continue_loss"] = q_continue_loss.detach()

        # Optional debug metrics (z-norms, etc.) when model.debug is enabled
        if getattr(self.model, "debug", False):
            # Prefer inner._log_z_* (matches original HRM pretrain logging), fallback to outputs[*] if not available.
            z_H_norm = z_L_norm = z_H_delta_norm = None
            inner = getattr(self.model, "inner", None)
            if inner is not None and hasattr(inner, "_log_z_H") and hasattr(inner, "_log_z_L") and hasattr(inner, "_log_z_H_delta"):
                try:
                    z_H_norm = inner._log_z_H.norm()
                    z_L_norm = inner._log_z_L.norm()
                    z_H_delta_norm = inner._log_z_H_delta.norm()
                except Exception:
                    z_H_norm = z_L_norm = z_H_delta_norm = None
            if z_H_norm is None:
                z_H_norm = outputs.get("z_H_norm", torch.tensor(0.0, device=labels.device))
                z_L_norm = outputs.get("z_L_norm", torch.tensor(0.0, device=labels.device))
                z_H_delta_norm = outputs.get("z_H_delta", torch.tensor(0.0, device=labels.device))
            metrics.update({
                "z_H_norm":        z_H_norm,
                "z_L_norm":        z_L_norm,
                "z_H_delta_norm":  z_H_delta_norm,
            })

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs and isinstance(outputs[k], torch.Tensor)}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()
