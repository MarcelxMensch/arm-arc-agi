"""TRM Abstraction Support TTT.

Support-conditioned TRM-abstraction variant for ARC episodes. The model receives
one query grid plus support input/output pairs. A single support pair is selected
per ACT step. H-level latents read that pair through a single delta channel that
emphasizes positions where the support output differs from the support input.

Two delta-construction modes are supported, selected via
``config.support_evidence_mode``:

- ``emp_token``: positions where input == output are replaced with a dedicated
  EMP token, otherwise the support output token is used. The resulting grid is
  embedded with a separate small embedding table (vocab = colors+PAD+EOS+EMP).
- ``vector_delta``: continuous embedding-space difference E_out - E_in, computed
  with a role-free SupportTokenEmbedder so role/position embeddings cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from utils.models.common import trunc_normal_init_
from utils.models.layers import (
    rms_norm,
    RotaryEmbedding,
    Rotary2DEmbeddingAxial,
    Rotary2DEmbeddingMixed,
    CastedEmbedding,
    CastedLinear,
)
from utils.models.recursive_reasoning.trm_abstraction import (
    CrossAttention,
    _LBlock,
    _HBlock,
    _LReasoningModule,
    _HReasoningModule,
    _BroadcastModule,
)


@dataclass
class AcceleratedRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class AcceleratedRecursiveReasoningModel_ACTV1Carry:
    inner_carry: AcceleratedRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class AcceleratedRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    H_layers: int
    H_hidden_size: int
    H_num_heads: int
    H_expansion: float
    num_latent_tokens: int = 32
    spatial_stride: int = 2

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    grid_size: int = 30

    halt_max_steps: int
    halt_exploration_prob: float
    no_ACT_continue: bool = True

    forward_dtype: str = "bfloat16"

    mlp_t: bool = False
    puzzle_emb_len: int = 0
    gate_latent_input: bool = False
    input_injection_mode: str = "add"
    consistency_loss_weight: float = 0.0
    use_position_gate: bool = False
    use_spatial_bias: bool = False

    use_layer_scale: bool = False
    layer_scale_init: float = 1e-4
    use_carry_gate: bool = False
    carry_gate_bias: float = -2.0
    # When > 0, expose post-grad-cycle z_H with gradient flow as outputs["z_H_with_grad"]
    # so the trainer can attach an auxiliary supervised-contrastive loss on transformation
    # deltas in latent space (A8 Stage 1).
    contrastive_aux_weight: float = 0.0
    grad_cycles: int = 1

    max_support_examples: int = 8
    support_gate_bias: float = -2.0

    # Delta-channel configuration.
    # support_evidence_mode in {"emp_token", "vector_delta"}.
    support_evidence_mode: str = "emp_token"
    # EMP path: vocabulary is colors + PAD + EOS + EMP (default 13 = 12 + EMP).
    delta_vocab_size: int = 13
    emp_token_id: int = 12


class DeltaTokenEmbedder(nn.Module):
    """Separate small embedding table for the EMP-aware delta token grid.

    The vocabulary is independent from the main grid vocabulary so the EMP token
    is never visible to ``lm_head`` or the main ``embed_tokens`` table. Only
    position embedding is included; there is a single role (delta) so no role
    embedding is needed.
    """

    def __init__(self, config: AcceleratedRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        D_H = config.H_hidden_size
        self.embed_scale = math.sqrt(D_H)
        self.token_emb = CastedEmbedding(
            config.delta_vocab_size,
            D_H,
            init_std=1.0 / self.embed_scale,
            cast_to=self.forward_dtype,
        )
        self.pos_emb = nn.Parameter(
            trunc_normal_init_(
                torch.empty(config.seq_len, D_H, dtype=self.forward_dtype),
                std=1.0 / math.sqrt(D_H),
            )
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens.to(torch.int32))
        x = self.embed_scale * x
        x = x + self.pos_emb.unsqueeze(0).to(x.dtype)
        return x


class SupportTokenEmbedder(nn.Module):
    """Embed support input/output grids into H-width memory tokens.

    Used by the ``vector_delta`` mode. ``use_role_emb`` controls whether a role
    embedding is added. For E_out - E_in to cancel cleanly, both calls must use
    the same role configuration; ``vector_delta`` instantiates this with
    ``use_role_emb=False`` so position embeddings cancel and there is no
    constant role bias in the difference.
    """

    def __init__(
        self,
        config: AcceleratedRecursiveReasoningModel_ACTV1Config,
        use_role_emb: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_role_emb = use_role_emb
        self.forward_dtype = getattr(torch, config.forward_dtype)
        D_H = config.H_hidden_size
        self.embed_scale = math.sqrt(D_H)
        self.token_emb = CastedEmbedding(
            config.vocab_size,
            D_H,
            init_std=1.0 / self.embed_scale,
            cast_to=self.forward_dtype,
        )
        self.pos_emb = nn.Parameter(
            trunc_normal_init_(
                torch.empty(config.seq_len, D_H, dtype=self.forward_dtype),
                std=1.0 / math.sqrt(D_H),
            )
        )
        if self.use_role_emb:
            self.role_emb = nn.Parameter(
                trunc_normal_init_(
                    torch.empty(2, D_H, dtype=self.forward_dtype),
                    std=1.0 / math.sqrt(D_H),
                )
            )

    def forward(self, tokens: torch.Tensor, role: int = 0) -> torch.Tensor:
        x = self.token_emb(tokens.to(torch.int32))
        x = self.embed_scale * x
        x = x + self.pos_emb.unsqueeze(0).to(x.dtype)
        if self.use_role_emb:
            x = x + self.role_emb[role].view(1, 1, -1).to(x.dtype)
        return x


class DeltaReadBlock(nn.Module):
    """Single-channel delta read.

    H cross-attends to a delta representation of the active support pair, then a
    gated MLP folds the read result back into z_H. Compared to the previous
    two-attention block, this halves the cross-attention cost and forces H to
    work with a difference signal.
    """

    def __init__(self, config: AcceleratedRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        D_H = config.H_hidden_size
        self.delta_read = CrossAttention(D_H, D_H, config.H_num_heads, config.rms_norm_eps)
        evidence_dim = 4 * D_H
        hidden_dim = 4 * D_H
        self.candidate_in = CastedLinear(evidence_dim, hidden_dim, bias=False)
        self.candidate_out = CastedLinear(hidden_dim, D_H, bias=False)
        self.gate_proj = CastedLinear(evidence_dim, D_H, bias=True)
        self.norm_eps = config.rms_norm_eps
        with torch.no_grad():
            self.gate_proj.bias.fill_(config.support_gate_bias)  # type: ignore

    def forward(self, z_H: torch.Tensor, delta_tokens: torch.Tensor) -> torch.Tensor:
        a = self.delta_read(z_H, delta_tokens)
        evidence = torch.cat([z_H, a, a.abs(), z_H * a], dim=-1)
        candidate = self.candidate_out(F.silu(self.candidate_in(evidence)))
        gate = torch.sigmoid(self.gate_proj(evidence))
        return rms_norm(z_H + gate * candidate, variance_epsilon=self.norm_eps)


class AcceleratedRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: AcceleratedRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        if config.support_evidence_mode not in ("emp_token", "vector_delta", "none"):
            raise ValueError(
                f"support_evidence_mode={config.support_evidence_mode!r} not supported"
            )

        D_L = config.hidden_size
        D_H = config.H_hidden_size

        self.embed_scale = math.sqrt(D_L)
        self.embed_tokens = CastedEmbedding(
            config.vocab_size,
            D_L,
            init_std=1.0 / self.embed_scale,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(D_L, config.vocab_size, bias=False)
        self.q_head = CastedLinear(D_H, 2, bias=True)
        self.puzzle_emb_len = 0

        if config.pos_encodings == "rope":
            self.rotary_emb_L = RotaryEmbedding(
                dim=D_L // config.num_heads,
                max_position_embeddings=config.seq_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "2d-rope-axial":
            self.rotary_emb_L = Rotary2DEmbeddingAxial(
                dim=D_L // config.num_heads,
                max_seq_len=config.seq_len,
                grid_size=config.grid_size,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "2d-rope-mixed":
            self.rotary_emb_L = Rotary2DEmbeddingMixed(
                dim=D_L // config.num_heads,
                max_seq_len=config.seq_len,
                grid_size=config.grid_size,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(config.seq_len, D_L, init_std=1.0 / self.embed_scale, cast_to=self.forward_dtype)

        self.L_level = _LReasoningModule(
            config=config,
            layers=[_LBlock(config) for _ in range(config.L_layers)],
        )
        self.H_level = _HReasoningModule(
            config=config,
            layers=[_HBlock(config) for _ in range(config.H_layers)],
        )
        self.broadcast = _BroadcastModule(config=config)

        if config.support_evidence_mode == "emp_token":
            self.delta_embedder_emp = DeltaTokenEmbedder(config)
        elif config.support_evidence_mode == "vector_delta":
            self.support_embedder_vec = SupportTokenEmbedder(config, use_role_emb=False)
        # support_evidence_mode == "none": no embedder. delta_read_block is still
        # constructed (LoRA installer expects it) but is never invoked in _cycle.

        self.delta_read_block = DeltaReadBlock(config)

        self.latent_tokens = nn.Parameter(
            trunc_normal_init_(torch.empty(config.num_latent_tokens, D_H, dtype=self.forward_dtype), std=1.0 / math.sqrt(D_H))
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(D_L, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        if config.use_carry_gate:
            self.carry_gate_proj = CastedLinear(D_H, D_H, bias=True)
            with torch.no_grad():
                self.carry_gate_proj.bias.fill_(config.carry_gate_bias)  # type: ignore

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input.to(torch.int32))
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def _build_delta_tokens(
        self,
        active_support_inputs: torch.Tensor,
        active_support_outputs: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.config.support_evidence_mode == "emp_token":
            same = active_support_inputs == active_support_outputs
            emp = torch.full_like(active_support_inputs, self.config.emp_token_id)
            delta_ids = torch.where(same, emp, active_support_outputs)
            return self.delta_embedder_emp(delta_ids)
        if self.config.support_evidence_mode == "vector_delta":
            e_in = self.support_embedder_vec(active_support_inputs)
            e_out = self.support_embedder_vec(active_support_outputs)
            return e_out - e_in
        # support_evidence_mode == "none": no support signal flows through the architecture.
        # TTT alone provides the task conditioning at inference time.
        return None

    def empty_carry(self, batch_size: int) -> AcceleratedRecursiveReasoningModel_ACTV1InnerCarry:
        return AcceleratedRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            z_L=torch.empty(batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: AcceleratedRecursiveReasoningModel_ACTV1InnerCarry,
    ) -> AcceleratedRecursiveReasoningModel_ACTV1InnerCarry:
        latent_init = self.latent_tokens.unsqueeze(0).expand(carry.z_H.shape[0], -1, -1)
        return AcceleratedRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), latent_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def _cycle(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        input_embeddings: torch.Tensor,
        delta_tokens: Optional[torch.Tensor],
        seq_info_L: Dict[str, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if delta_tokens is not None and self.config.support_evidence_mode != "none":
            z_H = self.delta_read_block(z_H, delta_tokens)
        h_broadcast = self.broadcast(z_L, z_H)
        input_injection = h_broadcast + input_embeddings
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, input_injection, **seq_info_L)

        z_H_prev = z_H
        z_H = self.H_level(z_H, z_L.to(self.forward_dtype))
        if hasattr(self, "carry_gate_proj"):
            gate = torch.sigmoid(self.carry_gate_proj(z_H))
            z_H = gate * z_H + (1 - gate) * z_H_prev
        return z_H.to(self.forward_dtype), z_L.to(self.forward_dtype)

    def forward(
        self,
        carry: AcceleratedRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        AcceleratedRecursiveReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Optional[torch.Tensor],
    ]:
        seq_info_L = dict(
            cos_sin=self.rotary_emb_L() if hasattr(self, "rotary_emb_L") else None,
        )
        input_embeddings = self._input_embeddings(batch["inputs"])
        delta_tokens = self._build_delta_tokens(
            batch["active_support_inputs"],
            batch["active_support_outputs"],
        )

        z_H, z_L = carry.z_H, carry.z_L
        z_H_penultimate = None
        no_grad_steps = max(0, self.config.H_cycles - self.config.grad_cycles)

        if no_grad_steps > 0:
            with torch.no_grad():
                for _ in range(no_grad_steps):
                    z_H, z_L = self._cycle(
                        z_H,
                        z_L,
                        input_embeddings,
                        delta_tokens,
                        seq_info_L,
                    )

        if self.config.consistency_loss_weight > 0:
            z_H_penultimate = z_H.detach().clone()

        for _ in range(self.config.grad_cycles):
            z_H, z_L = self._cycle(
                z_H,
                z_L,
                input_embeddings,
                delta_tokens,
                seq_info_L,
            )

        output = self.lm_head(z_L)
        new_carry = AcceleratedRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        q_logits = self.q_head(z_H.mean(dim=1)).to(torch.float32)
        z_H_with_grad: Optional[torch.Tensor] = z_H if self.config.contrastive_aux_weight > 0.0 else None
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), z_H_penultimate, z_H_with_grad


class AcceleratedRecursiveReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = AcceleratedRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = AcceleratedRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return None

    @property
    def sparse_embeddings(self):
        return []

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> AcceleratedRecursiveReasoningModel_ACTV1Carry:
        batch_size = batch["inputs"].shape[0]
        return AcceleratedRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def _select_support_indices(
        self,
        support_mask: torch.Tensor,
        steps: torch.Tensor,
    ) -> torch.Tensor:
        B, K = support_mask.shape
        if self.training:
            scores = torch.rand(B, K, device=support_mask.device)
            scores = scores.masked_fill(~support_mask, -1.0)
            return scores.argmax(dim=1)

        counts = support_mask.sum(dim=1).clamp_min(1).to(steps.dtype)
        return (steps % counts).to(torch.long)

    def forward(
        self,
        carry: AcceleratedRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[AcceleratedRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        if "support_inputs" not in batch or "support_outputs" not in batch or "support_mask" not in batch:
            raise KeyError(
                "trm_abstraction_support_ttt requires support_inputs, support_outputs, and support_mask. "
                "Run pretrain.py with support_ttt_mode=true."
            )

        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        support_mask = new_current_data["support_mask"].to(torch.bool)
        support_idx = self._select_support_indices(support_mask, new_steps)
        row_idx = torch.arange(support_idx.shape[0], device=support_idx.device)
        active_support_inputs = new_current_data["support_inputs"][row_idx, support_idx]
        active_support_outputs = new_current_data["support_outputs"][row_idx, support_idx]

        inner_data = dict(new_current_data)
        inner_data["active_support_inputs"] = active_support_inputs
        inner_data["active_support_outputs"] = active_support_outputs

        new_inner_carry, logits, (q_halt_logits, q_continue_logits), z_H_penultimate, z_H_with_grad = self.inner(
            new_inner_carry,
            inner_data,
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "support_indices": support_idx,
        }
        if z_H_penultimate is not None:
            outputs["z_H_penultimate"] = z_H_penultimate
            outputs["z_H_final"] = new_inner_carry.z_H
        if z_H_with_grad is not None:
            outputs["z_H_with_grad"] = z_H_with_grad

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(
                        new_inner_carry,
                        inner_data,
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return AcceleratedRecursiveReasoningModel_ACTV1Carry(
            new_inner_carry,
            new_steps,
            halted,
            new_current_data,
        ), outputs
