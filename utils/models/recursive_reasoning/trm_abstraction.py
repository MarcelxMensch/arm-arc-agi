"""Accelerated Recursive Reasoning Model (ARM) v2: Perceiver-style cross-attention H-level.

The H-level introduces a fixed set of learned latent tokens that interact with
the full-resolution L-level through bidirectional cross-attention. This design
draws on the Perceiver architecture (Jaegle et al., 2021), which demonstrated
that a small set of learned latent variables can compress high-dimensional
inputs via iterative cross-attention, and the Set Transformer (Lee et al.,
2019), whose Induced Set Attention Blocks (ISAB) established the use of learned
inducing points as a permutation-invariant information bottleneck. The latent
tokens carry no positional encoding, following the Set Transformer convention,
which prevents them from encoding spatial positions and instead forces them to
represent abstract concepts such as objects, transformations, and relational
structures.

The bidirectional reasoning loop extends the unidirectional encoder of the
original Perceiver with a broadcast mechanism inspired by the gated
cross-attention layers of Flamingo (Alayrac et al., 2022), where one
representational stream injects guidance into another via cross-attention with
an optional learned gate. Within each H-cycle the following steps are executed:

  1. H->L broadcast: L-level tokens cross-attend to H latent tokens, receiving
     abstract guidance that conditions subsequent spatial reasoning.
  2. L reasoning: standard transformer blocks process the full 30x30 grid at
     the original spatial resolution.
  3. L->H perceive: H latent tokens cross-attend to the L-level output,
     extracting updated information from the detailed spatial representation.
  4. H reasoning: self-attention among the latent tokens enables manipulation
     of abstract concept representations without spatial constraints.

The halting mechanism builds on Adaptive Computation Time (Graves, 2016),
operating on the mean-pooled latent token representation to determine whether
further reasoning cycles are required. This allows the model to modulate its
computational depth based on task complexity, allocating additional H-cycles
to problems that demand deeper abstraction.

References:
  - Jaegle, A. et al. (2021). Perceiver: General Perception with Iterative
    Attention. ICML.
  - Lee, J. et al. (2019). Set Transformer: A Framework for Attention-Based
    Permutation-Invariant Input. ICML.
  - Alayrac, J.-B. et al. (2022). Flamingo: a Visual Language Model for
    Few-Shot Learning. NeurIPS.
  - Graves, A. (2016). Adaptive Computation Time for Recurrent Neural
    Networks. arXiv:1603.08983.
"""
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from utils.models.common import trunc_normal_init_
from utils.models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding, Rotary2DEmbeddingAxial,
    Rotary2DEmbeddingMixed, CosSin, CastedEmbedding, CastedLinear,
)
from utils.models.sparse_embedding import CastedSparseEmbedding
from utils.models.fsq_embedding import FactoredFSQEmbedding

IGNORE_LABEL_ID = -100


# ---------------------------------------------------------------------------
# Carry dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AcceleratedRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor  # [B, num_latent_tokens, H_hidden_size]
    z_L: torch.Tensor  # [B, puzzle_emb_len + grid_len, hidden_size]


@dataclass
class AcceleratedRecursiveReasoningModel_ACTV1Carry:
    inner_carry: AcceleratedRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class AcceleratedRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    # L-level (detailed reasoning) transformer config
    L_layers: int
    hidden_size: int       # D_L
    expansion: float
    num_heads: int
    pos_encodings: str

    # H-level (abstract reasoning) config
    H_layers: int          # number of H self-attention blocks
    H_hidden_size: int     # D_H (latent token dimension)
    H_num_heads: int
    H_expansion: float

    # Perceiver latent tokens (replaces spatial_stride)
    num_latent_tokens: int = 32
    # Keep spatial_stride for backwards compat but it's unused
    spatial_stride: int = 2

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    grid_size: int = 30

    # Halting
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True
    gate_latent_input: bool = False
    input_injection_mode: str = "add"
    # Ablation: skip the H→L broadcast entirely. When True, input_injection is
    # just the input embeddings; the L-level never receives aggregated z_H info.
    disable_broadcast: bool = False

    consistency_loss_weight: float = 0.0
    use_position_gate: bool = False
    use_spatial_bias: bool = False

    use_composite_softprompt: bool = False
    num_task_identifiers: Optional[int] = None
    num_transform_identifiers: int = 8

    # FSQ embedding mode
    embedding_mode: str = "flat"  # "flat", "fsq", or "smb"
    fsq_n_blocks: int = 12
    fsq_dims_per_block: int = 2
    fsq_levels: int = 5

    # SMB (Shared Memory Bank) embedding mode
    smb_task_emb_dim: int = 64

    # TBPTT: how many H-cycles get gradients (K in TBPTT terminology)
    grad_cycles: int = 1

    # Stability features for deeper stacks
    use_layer_scale: bool = False
    layer_scale_init: float = 1e-4
    use_carry_gate: bool = False
    carry_gate_bias: float = -2.0

    # TTT / anti-memorization features
    puzzle_dropout_prob: float = 0.0
    use_task_identifiers: bool = False  # index puzzle_emb by task_id (size=num_task_identifiers)
    lora_rank: int = 0                  # >0 adds depth_lora on puzzle prefix for TTT Run 2


# ---------------------------------------------------------------------------
# Cross-attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from one space, keys/values from another."""

    def __init__(self, d_query: int, d_kv: int, num_heads: int, norm_eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_query // num_heads
        assert d_query % num_heads == 0

        self.q_proj = CastedLinear(d_query, d_query, bias=False)
        self.k_proj = CastedLinear(d_kv, d_query, bias=False)
        self.v_proj = CastedLinear(d_kv, d_query, bias=False)
        self.o_proj = CastedLinear(d_query, d_query, bias=False)
        self.norm_eps = norm_eps

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D_q]
            kv:    [B, N_kv, D_kv]
        Returns:
            [B, N_q, D_q]
        """
        B, N_q, _ = query.shape
        N_kv = kv.shape[1]
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(query).view(B, N_q, H, D).transpose(1, 2)   # [B, H, N_q, D]
        k = self.k_proj(kv).view(B, N_kv, H, D).transpose(1, 2)     # [B, H, N_kv, D]
        v = self.v_proj(kv).view(B, N_kv, H, D).transpose(1, 2)     # [B, H, N_kv, D]

        out = F.scaled_dot_product_attention(q, k, v)                 # [B, H, N_q, D]
        out = out.transpose(1, 2).contiguous().view(B, N_q, H * D)   # [B, N_q, D_q]
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Stability helpers
# ---------------------------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.to(x.dtype)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class _LBlock(nn.Module):
    """L-level transformer block (full resolution, D_L)."""

    def __init__(self, config: AcceleratedRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        if config.mlp_t:
            plen = config.puzzle_emb_len if config.puzzle_emb_len != 0 else -(config.puzzle_emb_ndim // -config.hidden_size)
            self.mlp_t = SwiGLU(hidden_size=config.seq_len + plen, expansion=config.expansion)
        else:
            plen = config.puzzle_emb_len if config.puzzle_emb_len != 0 else -(config.puzzle_emb_ndim // -config.hidden_size)
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
                use_spatial_bias=config.use_spatial_bias,
                grid_size=config.grid_size,
                max_seq_len=config.seq_len + plen,
            )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

        self.ls_attn = LayerScale(config.hidden_size, config.layer_scale_init) if config.use_layer_scale else None
        self.ls_mlp = LayerScale(config.hidden_size, config.layer_scale_init) if config.use_layer_scale else None

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            if self.ls_attn is not None:
                attn_out = self.ls_attn(attn_out)
            hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)
        out = self.mlp(hidden_states)
        if self.ls_mlp is not None:
            out = self.ls_mlp(out)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class _HBlock(nn.Module):
    """H-level self-attention block among latent tokens (D_H)."""

    def __init__(self, config: AcceleratedRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        D_H = config.H_hidden_size
        # No positional encoding for latent tokens (they're not spatial)
        self.self_attn = Attention(
            hidden_size=D_H,
            head_dim=D_H // config.H_num_heads,
            num_heads=config.H_num_heads,
            num_key_value_heads=config.H_num_heads,
            causal=False,
            use_spatial_bias=False,
            grid_size=1,  # no spatial structure
            max_seq_len=config.num_latent_tokens,
        )
        self.mlp = SwiGLU(hidden_size=D_H, expansion=config.H_expansion)
        self.norm_eps = config.rms_norm_eps

        self.ls_attn = LayerScale(D_H, config.layer_scale_init) if config.use_layer_scale else None
        self.ls_mlp = LayerScale(D_H, config.layer_scale_init) if config.use_layer_scale else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(cos_sin=None, hidden_states=hidden_states)
        if self.ls_attn is not None:
            attn_out = self.ls_attn(attn_out)
        hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)
        out = self.mlp(hidden_states)
        if self.ls_mlp is not None:
            out = self.ls_mlp(out)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


# ---------------------------------------------------------------------------
# Reasoning modules
# ---------------------------------------------------------------------------

class _LReasoningModule(nn.Module):
    """L-level reasoning with input injection (same as TRM)."""

    def __init__(self, config: AcceleratedRecursiveReasoningModel_ACTV1Config, layers: List[_LBlock]):
        super().__init__()
        self.injection_mode = config.input_injection_mode
        self.layers = nn.ModuleList(layers)

        if self.injection_mode == "gate" or config.gate_latent_input:
            self.injection_mode = "gate"
            self.latent_input_gate = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
            with torch.no_grad():
                self.latent_input_gate.bias.fill_(2.0)
        elif self.injection_mode == "film":
            self.film_generator = CastedLinear(config.hidden_size, 2 * config.hidden_size, bias=True)
            with torch.no_grad():
                self.film_generator.bias.zero_()

        self.use_position_gate = config.use_position_gate
        if self.use_position_gate:
            self.position_gate = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
            with torch.no_grad():
                self.position_gate.bias.fill_(2.0)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        if self.injection_mode == "gate":
            g = torch.sigmoid(self.latent_input_gate(hidden_states))
            hidden_states = hidden_states + g * input_injection
        elif self.injection_mode == "film":
            film_params = self.film_generator(input_injection)
            gamma, beta = film_params.chunk(2, dim=-1)
            hidden_states = (1.0 + gamma) * hidden_states + beta
        else:
            hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        if self.use_position_gate:
            gate = torch.sigmoid(self.position_gate(hidden_states))
            hidden_states = residual + gate * (hidden_states - residual)
        return hidden_states


class _HReasoningModule(nn.Module):
    """H-level: cross-attend to L, then self-attend among latent tokens."""

    def __init__(self, config: AcceleratedRecursiveReasoningModel_ACTV1Config, layers: List[_HBlock]):
        super().__init__()
        D_H = config.H_hidden_size
        D_L = config.hidden_size

        # L→H: latent tokens perceive the L-level output
        self.cross_attn_perceive = CrossAttention(
            d_query=D_H, d_kv=D_L, num_heads=config.H_num_heads,
            norm_eps=config.rms_norm_eps,
        )
        self.norm_eps = config.rms_norm_eps
        self.layers = nn.ModuleList(layers)

    def forward(self, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        # Perceive: latent tokens attend to full-resolution L-level
        z_H = rms_norm(
            z_H + self.cross_attn_perceive(query=z_H, kv=z_L),
            variance_epsilon=self.norm_eps,
        )
        # Self-attention reasoning among latent tokens
        for layer in self.layers:
            z_H = layer(z_H)
        return z_H


class _BroadcastModule(nn.Module):
    """H→L: L-level tokens cross-attend to H latent tokens for abstract guidance."""

    def __init__(self, config: AcceleratedRecursiveReasoningModel_ACTV1Config):
        super().__init__()
        D_H = config.H_hidden_size
        D_L = config.hidden_size

        self.cross_attn_broadcast = CrossAttention(
            d_query=D_L, d_kv=D_H, num_heads=config.num_heads,
            norm_eps=config.rms_norm_eps,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, z_L: torch.Tensor, z_H: torch.Tensor) -> torch.Tensor:
        # L-level attends to H latent tokens to receive abstract guidance
        return self.cross_attn_broadcast(query=z_L, kv=z_H)


# ---------------------------------------------------------------------------
# Depth LoRA (per-cycle adapter for puzzle prefix — used in TTT Run 2)
# ---------------------------------------------------------------------------

class _DepthLoRA(nn.Module):
    """Per-cycle LoRA adapter for z_problem.

    delta(x, t) = lora_up(lora_down(x) * scale[t])
    Shared down/up projections; only scale is per-cycle.
    lora_up zero-initialised so delta = 0 at init.
    """

    def __init__(self, D: int, rank: int, n_cycles: int) -> None:
        super().__init__()
        self.down = CastedLinear(D, rank, bias=False)
        self.up = CastedLinear(rank, D, bias=False)
        self.scale = nn.Embedding(n_cycles, rank)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)      # zero-init → delta = 0 at init
        nn.init.normal_(self.scale.weight, std=0.02)

    def forward(self, z: torch.Tensor, cycle_idx: int) -> torch.Tensor:
        t = torch.tensor(cycle_idx, device=z.device)
        scale = self.scale(t).to(z.dtype)              # [rank]
        down_out = self.down(z) * scale.view(1, 1, -1) # [B, plen, rank]
        return z + self.up(down_out)


# ---------------------------------------------------------------------------
# Inner model
# ---------------------------------------------------------------------------

class AcceleratedRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: AcceleratedRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        D_L = config.hidden_size
        D_H = config.H_hidden_size
        N_latent = config.num_latent_tokens

        # ── I/O ──────────────────────────────────────────────────────────
        self.embed_scale = math.sqrt(D_L)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(config.vocab_size, D_L, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(D_L, config.vocab_size, bias=False)
        self.q_head = CastedLinear(D_H, 2, bias=True)  # operates on mean-pooled latent tokens

        # Puzzle embeddings
        if config.embedding_mode == "smb":
            self.puzzle_emb_len = 3
            self.smb_task_emb = nn.Embedding(config.num_task_identifiers, config.smb_task_emb_dim)
            self.smb_task_proj = CastedLinear(config.smb_task_emb_dim, D_L, bias=False)
            self.smb_transform_emb = nn.Embedding(config.num_transform_identifiers, D_L)
            self.smb_color_proj = CastedLinear(10, D_L, bias=False)
            nn.init.normal_(self.smb_task_emb.weight, std=0.02)
            nn.init.normal_(self.smb_transform_emb.weight, std=0.02)
            with torch.no_grad():
                self.smb_color_proj.weight.zero_()
        elif config.puzzle_emb_ndim > 0 and config.use_composite_softprompt:
            self.puzzle_emb_len = 3
            self.task_emb = CastedSparseEmbedding(
                config.num_task_identifiers, D_L,
                batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype)
            self.transform_emb = CastedSparseEmbedding(
                config.num_transform_identifiers, D_L,
                batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype)
            self.color_proj = CastedLinear(10, D_L, bias=False)
            with torch.no_grad():
                self.color_proj.weight.zero_()
        else:
            self.puzzle_emb_len = -(config.puzzle_emb_ndim // -D_L) if config.puzzle_emb_len == 0 else config.puzzle_emb_len
            if config.puzzle_emb_ndim > 0:
                if config.embedding_mode == "fsq":
                    self.puzzle_emb = FactoredFSQEmbedding(
                        num_embeddings=config.num_puzzle_identifiers,
                        embedding_dim=config.puzzle_emb_ndim,
                        batch_size=config.batch_size,
                        n_blocks=config.fsq_n_blocks,
                        dims_per_block=config.fsq_dims_per_block,
                        levels=config.fsq_levels,
                        init_std=0.5,
                        cast_to=self.forward_dtype,
                    )
                else:
                    if config.use_task_identifiers:
                        if config.num_task_identifiers is None:
                            raise ValueError(
                                "use_task_identifiers=True requires num_task_identifiers in dataset"
                            )
                        table_size = config.num_task_identifiers
                    else:
                        table_size = config.num_puzzle_identifiers
                    self.puzzle_emb = CastedSparseEmbedding(
                        table_size, config.puzzle_emb_ndim,
                        batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype)
                if config.use_task_identifiers:
                    self.puzzle_emb_ttt_delta = nn.Buffer(
                        torch.zeros(config.puzzle_emb_ndim,
                                    dtype=getattr(torch, config.forward_dtype)),
                        persistent=False,
                    )
                if config.lora_rank > 0:
                    self.puzzle_emb_lora = _DepthLoRA(
                        D=config.hidden_size,
                        rank=config.lora_rank,
                        n_cycles=config.H_cycles,
                    )

        # Patch config so _LBlock sees the resolved prefix length
        if self.puzzle_emb_len != config.puzzle_emb_len:
            config = config.model_copy(update={"puzzle_emb_len": self.puzzle_emb_len})
            self.config = config

        # ── Positional encodings (L-level only — latent tokens have no position) ──
        L_total_len = config.seq_len + self.puzzle_emb_len
        if config.pos_encodings == "rope":
            self.rotary_emb_L = RotaryEmbedding(
                dim=D_L // config.num_heads,
                max_position_embeddings=L_total_len,
                base=config.rope_theta)
        elif config.pos_encodings == "2d-rope-axial":
            self.rotary_emb_L = Rotary2DEmbeddingAxial(
                dim=D_L // config.num_heads,
                max_seq_len=L_total_len,
                grid_size=config.grid_size,
                base=config.rope_theta)
        elif config.pos_encodings == "2d-rope-mixed":
            self.rotary_emb_L = Rotary2DEmbeddingMixed(
                dim=D_L // config.num_heads,
                max_seq_len=L_total_len,
                grid_size=config.grid_size,
                base=config.rope_theta)
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(L_total_len, D_L, init_std=embed_init_std, cast_to=self.forward_dtype)

        # ── L-level reasoning (full resolution, D_L) ────────────────────
        self.L_level = _LReasoningModule(
            config=config,
            layers=[_LBlock(config) for _ in range(config.L_layers)],
        )

        # ── H-level reasoning (latent tokens, D_H) ─────────────────────
        self.H_level = _HReasoningModule(
            config=config,
            layers=[_HBlock(config) for _ in range(config.H_layers)],
        )

        # ── H→L broadcast (cross-attention from L queries to H keys) ───
        self.broadcast = _BroadcastModule(config=config)

        # ── Learned latent tokens ────────────────────────────────────────
        self.latent_tokens = nn.Parameter(
            trunc_normal_init_(torch.empty(N_latent, D_H, dtype=self.forward_dtype), std=1.0 / math.sqrt(D_H))
        )

        # ── Initial states ───────────────────────────────────────────────
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(D_L, dtype=self.forward_dtype), std=1),
            persistent=True)

        # ── Carry gate (identity-biased GRU-like gate on H-level between cycles) ──
        if config.use_carry_gate:
            self.carry_gate_proj = CastedLinear(D_H, D_H, bias=True)
            with torch.no_grad():
                self.carry_gate_proj.bias.fill_(config.carry_gate_bias)

        # Q head init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    # ── Input embeddings ─────────────────────────────────────────────────

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor,
                          task_identifiers: Optional[torch.Tensor] = None,
                          transform_ids: Optional[torch.Tensor] = None,
                          color_maps: Optional[torch.Tensor] = None):
        embedding = self.embed_tokens(input.to(torch.int32))

        if hasattr(self, 'smb_task_emb'):
            task_tok = self.smb_task_proj(self.smb_task_emb(task_identifiers)).unsqueeze(1)
            transform_tok = self.smb_transform_emb(transform_ids).unsqueeze(1)
            color_tok = self.smb_color_proj(color_maps.float()).unsqueeze(1)
            puzzle_embedding = torch.cat([task_tok, transform_tok, color_tok], dim=1).to(self.forward_dtype)
            embedding = torch.cat((puzzle_embedding, embedding), dim=-2)
        elif hasattr(self, 'task_emb'):
            task_tok = self.task_emb(task_identifiers).unsqueeze(1)
            transform_tok = self.transform_emb(transform_ids).unsqueeze(1)
            color_tok = self.color_proj(color_maps.float()).unsqueeze(1)
            puzzle_embedding = torch.cat([task_tok, transform_tok, color_tok], dim=1)
            embedding = torch.cat((puzzle_embedding, embedding), dim=-2)
        elif self.config.puzzle_emb_ndim > 0:
            if self.config.use_task_identifiers:
                puzzle_embedding = self.puzzle_emb(task_identifiers)
            else:
                puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            # Apply TTT delta in eval mode
            if not self.training and hasattr(self, 'puzzle_emb_ttt_delta'):
                puzzle_embedding = puzzle_embedding + self.puzzle_emb_ttt_delta.to(puzzle_embedding.dtype)
            # Apply puzzle dropout in train mode (after TTT delta, before reshape)
            if self.training and self.config.puzzle_dropout_prob > 0.0:
                keep = (torch.rand(puzzle_embedding.shape[0], 1, device=puzzle_embedding.device)
                        > self.config.puzzle_dropout_prob)
                puzzle_embedding = puzzle_embedding * keep.to(puzzle_embedding.dtype)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2)

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    # ── Carry management ─────────────────────────────────────────────────

    def empty_carry(self, batch_size: int):
        return AcceleratedRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len,
                            self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: AcceleratedRecursiveReasoningModel_ACTV1InnerCarry):
        # Reset z_H to learned latent tokens, z_L to L_init
        latent_init = self.latent_tokens.unsqueeze(0).expand(carry.z_H.shape[0], -1, -1)
        return AcceleratedRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), latent_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    # ── Forward ──────────────────────────────────────────────────────────

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

        # Input encoding (full resolution, D_L)
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_identifiers"],
            task_identifiers=batch.get("task_identifiers"),
            transform_ids=batch.get("transform_ids"),
            color_maps=batch.get("color_maps"),
        )

        z_H, z_L = carry.z_H, carry.z_L
        z_H_penultimate = None
        has_carry_gate = hasattr(self, 'carry_gate_proj')
        has_lora = self.config.lora_rank > 0
        dt = self.forward_dtype

        K = self.config.grad_cycles
        no_grad_steps = self.config.H_cycles - K

        # Pre-split puzzle prefix for LoRA if needed
        if has_lora:
            plen = self.puzzle_emb_len
            puzzle_prefix = input_embeddings[:, :plen].clone()  # [B, plen, D_L]
            token_emb = input_embeddings[:, plen:]              # [B, 901, D_L]

        disable_broadcast = self.config.disable_broadcast

        # ── No-grad reasoning cycles (TBPTT truncation boundary) ───────
        if no_grad_steps > 0:
            with torch.no_grad():
                for _H_step in range(no_grad_steps):
                    if has_lora:
                        adapted_prefix = self.puzzle_emb_lora(puzzle_prefix, _H_step)
                        cur_emb = torch.cat([adapted_prefix, token_emb], dim=1)
                    else:
                        cur_emb = input_embeddings
                    if disable_broadcast:
                        input_injection = cur_emb
                    else:
                        h_broadcast = self.broadcast(z_L, z_H)
                        input_injection = h_broadcast + cur_emb
                    for _L_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, input_injection, **seq_info_L)
                    z_H_prev = z_H
                    z_L = z_L.to(dt)
                    z_H = self.H_level(z_H, z_L)
                    if has_carry_gate:
                        gate = torch.sigmoid(self.carry_gate_proj(z_H))
                        z_H = gate * z_H + (1 - gate) * z_H_prev
                    z_H = z_H.to(dt)

        # Save penultimate for consistency loss
        if self.config.consistency_loss_weight > 0:
            z_H_penultimate = z_H.detach().clone()

        # ── With-grad cycles (K steps of BPTT) ─────────────────────────
        for _H_step in range(K):
            if has_lora:
                adapted_prefix = self.puzzle_emb_lora(puzzle_prefix, no_grad_steps + _H_step)
                cur_emb = torch.cat([adapted_prefix, token_emb], dim=1)
            else:
                cur_emb = input_embeddings
            if disable_broadcast:
                input_injection = cur_emb
            else:
                h_broadcast = self.broadcast(z_L, z_H)
                input_injection = h_broadcast + cur_emb
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, input_injection, **seq_info_L)
            z_H_prev = z_H
            z_L = z_L.to(dt)
            z_H = self.H_level(z_H, z_L)
            if has_carry_gate:
                gate = torch.sigmoid(self.carry_gate_proj(z_H))
                z_H = gate * z_H + (1 - gate) * z_H_prev
            z_H = z_H.to(dt)

        # ── Output ──────────────────────────────────────────────────────
        # Output from L-level (full resolution), skip puzzle tokens
        output = self.lm_head(z_L[:, self.puzzle_emb_len:])

        # Carry (detached)
        new_carry = AcceleratedRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach())

        # Q head: mean-pool latent tokens
        q_logits = self.q_head(z_H.mean(dim=1)).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), z_H_penultimate


# ---------------------------------------------------------------------------
# ACT wrapper
# ---------------------------------------------------------------------------

class AcceleratedRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper for Accelerated Recursive Reasoning Model."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = AcceleratedRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = AcceleratedRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    @property
    def sparse_embeddings(self):
        if hasattr(self.inner, 'smb_task_emb'):
            return []
        if hasattr(self.inner, 'task_emb'):
            return [self.inner.task_emb, self.inner.transform_emb]
        return [self.inner.puzzle_emb]

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return AcceleratedRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: AcceleratedRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[AcceleratedRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Reset halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), z_H_penultimate = \
            self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }
        if z_H_penultimate is not None:
            outputs["z_H_penultimate"] = z_H_penultimate
            outputs["z_H_final"] = new_inner_carry.z_H

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
                    _, _, (next_q_halt_logits, next_q_continue_logits), _ = \
                        self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(is_last_step, next_q_halt_logits,
                                    torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return AcceleratedRecursiveReasoningModel_ACTV1Carry(
            new_inner_carry, new_steps, halted, new_current_data), outputs
