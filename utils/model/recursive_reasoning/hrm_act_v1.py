from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from ..common import trunc_normal_init_
from ..layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    RotaryEmbedding2DAxial,
    RotaryEmbedding2DMixed,
    CastedEmbedding,
    CastedLinear,
)
from ..embedding import CastedSparseEmbedding, CompositeSparseEmbedding, GridEncoderSoftPrompt


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    grid_size: int = 30  # ARC grid dimension for 2D RoPE
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Debug mode (enables extra metrics logging like z_H/z_L norms)
    debug: bool = False

    # Soft prompt config
    use_soft_prompts: bool = False  # False = legacy single-table puzzle embedding
    puzzle_softprompt_len: int = 13  # 2 task + 1 transform + 10 color (only if use_soft_prompts=True)
    num_task_identifiers: Optional[int] = None  # Auto-detected from data if None
    num_transform_identifiers: int = 8  # Fixed: 8 dihedral transforms
    num_color_identifiers: Optional[int] = None  # Auto-detected from data if None

    # If True, lm_head is fixed orthogonal (one-hot): logits = hidden[:, :vocab_size], no learned weights
    fixed_orthogonal_lm_head: bool = False

    # If True, use activation checkpointing in H/L reasoning layers to save memory (slower backward)
    gradient_checkpointing: bool = False


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(hidden_states=hidden_states, **kwargs),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block], gradient_checkpointing: bool = False):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    lambda h, l=layer: l(hidden_states=h, **kwargs),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class FixedOrthogonalLmHead(nn.Module):
    """LM head with fixed one-hot orthogonal output: logits = x[..., :vocab_size]. No parameters; effective weight is [I | 0] so rows are e_0, e_1, ... (PAD, EOS, c0..c9)."""

    def __init__(self, vocab_size: int, hidden_size: int, cast_dtype: Optional[torch.dtype] = None):
        super().__init__()
        assert hidden_size >= vocab_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self._cast_dtype = cast_dtype
        # Buffer so lm_head.weight exists for logging; shape (vocab_size, hidden_size), rows = e_0, e_1, ...
        w = torch.zeros(vocab_size, hidden_size)
        w[:, :vocab_size] = torch.eye(vocab_size)
        self.register_buffer("weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # logits = first vocab_size dimensions of hidden state (effective weight [I | 0])
        out = x[..., : self.vocab_size]
        if self._cast_dtype is not None:
            out = out.to(self._cast_dtype)
        return out


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        if getattr(self.config, "fixed_orthogonal_lm_head", False):
            self.lm_head = FixedOrthogonalLmHead(
                self.config.vocab_size,
                self.config.hidden_size,
                cast_dtype=self.forward_dtype,
            )
        else:
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Puzzle embedding configuration
        if getattr(self.config, "use_soft_prompts", False):
            # Grid-encoder soft prompts: [task_1, task_2, transform, color_0..color_9]
            self.puzzle_emb_len = GridEncoderSoftPrompt.SOFTPROMPT_LEN  # 13
            self.puzzle_emb = GridEncoderSoftPrompt(
                hidden_size=self.config.hidden_size,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )
            # Fallback single-table embedding when dataset lacks demo grids
            self.puzzle_emb_fallback = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim if self.config.puzzle_emb_ndim > 0 else self.config.hidden_size,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )
        else:
            # Legacy puzzle embedding: single table indexed by puzzle_identifiers
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
            if self.config.puzzle_emb_ndim > 0:
                self.puzzle_emb = CastedSparseEmbedding(
                    self.config.num_puzzle_identifiers,
                    self.config.puzzle_emb_ndim,
                    batch_size=self.config.batch_size,
                    init_std=0,
                    cast_to=self.forward_dtype,
                )

        # LM Blocks
        head_dim = self.config.hidden_size // self.config.num_heads
        max_seq_len = self.config.seq_len + self.puzzle_emb_len
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=max_seq_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings in ("2d-rope", "2d-rope-axial"):
            self.rotary_emb = RotaryEmbedding2DAxial(
                dim=head_dim,
                max_seq_len=max_seq_len,
                grid_size=self.config.grid_size,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "2d-rope-mixed":
            self.rotary_emb = RotaryEmbedding2DMixed(
                dim=head_dim,
                max_seq_len=max_seq_len,
                grid_size=self.config.grid_size,
                num_heads=self.config.num_heads,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise ValueError(f"Unknown pos_encodings: {self.config.pos_encodings}")

        # Reasoning Layers
        grad_ckpt = getattr(self.config, "gradient_checkpointing", False)
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)],
            gradient_checkpointing=grad_ckpt,
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)],
            gradient_checkpointing=grad_ckpt,
        )
        
        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build input embeddings, optionally prepending softprompt tokens."""
        # Token embedding
        embedding = self.embed_tokens(batch["inputs"].to(torch.int32))

        # Puzzle / softprompt embeddings
        if getattr(self.config, "use_soft_prompts", False):
            puzzle_embedding: Optional[torch.Tensor] = None

            # Primary path: grid encoder softprompt (new per-example format)
            if (
                "demo_inputs" in batch
                and "demo_outputs" in batch
                and "num_demos" in batch
                and "transform_ids" in batch
                and "color_maps" in batch
                and isinstance(self.puzzle_emb, GridEncoderSoftPrompt)
            ):
                puzzle_embedding = self.puzzle_emb(
                    demo_inputs=batch["demo_inputs"],
                    demo_outputs=batch["demo_outputs"],
                    num_demos=batch["num_demos"],
                    transform_ids=batch["transform_ids"],
                    color_maps=batch["color_maps"],
                )  # (B, 13, hidden_size)

            # Fallback: single-table embedding when demo grids are not available
            elif hasattr(self, "puzzle_emb_fallback"):
                fallback = self.puzzle_emb_fallback(batch["puzzle_identifiers"])
                batch_size = fallback.shape[0]
                emb_dim = fallback.shape[-1]
                if emb_dim < self.config.hidden_size:
                    pad = self.config.hidden_size - emb_dim
                    fallback = F.pad(fallback, (0, pad))
                puzzle_embedding = fallback.view(
                    batch_size, 1, self.config.hidden_size
                ).expand(-1, self.puzzle_emb_len, -1)

            if puzzle_embedding is not None:
                embedding = torch.cat((puzzle_embedding, embedding), dim=-2)

        elif self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(batch["puzzle_identifiers"])

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = {}
        if hasattr(self, "rotary_emb"):
            rope_output = self.rotary_emb()
            if self.config.pos_encodings in ("2d-rope", "2d-rope-axial"):
                seq_info["cos_sin_2d"] = rope_output
                seq_info["rope_mode"] = "2d-axial"
            elif self.config.pos_encodings == "2d-rope-mixed":
                cos_mixed, sin_mixed = rope_output
                seq_info["cos_sin_mixed"] = (cos_mixed, sin_mixed)
                seq_info["rope_mode"] = "2d-mixed"
            else:
                seq_info["cos_sin"] = rope_output
                seq_info["rope_mode"] = "1d"

        # Input encoding (optionally using soft prompts / grid encoder)
        input_embeddings = self._input_embeddings(batch)

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad (keep z_H before for delta)
        z_H_before_grad = z_H.detach().clone()
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Always save for system logging (norms, images, reverse embedding)
        seq_z_L = z_L[:, self.puzzle_emb_len:]
        seq_z_H = z_H[:, self.puzzle_emb_len:]
        seq_z_H_delta = (z_H - z_H_before_grad)[:, self.puzzle_emb_len:]
        self._log_z_L = seq_z_L.detach().cpu().float()
        self._log_z_H = seq_z_H.detach().cpu().float()
        self._log_z_H_delta = seq_z_H_delta.detach().cpu().float()

        # Snapshot z_H for reverse-embedding logging (train.py can read _z_H_snapshot and clear it)
        if getattr(self, "_save_z_H_for_logging", False):
            self._z_H_snapshot = z_H[:, self.puzzle_emb_len:].detach().cpu().float()
            self._save_z_H_for_logging = False

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    @property
    def debug(self) -> bool:
        return getattr(self.config, "debug", False)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items() if torch.is_tensor(v)}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # Optional debug metrics: z_H/z_L norms and z_H delta (match other recursive models)
        if self.debug:
            z_H = new_inner_carry.z_H.detach()
            z_L = new_inner_carry.z_L.detach()
            outputs["z_H_norm"] = z_H.norm(dim=-1).mean()
            outputs["z_L_norm"] = z_L.norm(dim=-1).mean()
            prev_z_H = carry.inner_carry.z_H.detach()
            if prev_z_H.shape == z_H.shape:
                outputs["z_H_delta"] = (z_H - prev_z_H).norm(dim=-1).mean()
            else:
                outputs["z_H_delta"] = torch.tensor(0.0, device=z_H.device, dtype=z_H.dtype)
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
