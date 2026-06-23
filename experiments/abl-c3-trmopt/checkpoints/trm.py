from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from utils.models.common import trunc_normal_init_
from utils.models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, Rotary2DEmbeddingAxial, Rotary2DEmbeddingMixed, CosSin, CastedEmbedding, CastedLinear
from utils.models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    grid_size: int = 30

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
    gate_latent_input: bool = False
    input_injection_mode: str = "add"  # "add", "gate", "film"

    # Consistency loss
    consistency_loss_weight: float = 0.0

    # Per-position confidence gate (discard mechanism)
    use_position_gate: bool = False

    # Spatial attention bias
    use_spatial_bias: bool = False

    # Composite softprompt: factor puzzle embedding into task + transform + color tokens
    use_composite_softprompt: bool = False
    num_task_identifiers: Optional[int] = None
    num_transform_identifiers: int = 8

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
                use_spatial_bias=config.use_spatial_bias,
                grid_size=config.grid_size,
                max_seq_len=config.seq_len + (-(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_len == 0 else config.puzzle_emb_len),
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
            use_conv=getattr(config, 'use_conv_swiglu', False),
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.injection_mode = config.input_injection_mode
        self.layers = torch.nn.ModuleList(layers)

        if self.injection_mode == "gate" or config.gate_latent_input:
            self.injection_mode = "gate"
            self.latent_input_gate = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
            with torch.no_grad():
                self.latent_input_gate.bias.fill_(2.0)
        elif self.injection_mode == "film":
            self.film_generator = CastedLinear(config.hidden_size, 2 * config.hidden_size, bias=True)
            with torch.no_grad():
                # Init gamma-part bias to 0 (so gamma starts at 1.0 after the +1)
                # Init beta-part bias to 0 (no shift initially)
                self.film_generator.bias.zero_()

        # Per-position confidence gate (discard mechanism)
        self.use_position_gate = config.use_position_gate
        if self.use_position_gate:
            self.position_gate = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
            with torch.no_grad():
                # Init bias positive so gate starts near 1.0 (pass-through), model learns to discard
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
        else:  # "add"
            hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        if self.use_position_gate:
            gate = torch.sigmoid(self.position_gate(hidden_states))
            hidden_states = residual + gate * (hidden_states - residual)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        if self.config.puzzle_emb_ndim > 0 and self.config.use_composite_softprompt:
            self.puzzle_emb_len = 3
            self.task_emb = CastedSparseEmbedding(
                self.config.num_task_identifiers, self.config.hidden_size,
                batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)
            self.transform_emb = CastedSparseEmbedding(
                self.config.num_transform_identifiers, self.config.hidden_size,
                batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)
            self.color_proj = CastedLinear(10, self.config.hidden_size, bias=False)
            with torch.no_grad():
                self.color_proj.weight.zero_()
        else:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
            if self.config.puzzle_emb_ndim > 0:
                # Zero init puzzle embeddings
                self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                        batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "2d-rope-axial":
            self.rotary_emb = Rotary2DEmbeddingAxial(
                dim=self.config.hidden_size // self.config.num_heads,
                max_seq_len=self.config.seq_len + self.puzzle_emb_len,
                grid_size=self.config.grid_size,
                base=self.config.rope_theta)
        elif self.config.pos_encodings == "2d-rope-mixed":
            self.rotary_emb = Rotary2DEmbeddingMixed(
                dim=self.config.hidden_size // self.config.num_heads,
                max_seq_len=self.config.seq_len + self.puzzle_emb_len,
                grid_size=self.config.grid_size,
                base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(config=self.config, layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor,
                          task_identifiers: Optional[torch.Tensor] = None,
                          transform_ids: Optional[torch.Tensor] = None,
                          color_maps: Optional[torch.Tensor] = None):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if hasattr(self, 'task_emb'):
            # Composite softprompt: 3 factored tokens
            task_tok = self.task_emb(task_identifiers).unsqueeze(1)        # (B,1,H)
            transform_tok = self.transform_emb(transform_ids).unsqueeze(1) # (B,1,H)
            color_tok = self.color_proj(color_maps.float()).unsqueeze(1)    # (B,1,H)
            puzzle_embedding = torch.cat([task_tok, transform_tok, color_tok], dim=1) # (B,3,H)
            embedding = torch.cat((puzzle_embedding, embedding), dim=-2)
        elif self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_identifiers"],
            task_identifiers=batch.get("task_identifiers"),
            transform_ids=batch.get("transform_ids"),
            color_maps=batch.get("color_maps"),
        )

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        z_H_penultimate = None
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                input_injection = z_H + input_embeddings
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, input_injection, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
            # Save penultimate z_H for consistency loss
            if self.config.consistency_loss_weight > 0:
                z_H_penultimate = z_H.detach().clone()
        # 1 with grad
        input_injection = z_H + input_embeddings
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, input_injection, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), z_H_penultimate


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    @property
    def sparse_embeddings(self):
        """All CastedSparseEmbedding modules for optimizer setup."""
        if hasattr(self.inner, 'task_emb'):
            return [self.inner.task_emb, self.inner.transform_emb]
        return [self.inner.puzzle_emb]

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), z_H_penultimate = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }
        if z_H_penultimate is not None:
            outputs["z_H_penultimate"] = z_H_penultimate
            outputs["z_H_final"] = new_inner_carry.z_H

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
