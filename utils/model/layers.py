from typing import Tuple
import os

import torch
from torch import nn
import torch.nn.functional as F

# Try to use flash-attn if available and compatible; otherwise fall back to standard attention.
FLASH_ATTN_DISABLED = os.environ.get("DISABLE_FLASH_ATTN", "0") == "1"
if not FLASH_ATTN_DISABLED:
    try:
        from flash_attn import flash_attn_func  # type: ignore[import]
    except Exception:
        flash_attn_func = None  # type: ignore[assignment]
else:
    flash_attn_func = None  # type: ignore[assignment]

from .common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    cos = cos.to(orig_dtype)
    sin = sin.to(orig_dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def apply_rotary_pos_emb_2d_axial(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_x: torch.Tensor,
    sin_x: torch.Tensor,
    cos_y: torch.Tensor,
    sin_y: torch.Tensor,
):
    """Apply 2D axial RoPE."""
    orig_dtype = q.dtype
    cos_x = cos_x.to(orig_dtype)
    sin_x = sin_x.to(orig_dtype)
    cos_y = cos_y.to(orig_dtype)
    sin_y = sin_y.to(orig_dtype)

    seq_len = q.shape[1]
    cos_x = cos_x[:seq_len]
    sin_x = sin_x[:seq_len]
    cos_y = cos_y[:seq_len]
    sin_y = sin_y[:seq_len]

    d = q.shape[-1] // 2
    q_x, q_y = q[..., :d], q[..., d:]
    k_x, k_y = k[..., :d], k[..., d:]

    q_x_rot = (q_x * cos_x.unsqueeze(-2)) + (rotate_half(q_x) * sin_x.unsqueeze(-2))
    q_y_rot = (q_y * cos_y.unsqueeze(-2)) + (rotate_half(q_y) * sin_y.unsqueeze(-2))
    k_x_rot = (k_x * cos_x.unsqueeze(-2)) + (rotate_half(k_x) * sin_x.unsqueeze(-2))
    k_y_rot = (k_y * cos_y.unsqueeze(-2)) + (rotate_half(k_y) * sin_y.unsqueeze(-2))

    q_out = torch.cat([q_x_rot, q_y_rot], dim=-1)
    k_out = torch.cat([k_x_rot, k_y_rot], dim=-1)

    return q_out.to(orig_dtype), k_out.to(orig_dtype)


def apply_rotary_pos_emb_2d_mixed(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_mixed: torch.Tensor,
    sin_mixed: torch.Tensor,
):
    """Apply 2D mixed RoPE using rotate_half (consistent with axial/1D RoPE)."""
    orig_dtype = q.dtype
    cos_mixed = cos_mixed.to(orig_dtype)
    sin_mixed = sin_mixed.to(orig_dtype)

    seq_len = q.shape[1]
    cos_mixed = cos_mixed[:, :seq_len, :]
    sin_mixed = sin_mixed[:, :seq_len, :]

    cos_mixed = cos_mixed.permute(1, 0, 2)
    sin_mixed = sin_mixed.permute(1, 0, 2)

    q_rot = (q * cos_mixed.unsqueeze(0)) + (rotate_half(q) * sin_mixed.unsqueeze(0))
    k_rot = (k * cos_mixed.unsqueeze(0)) + (rotate_half(k) * sin_mixed.unsqueeze(0))

    return q_rot.to(orig_dtype), k_rot.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class RotaryEmbedding2DAxial(nn.Module):
    """2D Axial Rotary Position Embedding.

    Applies RoPE separately to x and y dimensions, then concatenates.
    For sequence position i mapping to grid (row, col):
      - row = i // grid_size
      - col = i % grid_size
    Head dimension is split: first half for x-axis, second half for y-axis.

    CRITICAL: No clamp on t_y - allow row 30 (phantom row) so soft prompts at
    indices 900-902 are distinguishable from grid pixels at (29, 0-2).
    """

    def __init__(self, dim, max_seq_len, grid_size, base, device=None):
        super().__init__()
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim // 2, 2, dtype=torch.float32, device=device)
                / (dim // 2)
            )
        )
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        t_x = t % grid_size
        t_y = torch.div(t, grid_size, rounding_mode="floor")

        freqs_x = torch.outer(t_x, inv_freq)
        freqs_y = torch.outer(t_y, inv_freq)

        emb_x = torch.cat((freqs_x, freqs_x), dim=-1)
        emb_y = torch.cat((freqs_y, freqs_y), dim=-1)

        self.cos_x_cached = nn.Buffer(emb_x.cos(), persistent=False)
        self.sin_x_cached = nn.Buffer(emb_x.sin(), persistent=False)
        self.cos_y_cached = nn.Buffer(emb_y.cos(), persistent=False)
        self.sin_y_cached = nn.Buffer(emb_y.sin(), persistent=False)

    def forward(self):
        return self.cos_x_cached, self.sin_x_cached, self.cos_y_cached, self.sin_y_cached


class RotaryEmbedding2DMixed(nn.Module):
    """2D Mixed Rotary Position Embedding with learnable per-head rotations.

    Each attention head has learnable rotation angles that mix x/y frequencies.
    Uses rotate_half convention (not complex numbers) for consistency with
    RotaryEmbedding2DAxial and standard 1D RoPE.

    CRITICAL: Unlike Axial (which splits head_dim into x/y halves), Mixed applies
    to the FULL head_dim - use full 'dim' for inv_freq calculation.

    CRITICAL: No clamp on t_y - allow row 30 (phantom row) so soft prompts at
    indices 900-902 are distinguishable from grid pixels.
    """

    def __init__(self, dim, max_seq_len, grid_size, num_heads, base, device=None):
        super().__init__()
        self.angles = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        t_x = t % grid_size
        t_y = torch.div(t, grid_size, rounding_mode="floor")

        self.freqs_x_base = nn.Buffer(
            torch.outer(t_x, inv_freq), persistent=False
        )
        self.freqs_y_base = nn.Buffer(
            torch.outer(t_y, inv_freq), persistent=False
        )

    def forward(self):
        num_heads = self.angles.shape[0]

        cos_mixed_list = []
        sin_mixed_list = []

        for h in range(num_heads):
            angle = self.angles[h]
            freqs_mixed = (
                self.freqs_x_base * torch.cos(angle)
                + self.freqs_y_base * torch.sin(angle)
            )
            freqs_mixed_full = torch.cat([freqs_mixed, freqs_mixed], dim=-1)

            cos_mixed_list.append(freqs_mixed_full.cos())
            sin_mixed_list.append(freqs_mixed_full.sin())

        cos_mixed = torch.stack(cos_mixed_list, dim=0)
        sin_mixed = torch.stack(sin_mixed_list, dim=0)

        return cos_mixed, sin_mixed


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)

        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        rope_mode = kwargs.get("rope_mode")
        if rope_mode == "1d":
            cos, sin = kwargs["cos_sin"]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        elif rope_mode == "2d-axial":
            cos_x, sin_x, cos_y, sin_y = kwargs["cos_sin_2d"]
            query, key = apply_rotary_pos_emb_2d_axial(
                query, key, cos_x, sin_x, cos_y, sin_y
            )
        elif rope_mode == "2d-mixed":
            cos_mixed, sin_mixed = kwargs["cos_sin_mixed"]
            query, key = apply_rotary_pos_emb_2d_mixed(
                query, key, cos_mixed, sin_mixed
            )

        # flash attn (if available), otherwise standard scaled dot-product attention
        attn_output = None
        if flash_attn_func is not None:
            try:
                attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
                if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
                    attn_output = attn_output[0]
            except Exception:
                attn_output = None

        if attn_output is None:
            # Prefer PyTorch SDPA fallback to avoid materializing [B,H,S,S] scores.
            # This uses memory-efficient kernels on modern CUDA GPUs (including L40S).
            q = query.transpose(1, 2)  # [bs, heads, seq, dim]
            k = key.transpose(1, 2)    # [bs, heads, seq, dim]
            v = value.transpose(1, 2)  # [bs, heads, seq, dim]

            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=self.causal,
            )
            attn_output = attn_output.transpose(1, 2)  # [bs, seq, heads, dim]

        # Use reshape rather than view to handle non-contiguous tensors from attention kernels.
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
