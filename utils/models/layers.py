from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from utils.models.common import trunc_normal_init_

# Auto-detect fused rms_norm support (PyTorch >= 2.4)
_HAS_FUSED_RMS_NORM = hasattr(F, 'rms_norm')


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
    if q.dtype != cos.dtype:
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed, k_embed


def apply_rotary_pos_emb_2d_axial(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_x: torch.Tensor,
    sin_x: torch.Tensor,
    cos_y: torch.Tensor,
    sin_y: torch.Tensor,
):
    """Apply 2D axial RoPE."""
    target_dtype = q.dtype
    if cos_x.dtype != target_dtype:
        cos_x = cos_x.to(target_dtype)
        sin_x = sin_x.to(target_dtype)
        cos_y = cos_y.to(target_dtype)
        sin_y = sin_y.to(target_dtype)

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

    return q_out, k_out


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


class Rotary2DEmbeddingAxial(nn.Module):
    """2D Axial Rotary Position Embedding.

    Applies RoPE separately to x and y dimensions, then concatenates.
    For sequence position i mapping to grid (row, col):
      - col = i % grid_size
      - row = i // grid_size
    Head dimension is split: first half for x-axis, second half for y-axis.
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


class Rotary2DEmbeddingMixed(nn.Module):
    """2D Mixed Rotary Position Embedding.

    Interleaves x and y frequencies across the full head dimension,
    rather than splitting into separate halves (as in axial).
    Even frequency indices encode x-position, odd indices encode y-position.
    Returns standard (cos, sin) tensors usable with apply_rotary_pos_emb.
    """

    def __init__(self, dim, max_seq_len, grid_size, base, device=None):
        super().__init__()
        half_dim = dim // 2  # number of frequency pairs

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                / dim
            )
        )
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        t_x = t % grid_size
        t_y = torch.div(t, grid_size, rounding_mode="floor")

        # Each has shape [max_seq_len, half_dim]
        freqs_x = torch.outer(t_x, inv_freq)
        freqs_y = torch.outer(t_y, inv_freq)

        # Interleave: even freq indices get x, odd get y
        # freqs_mixed shape: [max_seq_len, half_dim]
        freqs_mixed = torch.stack([freqs_x, freqs_y], dim=-1).reshape(max_seq_len, -1)
        # If dim is not divisible by 4, freqs_mixed may have more elements; truncate
        freqs_mixed = freqs_mixed[:, :half_dim]

        emb = torch.cat((freqs_mixed, freqs_mixed), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False,
                 use_spatial_bias=False, grid_size=30, max_seq_len=916):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.use_spatial_bias = use_spatial_bias

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

        if use_spatial_bias:
            # 4 spatial features: same_row, same_col, adjacent, log_distance
            self.spatial_weights = nn.Parameter(torch.zeros(num_heads, 4))
            # Precompute spatial feature matrices for grid positions
            self._precompute_spatial_features(grid_size, max_seq_len)

    def _precompute_spatial_features(self, grid_size, max_seq_len):
        pos = torch.arange(max_seq_len)
        rows = pos // grid_size  # (S,)
        cols = pos % grid_size   # (S,)

        # Pairwise features: (S, S)
        same_row = (rows.unsqueeze(1) == rows.unsqueeze(0)).float()
        same_col = (cols.unsqueeze(1) == cols.unsqueeze(0)).float()
        manhattan = (rows.unsqueeze(1) - rows.unsqueeze(0)).abs() + (cols.unsqueeze(1) - cols.unsqueeze(0)).abs()
        adjacent = (manhattan <= 1).float()
        log_dist = torch.log1p(manhattan.float())

        # Stack: (S, S, 4)
        spatial_features = torch.stack([same_row, same_col, adjacent, log_dist], dim=-1)
        self.spatial_features = nn.Buffer(spatial_features, persistent=False)

    def _get_spatial_bias(self, seq_len):
        # spatial_features: (max_S, max_S, 4), spatial_weights: (H, 4)
        feat = self.spatial_features[:seq_len, :seq_len]  # (S, S, 4)
        # Einsum: (H, 4) x (S, S, 4) -> (H, S, S)
        bias = torch.einsum('hf,ijf->hij', self.spatial_weights, feat)
        return bias.unsqueeze(0)  # (1, H, S, S) for broadcasting over batch

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            if len(cos_sin) == 4:
                cos_x, sin_x, cos_y, sin_y = cos_sin
                query, key = apply_rotary_pos_emb_2d_axial(query, key, cos_x, sin_x, cos_y, sin_y)
            else:
                cos, sin = cos_sin
                query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Transpose to (B, H, S, D) for scaled_dot_product_attention
        query, key, value = (t.transpose(1, 2) for t in (query, key, value))

        attn_mask = None
        if self.use_spatial_bias:
            attn_mask = self._get_spatial_bias(seq_len).to(query.dtype)

        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal, attn_mask=attn_mask)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)

class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, use_conv: bool = False, conv_kernel: int = 2):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

        # ConvSwiGLU: depthwise conv for local token mixing (URM, Gao et al. arXiv:2512.14693)
        self.conv = None
        if use_conv:
            self.conv = nn.Conv1d(inter, inter, kernel_size=conv_kernel, padding=conv_kernel - 1, groups=inter, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        h = F.silu(gate) * up
        if self.conv is not None:
            # x is (B, SeqLen, inter) -> transpose for Conv1d -> transpose back
            h = self.conv(h.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        return self.down_proj(h)

class ConvSwiGLU(nn.Module):
    """ConvSwiGLU from URM (Gao et al., arXiv:2512.14693).

    Differs from SwiGLU+conv in two ways:
    1. Applies a second SiLU activation AFTER the depthwise conv
    2. Depthwise conv uses bias=True and bfloat16 dtype
    """
    def __init__(self, hidden_size: int, expansion: float, conv_kernel: int = 2):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.dwconv = nn.Conv1d(
            in_channels=inter, out_channels=inter,
            kernel_size=conv_kernel, padding=conv_kernel // 2,
            groups=inter, bias=True,
        ).to(dtype=torch.bfloat16)
        self.act = nn.SiLU()
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x_ffn = self.act(gate) * up
        x_conv = self.dwconv(x_ffn.transpose(1, 2).to(self.dwconv.weight.dtype))
        x_conv = x_conv[..., :up.size(1)]
        x_conv = self.act(x_conv)
        x_conv = x_conv.transpose(1, 2).contiguous()
        return self.down_proj(x_conv)


if _HAS_FUSED_RMS_NORM:
    def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), eps=variance_epsilon)
else:
    def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        return hidden_states.to(input_dtype)
