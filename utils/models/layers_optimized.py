"""Drop-in optimized replacements for layers.py operations.

Two optimization modes:
1. With torch.compile: Only precast weights to bf16 (compile handles fusion)
2. Without torch.compile: Full Triton kernels + bf16 weights (we handle fusion)
"""
import os
import torch
from torch import nn
import torch.nn.functional as F

import utils.models.layers as layers


# ============================================================================
# Optimized CastedLinear - eliminate dtype casting overhead
# ============================================================================

class OptimizedCastedLinear(nn.Module):
    """Linear layer that stores weight in compute dtype to avoid runtime casting."""

    def __init__(self, original: layers.CastedLinear, compute_dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(original.weight.data.to(compute_dtype))
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.to(compute_dtype))
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class OptimizedCastedEmbedding(nn.Module):
    """Embedding that stores weight in target dtype to avoid runtime casting."""

    def __init__(self, original: layers.CastedEmbedding):
        super().__init__()
        self.embedding_weight = nn.Parameter(
            original.embedding_weight.data.to(original.cast_to)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight)


# ============================================================================
# Optimized SwiGLU with Triton kernel (for no-compile mode)
# ============================================================================

class TritonSwiGLU(nn.Module):
    """SwiGLU with fused Triton activation kernel."""

    def __init__(self, original: layers.SwiGLU):
        super().__init__()
        self.gate_up_proj = original.gate_up_proj
        self.down_proj = original.down_proj
        self.conv = original.conv

    def forward(self, x):
        from utils.models.triton_kernels import triton_swiglu
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        h = triton_swiglu(gate, up)
        if self.conv is not None:
            h = self.conv(h.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        return self.down_proj(h)


def optimize_model(model: nn.Module, compute_dtype=torch.bfloat16) -> nn.Module:
    """Apply optimizations to the model.

    When torch.compile is active: only precast weights to bf16.
    When torch.compile is disabled: also apply Triton kernels for fusion.
    """
    use_triton_kernels = "DISABLE_COMPILE" in os.environ

    if use_triton_kernels:
        # Full optimization: Triton kernels + bf16 weights
        from utils.models.triton_kernels import triton_rms_norm, triton_rope

        # Replace SwiGLU with Triton version
        _replace_modules(model, layers.SwiGLU, lambda m: TritonSwiGLU(m))

        # Patch RMSNorm
        layers.rms_norm = lambda hidden_states, variance_epsilon: triton_rms_norm(hidden_states, eps=variance_epsilon)

        # Patch RoPE
        layers.apply_rotary_pos_emb = triton_rope

        print("[TRITON] Applied Triton kernels: RMSNorm, SwiGLU, RoPE")

    # Always precast weights
    _replace_modules(model, layers.CastedLinear,
                     lambda m: OptimizedCastedLinear(m, compute_dtype))
    _replace_modules(model, layers.CastedEmbedding,
                     lambda m: OptimizedCastedEmbedding(m))
    print(f"[OPT] Precast all weights to {compute_dtype}")

    return model


def _replace_modules(model: nn.Module, target_class, replacement_fn):
    """Recursively replace all instances of target_class in model."""
    for name, module in model.named_children():
        if isinstance(module, target_class):
            setattr(model, name, replacement_fn(module))
        else:
            _replace_modules(module, target_class, replacement_fn)
