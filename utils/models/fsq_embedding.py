"""Factored FSQ Embedding: structured discrete codes for puzzle representations.

Replaces the flat CastedSparseEmbedding with a factored discrete code table
using Finite Scalar Quantization (Mentzer et al., ICLR 2024). Each puzzle's
embedding is split into M blocks, and each dimension is independently rounded
to one of L discrete levels. The implicit codebook is the Cartesian product
of per-dimension level sets — no learned codebook parameters, no collapse.

Reference: arxiv.org/abs/2309.15505
"""
from typing import Union

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, ParamsT

from utils.models.common import trunc_normal_init_


def _fsq_quantize(z: torch.Tensor, levels: int, temperature: float = 1.0) -> torch.Tensor:
    """Quantize each element to one of `levels` discrete values in [-1, 1].

    Uses straight-through estimator: forward rounds, backward passes gradients through.
    """
    half_width = (levels - 1) / 2
    # Bound to [-1, 1] via tanh
    z_bounded = torch.tanh(z / temperature)
    # Scale to [0, levels-1], round, scale back to [-1, 1]
    z_scaled = (z_bounded + 1) * half_width  # [0, levels-1]
    z_rounded = torch.round(z_scaled)
    z_quantized = z_rounded / half_width - 1  # [-1, 1]
    # Straight-through: gradient flows through as if no rounding happened
    return z_bounded + (z_quantized - z_bounded).detach()


class FactoredFSQEmbedding(nn.Module):
    """Per-puzzle embedding with factored FSQ discretization.

    Each puzzle has a learnable continuous vector of size (n_blocks * dims_per_block).
    During forward, this is split into n_blocks groups, each FSQ-quantized to
    `levels` discrete values per dimension. The quantized vector is then projected
    to the output embedding dimension.

    Args:
        num_embeddings: Number of puzzle identifiers.
        embedding_dim: Output dimension (typically hidden_size).
        batch_size: For sparse optimizer compatibility.
        n_blocks: Number of independent factor blocks (M).
        dims_per_block: Dimensions per block.
        levels: Number of discrete levels per dimension.
        init_std: Initialization standard deviation.
        cast_to: Output dtype.
        temperature: FSQ temperature (higher = softer quantization). Anneal to 1.0.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        batch_size: int,
        n_blocks: int = 6,
        dims_per_block: int = 2,
        levels: int = 5,
        init_std: float = 0.0,
        cast_to: torch.dtype = torch.bfloat16,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.cast_to = cast_to
        self.n_blocks = n_blocks
        self.dims_per_block = dims_per_block
        self.levels = levels
        self.temperature = temperature
        self.code_dim = n_blocks * dims_per_block
        self.embedding_dim = embedding_dim

        # Learnable continuous codes per puzzle (pre-quantization)
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, self.code_dim)), std=init_std),
            persistent=True,
        )

        # Projection from quantized code space to embedding space
        self.code_proj = nn.Linear(self.code_dim, embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.code_proj.weight)

        # Sparse optimizer compatibility buffers
        self.local_weights = nn.Buffer(
            torch.zeros(batch_size, self.code_dim, requires_grad=True), persistent=False
        )
        self.local_ids = nn.Buffer(
            torch.zeros(batch_size, dtype=torch.int32), persistent=False
        )

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        return _fsq_quantize(z, self.levels, self.temperature)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            z_continuous = self.weights[inputs]
            z_quantized = self._quantize(z_continuous)
            return self.code_proj(z_quantized).to(self.cast_to)

        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)

        z_quantized = self._quantize(self.local_weights)
        return self.code_proj(z_quantized).to(self.cast_to)

    def get_codes(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return discrete codes for analysis (no projection)."""
        z_continuous = self.weights[inputs]
        z_bounded = torch.tanh(z_continuous / self.temperature)
        half_width = (self.levels - 1) / 2
        codes = torch.round((z_bounded + 1) * half_width).long()
        return codes.view(-1, self.n_blocks, self.dims_per_block)

    def codebook_utilization(self) -> dict:
        """Compute utilization statistics for logging."""
        all_codes = self.get_codes(torch.arange(self.weights.shape[0]))
        stats = {}
        for b in range(self.n_blocks):
            block_codes = all_codes[:, b, :]
            # Pack dims into single code index
            multipliers = torch.tensor([self.levels ** i for i in range(self.dims_per_block)], device=all_codes.device)
            packed = (block_codes * multipliers).sum(dim=-1)
            n_unique = packed.unique().numel()
            n_possible = self.levels ** self.dims_per_block
            stats[f"block_{b}_utilization"] = n_unique / n_possible
            stats[f"block_{b}_unique_codes"] = n_unique
        return stats


class FactoredFSQEmbeddingSignSGD_Distributed(Optimizer):
    """SignSGD optimizer for FactoredFSQEmbedding, compatible with distributed training.

    Same interface as CastedSparseEmbeddingSignSGD_Distributed — operates on
    the continuous pre-quantization weights.
    """

    def __init__(
        self,
        params: ParamsT,
        world_size: int,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, world_size=world_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            local_weights_grad = None
            local_ids = None
            weights = None

            assert len(group["params"]) == 3
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                else:
                    assert False

            assert local_ids is not None
            assert weights is not None

            if local_weights_grad is not None:
                _fsq_sparse_signsgd_dist(
                    local_weights_grad,
                    local_ids,
                    weights,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    world_size=group["world_size"],
                )


def _fsq_sparse_signsgd_dist(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    lr: float,
    weight_decay: float,
    world_size: int,
) -> None:
    N, D = local_weights_grad.shape

    all_weights_grad = local_weights_grad
    all_ids = local_ids

    if world_size > 1:
        all_weights_grad = torch.empty(
            (world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device
        )
        all_ids = torch.empty(world_size * N, dtype=local_ids.dtype, device=local_ids.device)
        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids, local_ids)

    grad_ids, inv = all_ids.unique(return_inverse=True)
    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    p = weights[grad_ids]
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)
    weights[grad_ids] = p
