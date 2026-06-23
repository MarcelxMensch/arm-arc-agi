"""Embedding modules for the HRM architecture.

Contains:
- CastedSparseEmbedding: Legacy sparse embedding with local gradient tracking.
- CompositeSparseEmbedding: Legacy 3-token softprompt (task + transform + color).
- GridEncoder: CNN that processes demo grid pairs into dense task tokens.
- GridEncoderSoftPrompt: New 13-token softprompt (2 task + 1 transform + 10 color).
- CastedSparseEmbeddingSignSGD_Distributed: Distributed SignSGD optimizer for sparse embeddings.
"""

from typing import Union, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, ParamsT

from .common import trunc_normal_init_


# ---------------------------------------------------------------------------
# Legacy sparse embedding
# ---------------------------------------------------------------------------

class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )

        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.weights[inputs].to(self.cast_to)

        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)

        return self.local_weights.to(self.cast_to)


class CompositeSparseEmbedding(nn.Module):
    """Legacy 3-token composite soft prompt: [task, transform, color]."""

    def __init__(
        self,
        num_task_embeddings: int,
        num_transform_embeddings: int,
        num_color_embeddings: int,
        embedding_dim: int,
        batch_size: int,
        init_std: float,
        cast_to: torch.dtype,
        puzzle_softprompt_len: int = 3,
    ):
        super().__init__()
        self.cast_to = cast_to
        self.puzzle_softprompt_len = puzzle_softprompt_len
        self.embedding_dim = embedding_dim

        self.task_emb = CastedSparseEmbedding(
            num_task_embeddings, embedding_dim, batch_size, init_std, cast_to
        )
        self.transform_emb = CastedSparseEmbedding(
            num_transform_embeddings, embedding_dim, batch_size, init_std, cast_to
        )
        self.color_emb = CastedSparseEmbedding(
            num_color_embeddings, embedding_dim, batch_size, init_std, cast_to
        )

    def forward(
        self,
        task_ids: torch.Tensor,
        transform_ids: torch.Tensor,
        color_ids: torch.Tensor,
    ) -> torch.Tensor:
        task_emb = self.task_emb(task_ids)
        transform_emb = self.transform_emb(transform_ids)
        color_emb = self.color_emb(color_ids)

        composite_emb = torch.stack([task_emb, transform_emb, color_emb], dim=1)
        batch_size = composite_emb.shape[0]
        composite_emb = composite_emb.reshape(batch_size, self.puzzle_softprompt_len * self.embedding_dim)
        return composite_emb.to(self.cast_to)


# ---------------------------------------------------------------------------
# Grid Encoder: CNN that converts demo grid pairs into dense task tokens
# ---------------------------------------------------------------------------

class GridEncoder(nn.Module):
    """Encode ARC demonstration grid pairs into dense task tokens via a CNN.

    ARC grids contain categorical colour IDs (0-9, plus PAD=0 and EOS=1 after
    VOCAB_OFFSET encoding gives 0-11).  A dedicated ``nn.Embedding`` maps each
    cell token to a dense vector before the convolutional stack, preserving the
    categorical nature and ensuring PAD cells contribute zero features.

    Architecture
    ------------
    1. Cell embedding: ``nn.Embedding(12, embed_channels, padding_idx=0)``
    2. Stack input+output embeddings per demo: ``(B*N, 2*embed_channels, 30, 30)``
    3. 3x Conv2d with GELU + BatchNorm + stride-2 downsampling
    4. AdaptiveAvgPool2d(1) -> per-pair feature (128-d)
    5. Masked mean-pool across demos -> (B, 128)
    6. Linear projection -> (B, 2 * hidden_size) -> (B, 2, hidden_size)
    """

    EMBED_CHANNELS = 16
    CNN_CHANNELS = [64, 128, 128]
    NUM_CELL_TOKENS = 12  # PAD(0), EOS(1), colors(2-11)

    def __init__(self, hidden_size: int, max_demos: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_demos = max_demos

        # Categorical cell embedding (PAD -> zero vector)
        self.cell_emb = nn.Embedding(
            self.NUM_CELL_TOKENS, self.EMBED_CHANNELS, padding_idx=0
        )

        in_ch = 2 * self.EMBED_CHANNELS  # input grid + output grid channels stacked
        layers = []
        for out_ch in self.CNN_CHANNELS:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.cnn = nn.Sequential(*layers)

        feat_dim = self.CNN_CHANNELS[-1]
        self.proj = nn.Linear(feat_dim, 2 * hidden_size)

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        num_demos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            demo_inputs:  (B, N, H, W) long tensor, encoded grid tokens.
            demo_outputs: (B, N, H, W) long tensor, encoded grid tokens.
            num_demos:    (B,) int tensor, number of valid demos per example.

        Returns:
            (B, 2, hidden_size) -- two task tokens.
        """
        B, N, H, W = demo_inputs.shape

        # Embed cells: (B, N, H, W) -> (B, N, H, W, C)
        inp_emb = self.cell_emb(demo_inputs.clamp(0, self.NUM_CELL_TOKENS - 1))
        out_emb = self.cell_emb(demo_outputs.clamp(0, self.NUM_CELL_TOKENS - 1))

        # Stack input+output channels: (B, N, H, W, 2C)
        pair_emb = torch.cat([inp_emb, out_emb], dim=-1)
        # Reshape for Conv2d: (B*N, 2C, H, W)
        pair_emb = pair_emb.reshape(B * N, H, W, 2 * self.EMBED_CHANNELS)
        pair_emb = pair_emb.permute(0, 3, 1, 2).contiguous()

        # CNN feature extraction: (B*N, feat_dim, 1, 1)
        features = self.cnn(pair_emb).squeeze(-1).squeeze(-1)  # (B*N, feat_dim)
        features = features.reshape(B, N, -1)  # (B, N, feat_dim)

        # Masked mean-pool across demos
        mask = torch.arange(N, device=num_demos.device).unsqueeze(0) < num_demos.unsqueeze(1)  # (B, N)
        mask_f = mask.unsqueeze(-1).float()  # (B, N, 1)
        pooled = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (B, feat_dim)

        # Project to 2 task tokens
        task_vec = self.proj(pooled)  # (B, 2*hidden_size)
        return task_vec.reshape(B, 2, self.hidden_size)


# ---------------------------------------------------------------------------
# GridEncoderSoftPrompt: 13-token softprompt
# [task_1, task_2, transform, color_0, ..., color_9]
# ---------------------------------------------------------------------------

class GridEncoderSoftPrompt(nn.Module):
    """New-style softprompt: 2 CNN task tokens + 1 learnable transform + 10 frozen one-hot colors.

    Token layout (13 tokens, each of dimension ``hidden_size``):
    - Positions 0-1: Task tokens from ``GridEncoder`` (learned CNN).
    - Position 2:    Transform token from ``CastedSparseEmbedding`` (learned, 8 entries).
    - Positions 3-12: One-hot colour mapping tokens (frozen, no gradient).
      Each colour token ``i`` has a 1.0 at dimension ``color_map[i]`` within the
      first 10 dims, and zeros elsewhere.
    """

    SOFTPROMPT_LEN = 13  # 2 task + 1 transform + 10 color

    def __init__(
        self,
        hidden_size: int,
        batch_size: int,
        init_std: float,
        cast_to: torch.dtype,
        max_demos: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cast_to = cast_to

        self.grid_encoder = GridEncoder(hidden_size, max_demos=max_demos)
        self.transform_emb = CastedSparseEmbedding(
            num_embeddings=8,
            embedding_dim=hidden_size,
            batch_size=batch_size,
            init_std=init_std,
            cast_to=cast_to,
        )

    def _build_color_tokens(self, color_maps: torch.Tensor) -> torch.Tensor:
        """Build 10 frozen one-hot colour tokens from the colour map.

        Args:
            color_maps: (B, 10) int tensor. ``color_maps[b, i]`` is the colour
                that original colour ``i`` was remapped to.

        Returns:
            (B, 10, hidden_size) float tensor (no gradient).
        """
        B = color_maps.shape[0]
        tokens = torch.zeros(B, 10, self.hidden_size, device=color_maps.device, dtype=self.cast_to)
        # For each colour slot i, place a 1.0 at position color_maps[b, i]
        idx = color_maps.clamp(0, 9).long()  # (B, 10)
        tokens.scatter_(2, idx.unsqueeze(-1), 1.0)
        return tokens

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        num_demos: torch.Tensor,
        transform_ids: torch.Tensor,
        color_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            (B, 13, hidden_size) tensor -- the full softprompt sequence.
        """
        # 2 task tokens from CNN (B, 2, H)
        task_tokens = self.grid_encoder(demo_inputs, demo_outputs, num_demos).to(self.cast_to)

        # 1 transform token (B, 1, H)
        transform_token = self.transform_emb(transform_ids).unsqueeze(1)

        # 10 frozen colour tokens (B, 10, H) -- detach to block gradients
        with torch.no_grad():
            color_tokens = self._build_color_tokens(color_maps)

        # Concat: [task_1, task_2, transform, color_0, ..., color_9]
        return torch.cat([task_tokens, transform_token, color_tokens], dim=1)


# ---------------------------------------------------------------------------
# Distributed SignSGD optimizer for sparse embeddings
# ---------------------------------------------------------------------------

class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        world_size: int,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            world_size=world_size
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):  # type: ignore
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
                _sparse_emb_signsgd_dist(
                    local_weights_grad,
                    local_ids,
                    weights,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    world_size=group["world_size"]
                )


def _sparse_emb_signsgd_dist(
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
        all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * N, dtype=local_ids.dtype, device=local_ids.device)

        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids, local_ids)

    grad_ids, inv = all_ids.unique(return_inverse=True)

    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    p = weights[grad_ids]
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

    weights[grad_ids] = p
