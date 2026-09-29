"""Pre-norm encoder block: attention residual, then SwiGLU residual."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from buka_rs.attention import EncoderAttention
from buka_rs.config import OneConfig


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(mean_square + eps)
    return (x.float() * inv * weight.float()).to(dtype=x.dtype)


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    return F.silu(gate) * up


def mean_pool(x: Tensor, mask: Tensor) -> Tensor:
    """Average token states, ignoring pad. `mask` is `(batch, time)`."""
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, config: OneConfig) -> None:
        super().__init__()
        d = config.d_model
        hidden = config.d_ff
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class EncoderBlock(nn.Module):
    def __init__(self, config: OneConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = EncoderAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, key_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), key_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x
