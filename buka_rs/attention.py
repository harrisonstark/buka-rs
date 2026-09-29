"""Bidirectional attention. Every real token can see the whole state.

Buka's causal mask exists so a language model cannot read the token it is
about to predict. A System One model does not predict the next byte. The
state is already complete, so the triangle mask comes off. Padding positions
are still hidden, or they would dilute the pool.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from buka_rs.config import OneConfig


def bidirectional_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    key_mask: Tensor | None = None,
) -> Tensor:
    """Scaled dot-product attention.

    `q`, `k`, `v` are `(batch, heads, time, head_dim)`.
    `key_mask` is `(batch, time)` with 1 for real tokens and 0 for pad.
    """
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if key_mask is not None:
        keep = key_mask.to(dtype=scores.dtype)
        scores = scores + (1.0 - keep)[:, None, None, :] * -1.0e9
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


class EncoderAttention(nn.Module):
    def __init__(self, config: OneConfig) -> None:
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: Tensor, key_mask: Tensor | None = None) -> Tensor:
        batch, time, _ = x.shape
        q = self.q(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        mixed = bidirectional_attention(q, k, v, key_mask)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, time, -1)
        return self.o(mixed)
