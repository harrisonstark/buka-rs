"""One forward pass: bytes in, three decision heads out.

The heads are fixed at training time (topic, urgency, needs-a-reply). Jev can
answer questions you write in the request because it is a frontier model.
A model this small only answers the questions its heads were trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from buka_rs.block import EncoderBlock, RMSNorm, mean_pool
from buka_rs.config import OneConfig


@dataclass
class DecisionLogits:
    topic: Tensor
    urgency: Tensor
    reply: Tensor


class OneModel(nn.Module):
    def __init__(self, config: OneConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.tok = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_id)
        self.pos = nn.Embedding(config.max_seq_len, d)
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(d, config.rms_norm_eps)
        self.topic = nn.Linear(d, len(config.topics), bias=False)
        self.urgency = nn.Linear(d, len(config.urgency), bias=False)
        self.reply = nn.Linear(d, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.normal_(param, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def forward(self, input_ids: Tensor, mask: Tensor) -> DecisionLogits:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"sequence {input_ids.size(1)} exceeds max_seq_len {self.config.max_seq_len}"
            )
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.tok(input_ids) + self.pos(positions)[None, :, :]
        for block in self.blocks:
            x = block(x, mask)
        pooled = mean_pool(self.norm(x), mask)
        return DecisionLogits(
            topic=self.topic(pooled),
            urgency=self.urgency(pooled),
            reply=self.reply(pooled).squeeze(-1),
        )
