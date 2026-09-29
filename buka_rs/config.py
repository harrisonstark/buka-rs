"""Widths for a from-scratch decision model.

The 8 GB card is not the limit. Labeled chat is. See docs/model_size.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


TOPICS: tuple[str, ...] = ("chat", "ops", "learning")
URGENCY: tuple[str, ...] = ("calm", "soon", "now")


@dataclass
class OneConfig:
    """Encoder plus three fixed heads: topic (choice), urgency (score), reply (noul)."""

    name: str = "tiny"
    vocab_size: int = 257
    pad_id: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384
    max_seq_len: int = 128
    rms_norm_eps: float = 1e-6
    topics: tuple[str, ...] = TOPICS
    urgency: tuple[str, ...] = URGENCY

    def __post_init__(self) -> None:
        if isinstance(self.topics, list):
            self.topics = tuple(self.topics)
        if isinstance(self.urgency, list):
            self.urgency = tuple(self.urgency)
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.pad_id >= self.vocab_size:
            raise ValueError("pad_id must be inside the embedding table")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OneConfig:
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def tiny() -> OneConfig:
    """About 0.9M parameters. Default for the sample set and a first private label set."""
    return OneConfig(name="tiny")


def small() -> OneConfig:
    """About 5M parameters. Use when you have more labeled states than the sample file."""
    return OneConfig(
        name="small",
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=768,
        max_seq_len=192,
    )


def stretch() -> OneConfig:
    """About 14M parameters. Still a small float32 model on an 8 GB card."""
    return OneConfig(
        name="stretch",
        d_model=384,
        n_layers=8,
        n_heads=8,
        d_ff=1024,
        max_seq_len=192,
    )


PRESETS = {"tiny": tiny, "small": small, "stretch": stretch}


def config_from_preset(name: str) -> OneConfig:
    if name not in PRESETS:
        known = ", ".join(PRESETS)
        raise KeyError(f"unknown preset {name!r}; choose from {known}")
    return PRESETS[name]()
