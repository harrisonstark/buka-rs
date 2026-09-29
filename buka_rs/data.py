"""Byte encoding and the labeled-state JSONL this course trains on.

Each line is one decision problem, not a chat turn to imitate:

    {"state": "...", "topic": "ops", "urgency": 2, "needs_reply": true}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class Example:
    state: str
    topic: str
    urgency: int
    needs_reply: bool


def clean_state(text: str) -> str | None:
    text = " ".join(text.split())
    if not text:
        return None
    lower = text.lower()
    if (lower.startswith("http://") or lower.startswith("https://")) and " " not in text:
        return None
    return text


def encode_state(text: str, max_len: int, pad_id: int = 256) -> tuple[list[int], list[float]]:
    """UTF-8 bytes, then pad. A string longer than `max_len` keeps its last bytes."""
    ids = list(text.encode("utf-8"))
    if len(ids) > max_len:
        ids = ids[-max_len:]
    if not ids:
        ids = [ord(" ")]
    width = len(ids)
    pad = max_len - width
    return ids + [pad_id] * pad, [1.0] * width + [0.0] * pad


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"needs_reply must be a bool, got {value!r}")


def load_examples(path: str | Path, topics: tuple[str, ...], n_urgency: int) -> list[Example]:
    rows: list[Example] = []
    # utf-8-sig drops a Notepad BOM so the first line still parses.
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        state = clean_state(str(raw.get("state", "")))
        if state is None:
            continue
        topic = str(raw["topic"])
        urgency = int(raw["urgency"])
        if topic not in topics:
            raise ValueError(f"{path}:{lineno}: topic {topic!r} not in {topics}")
        if not 0 <= urgency < n_urgency:
            raise ValueError(f"{path}:{lineno}: urgency {urgency} out of range")
        rows.append(
            Example(
                state=state,
                topic=topic,
                urgency=urgency,
                needs_reply=parse_bool(raw["needs_reply"]),
            )
        )
    if not rows:
        raise ValueError(f"no examples in {path}")
    return rows


def collate(
    rows: list[Example],
    topics: tuple[str, ...],
    seq_len: int,
    pad_id: int,
    device: torch.device,
) -> dict[str, Tensor]:
    ids: list[list[int]] = []
    masks: list[list[float]] = []
    for row in rows:
        encoded, mask = encode_state(row.state, seq_len, pad_id)
        ids.append(encoded)
        masks.append(mask)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "mask": torch.tensor(masks, dtype=torch.float32, device=device),
        "topic": torch.tensor([topics.index(row.topic) for row in rows], dtype=torch.long, device=device),
        "urgency": torch.tensor([row.urgency for row in rows], dtype=torch.long, device=device),
        "reply": torch.tensor(
            [1.0 if row.needs_reply else 0.0 for row in rows],
            dtype=torch.float32,
            device=device,
        ),
    }
