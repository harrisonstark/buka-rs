"""AdamW on the three heads. One step is the lesson; `run` is the capstone."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from buka_rs.config import OneConfig
from buka_rs.data import Example, collate
from buka_rs.decisions import fit_temperature
from buka_rs.model import DecisionLogits, OneModel


def decision_loss(logits: DecisionLogits, batch: dict[str, Tensor]) -> Tensor:
    topic = F.cross_entropy(logits.topic, batch["topic"])
    urgency = F.cross_entropy(logits.urgency, batch["urgency"])
    reply = F.binary_cross_entropy_with_logits(logits.reply, batch["reply"])
    return topic + urgency + reply


def train_step(model: OneModel, batch: dict[str, Tensor], optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = decision_loss(model(batch["input_ids"], batch["mask"]), batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(
    model: OneModel,
    examples: list[Example],
    seq_len: int,
    device: torch.device,
    temperatures: dict[str, float] | None = None,
) -> dict[str, float]:
    model.eval()
    urgency_t = 1.0
    if temperatures:
        urgency_t = max(float(temperatures.get("urgency", 1.0)), 1e-3)
    topic_hit = 0
    reply_hit = 0
    urgency_abs = 0.0
    count = 0
    for start in range(0, len(examples), 32):
        chunk = examples[start : start + 32]
        batch = collate(chunk, model.config.topics, seq_len, model.config.pad_id, device)
        out = model(batch["input_ids"], batch["mask"])
        topic_hit += int((out.topic.argmax(dim=-1) == batch["topic"]).sum().item())
        reply_pred = (torch.sigmoid(out.reply) >= 0.5).to(batch["reply"].dtype)
        reply_hit += int((reply_pred == batch["reply"]).sum().item())
        probs = torch.softmax(out.urgency.float() / urgency_t, dim=-1)
        levels = torch.arange(probs.size(-1), device=device, dtype=probs.dtype)
        expected = (probs * levels).sum(dim=-1)
        urgency_abs += float((expected - batch["urgency"].float()).abs().sum().item())
        count += len(chunk)
    return {
        "topic_acc": topic_hit / count,
        "reply_acc": reply_hit / count,
        "urgency_mae": urgency_abs / count,
    }


def run(
    config: OneConfig,
    examples: list[Example],
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    seed: int,
    device: torch.device,
    log_every: int = 20,
) -> tuple[OneModel, dict[str, float]]:
    if seq_len > config.max_seq_len:
        raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {config.max_seq_len}")
    torch.manual_seed(seed)
    model = OneModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    first = None
    last = 0.0
    for step in range(1, steps + 1):
        rows = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
        batch = collate(rows, config.topics, seq_len, config.pad_id, device)
        last = train_step(model, batch, optimizer)
        if first is None:
            first = last
        if step == 1 or step == steps or (log_every and step % log_every == 0):
            print(f"step {step}/{steps} loss {last:.4f}")
    metrics = evaluate(model, examples, seq_len, device)
    metrics["first_loss"] = float(first if first is not None else last)
    metrics["last_loss"] = last
    metrics["params"] = float(model.num_parameters())
    return model, metrics


def collect_logits(
    model: OneModel,
    examples: list[Example],
    seq_len: int,
    device: torch.device,
    which: str,
) -> tuple[Tensor, Tensor]:
    """Stack `which` head logits (`topic` or `urgency`) and integer labels."""
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for start in range(0, len(examples), 32):
            chunk = examples[start : start + 32]
            batch = collate(chunk, model.config.topics, seq_len, model.config.pad_id, device)
            out = model(batch["input_ids"], batch["mask"])
            logits.append(getattr(out, which).detach().float().cpu())
            labels.append(batch[which].detach().cpu())
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def calibrate(
    model: OneModel,
    examples: list[Example],
    seq_len: int,
    device: torch.device,
) -> dict[str, float]:
    """Temperature for the two categorical heads, fit on these rows."""
    temps = {}
    for name in ("topic", "urgency"):
        logits, labels = collect_logits(model, examples, seq_len, device, name)
        temps[name] = fit_temperature(logits, labels)
    return temps


def save_checkpoint(
    path: str | Path,
    model: OneModel,
    temperatures: dict[str, float] | None = None,
    seq_len: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "temperatures": temperatures or {"topic": 1.0, "urgency": 1.0},
            "seq_len": int(seq_len if seq_len is not None else model.config.max_seq_len),
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[OneModel, dict[str, float], int]:
    blob = torch.load(path, map_location=device, weights_only=True)
    config = OneConfig.from_dict(blob["config"])
    model = OneModel(config).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    temps = blob.get("temperatures") or {"topic": 1.0, "urgency": 1.0}
    seq_len = int(blob.get("seq_len") or config.max_seq_len)
    return model, {key: float(value) for key, value in temps.items()}, seq_len
