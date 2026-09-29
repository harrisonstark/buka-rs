"""Turn one state string into the three trained answers."""

from __future__ import annotations

import torch

from buka_rs.data import clean_state, encode_state
from buka_rs.decisions import choice_answer, noul_answer, score_answer
from buka_rs.model import OneModel


@torch.no_grad()
def decide(
    model: OneModel,
    state: str,
    temperatures: dict[str, float] | None = None,
    seq_len: int | None = None,
) -> dict:
    model.eval()
    temps = {"topic": 1.0, "urgency": 1.0}
    if temperatures:
        temps.update(temperatures)
    cleaned = clean_state(state)
    if cleaned is None:
        raise ValueError("state is empty or only a URL")
    length = seq_len if seq_len is not None else model.config.max_seq_len
    ids, mask = encode_state(cleaned, length, model.config.pad_id)
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask_t = torch.tensor([mask], dtype=torch.float32, device=device)
    out = model(input_ids, mask_t)
    return {
        "answers": {
            "topic": choice_answer(out.topic[0], model.config.topics, temps["topic"]),
            "urgency": score_answer(out.urgency[0], model.config.urgency, temps["urgency"]),
            "needs_reply": noul_answer(out.reply[0]),
        }
    }


def response_payload(topic: dict, urgency: dict, reply: dict) -> dict:
    """The shape the local page and `/decide` return."""
    return {"answers": {"topic": topic, "urgency": urgency, "needs_reply": reply}}
