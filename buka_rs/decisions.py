"""Typed answers: choice, score, noul, and a confidence summary.

Jev returns these from a hosted model. The formulas here are the ones you can
implement and test. Confidence is 1 minus the entropy of the distribution,
divided by log(number of options): 1 when the mass sits on one option, 0 when
it is uniform. TypeSafe has not published their exact confidence formula.
Noul is only P(yes); Jev does not attach a second confidence field to it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def choice_probs(logits: Tensor) -> Tensor:
    """Softmax over the last dimension. Rows sum to 1."""
    return torch.softmax(logits.float(), dim=-1)


def noul_prob(logit: Tensor) -> Tensor:
    """Probability that the yes/no statement is true."""
    return torch.sigmoid(logit.float())


def confidence(probs: Tensor) -> Tensor:
    """How peaked `probs` is, in `[0, 1]`.

    `probs` is a distribution on the last axis (it should already sum to 1).
    """
    probs = probs.float()
    k = probs.size(-1)
    if k < 2:
        return torch.ones(probs.shape[:-1], device=probs.device, dtype=probs.dtype)
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return (1.0 - entropy / math.log(k)).clamp(0.0, 1.0)


def choice_answer(
    logits: Tensor,
    names: tuple[str, ...] | list[str],
    temperature: float = 1.0,
) -> dict:
    """One choice question: winning label, full distribution, confidence."""
    temperature = max(float(temperature), 1e-3)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    index = int(torch.argmax(probs).item())
    return {
        "choice": names[index],
        "probabilities": {str(name): float(probs[i]) for i, name in enumerate(names)},
        "confidence": float(confidence(probs).item()),
    }


def score_answer(
    logits: Tensor,
    legend: tuple[str, ...] | list[str],
    temperature: float = 1.0,
) -> dict:
    """Ordered levels. `score` is the probability-weighted level index."""
    temperature = max(float(temperature), 1e-3)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    levels = torch.arange(probs.numel(), device=probs.device, dtype=probs.dtype)
    return {
        "score": float((probs * levels).sum().item()),
        "probabilities": {str(i): float(probs[i]) for i in range(probs.numel())},
        "legend": list(legend),
        "confidence": float(confidence(probs).item()),
    }


def noul_answer(logit: Tensor) -> dict:
    """P(yes). No separate confidence field, matching the Noul primitive."""
    return {"noul": float(torch.sigmoid(logit.float()).item())}


def fit_temperature(
    logits: Tensor,
    labels: Tensor,
    grid: list[float] | None = None,
) -> float:
    """Pick a temperature that lowers next-class cross-entropy on held-out rows.

    Divide logits by T before the softmax. T > 1 softens an over-confident
    model. This is temperature scaling, not RLCD (Jev's training method, which
    is not published in enough detail to implement from the announcement).
    """
    if logits.numel() == 0:
        return 1.0
    candidates = grid or [0.5, 0.7, 1.0, 1.3, 1.7, 2.2, 3.0]
    best_t = 1.0
    best_nll = float("inf")
    for temperature in candidates:
        nll = float(F.cross_entropy(logits.float() / temperature, labels).item())
        if nll < best_nll:
            best_t = float(temperature)
            best_nll = nll
    return best_t
