"""Write notebooks/01..11. Stub, collapsed solution, check.

    python scripts/build_course_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    text = text.strip("\n")
    if not text.endswith("\n"):
        text += "\n"
    return [line + "\n" for line in text.split("\n")]


def details(solution: str) -> dict:
    body = (
        "<details>\n"
        "<summary><b>Solution</b> (try yourself first — click to expand)</summary>\n\n"
        "```python\n"
        f"{solution.rstrip()}\n"
        "```\n\n"
        "Copy into the stub cell above, then run **Check**.\n\n"
        "</details>\n"
    )
    return md(body)


def write_nb(name: str, cells: list[dict]) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    path = OUT / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))


def lesson(name: str, guide: str, stub: str, solution: str, check: str) -> None:
    write_nb(
        name,
        [
            md(guide),
            code(stub),
            details(solution),
            md("## Check"),
            code(check),
        ],
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    lesson(
        "01_setup.ipynb",
        """# 01 — Setup

This course trains a **System One** model: a finished piece of text goes in, and probabilities come out in one pass. Nothing is generated token by token.

Jev ([announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)) is a hosted model of that kind. You are going to build a tiny one in PyTorch and see every tensor.

The GTX 1080 trains this in **float32**. It has no tensor cores, so float16 is not a speedup.

**You implement:** `cuda_report`, `course_dtype`.""",
        '''from __future__ import annotations

def cuda_report() -> dict:
    """Keys: cuda (bool), name (str | None), vram_gb (float)."""
    raise NotImplementedError("cuda_report")


def course_dtype() -> str:
    """The dtype this course trains in."""
    raise NotImplementedError("course_dtype")
''',
        '''import torch

def cuda_report() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False, "name": None, "vram_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
    }

def course_dtype() -> str:
    return "float32"
''',
        '''report = cuda_report()
assert set(report) >= {"cuda", "name", "vram_gb"}
assert course_dtype() == "float32"
print(report, course_dtype())
''',
    )
    lesson(
        "02_decisions.ipynb",
        """# 02 — Three answers

| Primitive | Question | You return |
|---|---|---|
| **Choice** | which label? | a distribution over names |
| **Score** | where on an ordered scale? | a distribution over levels, and their mean |
| **Noul** | is this true? | P(yes) |

**Confidence** (choice and score only) is how peaked that distribution is: `1 - entropy / log(k)`. A one-hot distribution scores 1. A uniform distribution scores 0. Noul does not get a second number; the probability is the answer.

**You implement:** `choice_probs`, `noul_prob`, `confidence`.""",
        '''from __future__ import annotations

import torch
from torch import Tensor


def choice_probs(logits: Tensor) -> Tensor:
    raise NotImplementedError("choice_probs")


def noul_prob(logit: Tensor) -> Tensor:
    raise NotImplementedError("noul_prob")


def confidence(probs: Tensor) -> Tensor:
    """probs sums to 1 on the last axis. Return a tensor in [0, 1]."""
    raise NotImplementedError("confidence")
''',
        '''import math
import torch
from torch import Tensor

def choice_probs(logits: Tensor) -> Tensor:
    return torch.softmax(logits.float(), dim=-1)

def noul_prob(logit: Tensor) -> Tensor:
    return torch.sigmoid(logit.float())

def confidence(probs: Tensor) -> Tensor:
    probs = probs.float()
    k = probs.size(-1)
    if k < 2:
        return torch.ones(probs.shape[:-1], device=probs.device, dtype=probs.dtype)
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return (1.0 - entropy / math.log(k)).clamp(0.0, 1.0)
''',
        '''logits = torch.tensor([0.0, 0.0, 4.0])
probs = choice_probs(logits)
assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
assert int(probs.argmax()) == 2
assert abs(float(noul_prob(torch.tensor(0.0))) - 0.5) < 1e-5
assert float(confidence(torch.tensor([0.0, 1.0]))) > 0.99
assert float(confidence(torch.tensor([0.5, 0.5]))) < 0.01
print("ok", probs)
''',
    )
    lesson(
        "03_bytes.ipynb",
        """# 03 — The state is just bytes

There is no sentencepiece download. Each UTF-8 byte is a token id in `0..255`. Id `256` is padding and is not a byte, so it never collides with text.

A personal chat log is a few million of these bytes. That is the whole corpus size this model is built for. A string longer than `max_len` keeps its **last** bytes, so a packed thread still ends on the newest line.

**You implement:** `clean_state`, `encode_state`.""",
        '''from __future__ import annotations


def clean_state(text: str) -> str | None:
    """Trim, collapse whitespace. None if empty or a bare URL."""
    raise NotImplementedError("clean_state")


def encode_state(text: str, max_len: int, pad_id: int = 256) -> tuple[list[int], list[float]]:
    """Return (ids length max_len, mask of 1.0 on real bytes)."""
    raise NotImplementedError("encode_state")
''',
        '''def clean_state(text: str) -> str | None:
    text = " ".join(text.split())
    if not text:
        return None
    lower = text.lower()
    if (lower.startswith("http://") or lower.startswith("https://")) and " " not in text:
        return None
    return text

def encode_state(text: str, max_len: int, pad_id: int = 256) -> tuple[list[int], list[float]]:
    ids = list(text.encode("utf-8"))
    if len(ids) > max_len:
        ids = ids[-max_len:]
    if not ids:
        ids = [ord(" ")]
    width = len(ids)
    pad = max_len - width
    return ids + [pad_id] * pad, [1.0] * width + [0.0] * pad
''',
        '''assert clean_state("  hey   there ") == "hey there"
assert clean_state("https://example.com/a") is None
ids, mask = encode_state("hi", max_len=4)
assert ids == [ord("h"), ord("i"), 256, 256]
assert mask == [1.0, 1.0, 0.0, 0.0]
tail, _ = encode_state("abcdef", max_len=3)
assert tail == [ord("d"), ord("e"), ord("f")]
print(ids, mask)
''',
    )
    lesson(
        "04_attention.ipynb",
        """# 04 — Attention with the mask off

In [buka](https://github.com/harrisonstark/buka) a causal mask stops position 0 from reading the future, because the future is the answer being predicted.

Here the message is already written. Position 0 is allowed to look at the end of the line. That single change is why one forward pass can answer a question about the whole state.

`key_mask` is 1 on real tokens and 0 on pad. Pad keys get a large negative score so softmax ignores them.

**You implement:** `bidirectional_attention`. Tensors are `(batch, heads, time, dim)`.""",
        '''from __future__ import annotations

import torch
from torch import Tensor


def bidirectional_attention(q: Tensor, k: Tensor, v: Tensor, key_mask: Tensor | None = None) -> Tensor:
    raise NotImplementedError("bidirectional_attention")
''',
        '''import torch
from torch import Tensor

def bidirectional_attention(q: Tensor, k: Tensor, v: Tensor, key_mask: Tensor | None = None) -> Tensor:
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if key_mask is not None:
        keep = key_mask.to(dtype=scores.dtype)
        scores = scores + (1.0 - keep)[:, None, None, :] * -1.0e9
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
''',
        '''q = torch.zeros(1, 1, 3, 2)
k = torch.zeros(1, 1, 3, 2)
v = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).view(1, 1, 3, 2)
out = bidirectional_attention(q, k, v)
assert abs(float(out[0, 0, 0, 0]) - 1.0) < 1e-4, float(out[0, 0, 0, 0])
masked = bidirectional_attention(q, k, v, key_mask=torch.tensor([[1.0, 1.0, 0.0]]))
assert abs(float(masked[0, 0, 0, 0]) - 0.5) < 1e-4
print("position 0 saw the future:", float(out[0, 0, 0, 0]))
''',
    )
    lesson(
        "05_block.ipynb",
        """# 05 — RMSNorm and SwiGLU

Same two pieces as a modern decoder block. They sit in an **encoder** block here: pre-norm, residual, no causal mask (you already wrote the attention).

RMSNorm rescales a vector so its root-mean-square is 1, then multiplies by a learned weight. SwiGLU is `silu(gate) * up`. The down-projection is a separate linear in the full block (`buka_rs/block.py`).

**You implement:** `rms_norm`, `swiglu`.""",
        '''from __future__ import annotations

import torch
from torch import Tensor


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    raise NotImplementedError("rms_norm")


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    raise NotImplementedError("swiglu")
''',
        '''import torch
import torch.nn.functional as F
from torch import Tensor

def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(mean_square + eps)
    return (x.float() * inv * weight.float()).to(dtype=x.dtype)

def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    return F.silu(gate) * up
''',
        '''weight = torch.ones(2)
y = rms_norm(torch.tensor([[3.0, 4.0]]), weight)
inv = 1.0 / (12.5 ** 0.5)
assert abs(float(y[0, 0]) - 3 * inv) < 1e-4
hidden = swiglu(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0]))
assert abs(float(hidden[0])) < 1e-5
assert float(hidden[1]) > 2.0
assert y.dtype == torch.tensor([[3.0, 4.0]]).dtype
print(float(y[0, 0]), float(hidden[1]))
''',
    )
    lesson(
        "06_pool.ipynb",
        """# 06 — One vector for the whole state

The heads do not run at every byte. They run once, on a single summary of the message. Mean-pool the token states and ignore padding, or the pad positions would drag the average toward zero.

**You implement:** `mean_pool`. `x` is `(batch, time, dim)`, `mask` is `(batch, time)`.""",
        '''from __future__ import annotations

import torch
from torch import Tensor


def mean_pool(x: Tensor, mask: Tensor) -> Tensor:
    raise NotImplementedError("mean_pool")
''',
        '''import torch
from torch import Tensor

def mean_pool(x: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom
''',
        '''x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]])
mask = torch.tensor([[1.0, 1.0, 0.0]])
pooled = mean_pool(x, mask)
assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))
print(pooled)
''',
    )
    lesson(
        "07_heads.ipynb",
        """# 07 — Parallel heads

One pooled vector, three linears, three answers. They do not wait on each other. That is the "parallel sampler" idea at a size you can train: every question is a head, and they all read the same state.

- **Choice** — softmax, argmax name, confidence
- **Score** — softmax over ordered levels, score = sum of `index * probability`
- **Noul** — sigmoid

Temperature divides the logits before softmax. `1` leaves them alone.

**You implement:** `choice_answer`, `score_answer`, `noul_answer`.""",
        '''from __future__ import annotations

import torch
from torch import Tensor


def choice_answer(logits: Tensor, names: list[str], temperature: float = 1.0) -> dict:
    raise NotImplementedError("choice_answer")


def score_answer(logits: Tensor, legend: list[str], temperature: float = 1.0) -> dict:
    raise NotImplementedError("score_answer")


def noul_answer(logit: Tensor) -> dict:
    raise NotImplementedError("noul_answer")
''',
        '''import math
import torch
from torch import Tensor

def _confidence(probs: Tensor) -> float:
    k = probs.numel()
    if k < 2:
        return 1.0
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum()
    return float((1.0 - entropy / math.log(k)).clamp(0.0, 1.0))

def choice_answer(logits: Tensor, names: list[str], temperature: float = 1.0) -> dict:
    probs = torch.softmax(logits.float() / max(temperature, 1e-3), dim=-1)
    index = int(torch.argmax(probs))
    return {
        "choice": names[index],
        "probabilities": {name: float(probs[i]) for i, name in enumerate(names)},
        "confidence": _confidence(probs),
    }

def score_answer(logits: Tensor, legend: list[str], temperature: float = 1.0) -> dict:
    probs = torch.softmax(logits.float() / max(temperature, 1e-3), dim=-1)
    levels = torch.arange(probs.numel(), device=probs.device, dtype=probs.dtype)
    return {
        "score": float((probs * levels).sum()),
        "probabilities": {str(i): float(probs[i]) for i in range(probs.numel())},
        "legend": list(legend),
        "confidence": _confidence(probs),
    }

def noul_answer(logit: Tensor) -> dict:
    return {"noul": float(torch.sigmoid(logit.float()))}
''',
        '''choice = choice_answer(torch.tensor([0.0, 5.0, 0.0]), ["chat", "ops", "learning"])
assert choice["choice"] == "ops"
assert abs(sum(choice["probabilities"].values()) - 1) < 1e-4
score = score_answer(torch.tensor([0.0, 0.0, 8.0]), ["calm", "soon", "now"])
assert score["score"] > 1.9
reply = noul_answer(torch.tensor(0.0))
assert abs(reply["noul"] - 0.5) < 1e-5
print(choice["choice"], round(score["score"], 3), reply["noul"])
''',
    )
    lesson(
        "08_data.ipynb",
        """# 08 — Labeled states, not a monologue

[buka](https://github.com/harrisonstark/buka) trains on text and then talks. [buka-evo](https://github.com/harrisonstark/buka-evo) adapts a 3B chat model. This file is a third shape: each line is a **state** plus the decision you want.

```json
{"state": "server is down, can you look tonight?", "topic": "ops", "urgency": 2, "needs_reply": true}
```

`topic` is the choice (`chat`, `ops`, `learning`). `urgency` is the score level (`0` calm, `1` soon, `2` now). `needs_reply` is the noul.

The committed sample is synthetic. A Discord export can use the same schema after you label it, and it stays out of git. See `docs/bring_your_own_data.md`.

**You implement:** `parse_example`.""",
        '''from __future__ import annotations


def parse_example(raw: dict) -> dict | None:
    """Return state/topic/urgency/needs_reply, or None if the state is empty.

    urgency must be an int. needs_reply must be a bool.
    """
    raise NotImplementedError("parse_example")
''',
        '''def parse_example(raw: dict) -> dict | None:
    if raw.get("state") is None:
        return None
    state = " ".join(str(raw.get("state", "")).split())
    if not state:
        return None
    lower = state.lower()
    if (lower.startswith("http://") or lower.startswith("https://")) and " " not in state:
        return None
    flag = raw["needs_reply"]
    if isinstance(flag, str):
        lowered = flag.strip().lower()
        if lowered in {"true", "1", "yes"}:
            flag = True
        elif lowered in {"false", "0", "no"}:
            flag = False
        else:
            raise ValueError(f"needs_reply must be a bool, got {flag!r}")
    elif not isinstance(flag, bool):
        raise ValueError(f"needs_reply must be a bool, got {flag!r}")
    urgency = raw["urgency"]
    if isinstance(urgency, bool) or isinstance(urgency, float):
        raise ValueError(f"urgency must be an int, got {urgency!r}")
    if not isinstance(urgency, int):
        raise ValueError(f"urgency must be an int, got {urgency!r}")
    return {
        "state": state,
        "topic": str(raw["topic"]),
        "urgency": urgency,
        "needs_reply": flag,
    }
''',
        '''row = parse_example({
    "state": "  server is down  ",
    "topic": "ops",
    "urgency": 2,
    "needs_reply": True,
})
assert row["topic"] == "ops" and row["urgency"] == 2 and row["needs_reply"] is True
assert parse_example({"state": "  ", "topic": "chat", "urgency": 0, "needs_reply": False}) is None
no = parse_example({"state": "hey", "topic": "chat", "urgency": 0, "needs_reply": "false"})
assert no["needs_reply"] is False
assert parse_example({"state": None, "topic": "chat", "urgency": 0, "needs_reply": False}) is None
try:
    parse_example({"state": "hey", "topic": "chat", "urgency": 1.9, "needs_reply": False})
    raise SystemExit("1.9 should fail")
except ValueError:
    pass
print(row)
''',
    )
    lesson(
        "09_train.ipynb",
        """# 09 — One training step

Three losses, added:

- cross-entropy on topic
- cross-entropy on urgency level
- binary cross-entropy on the reply logit

There is no next-token shift. The label is a decision about the whole window.

**You implement:** `train_step(model, batch, optimizer) -> float`.

`batch` has `input_ids`, `mask`, `topic` (long), `urgency` (long), `reply` (float 0/1). The model returns an object with `.topic`, `.urgency`, and `.reply` logits.""",
        '''from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def train_step(model, batch: dict[str, Tensor], optimizer: torch.optim.Optimizer) -> float:
    raise NotImplementedError("train_step")
''',
        '''import torch
import torch.nn.functional as F
from torch import Tensor

def train_step(model, batch: dict[str, Tensor], optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(batch["input_ids"], batch["mask"])
    loss = (
        F.cross_entropy(out.topic, batch["topic"])
        + F.cross_entropy(out.urgency, batch["urgency"])
        + F.binary_cross_entropy_with_logits(out.reply, batch["reply"])
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())
''',
        '''import torch
from buka_rs.config import OneConfig
from buka_rs.data import Example, collate
from buka_rs.model import OneModel

cfg = OneConfig(name="micro", d_model=32, n_layers=1, n_heads=4, d_ff=64, max_seq_len=32)
model = OneModel(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
rows = [Example("server is down tonight?", "ops", 2, True)] * 4
batch = collate(rows, cfg.topics, 24, cfg.pad_id, torch.device("cpu"))
before = [p.detach().clone() for p in model.parameters()]
loss = train_step(model, batch, opt)
assert loss > 0
assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
print("step loss", round(loss, 4))
''',
    )
    lesson(
        "10_calibrate.ipynb",
        """# 10 — Make the probabilities less theatrical

A head can be almost always right and still shout `0.99` when it should say `0.7`. Temperature scaling divides the logits by `T` before the softmax. You pick `T` by trying a grid and keeping the value with the lowest cross-entropy on rows the optimizer did not train on.

Jev's training method is called RLCD (reinforcement learning for calibrated decisions). The announcement describes the goal, not the algorithm, so this lesson is the calibration step you can actually code and check.

**You implement:** `fit_temperature(logits, labels, grid) -> float`.

`logits` is `(n, classes)`, `labels` is `(n,)` long.""",
        '''from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def fit_temperature(logits: Tensor, labels: Tensor, grid: list[float] | None = None) -> float:
    raise NotImplementedError("fit_temperature")
''',
        '''import torch
import torch.nn.functional as F
from torch import Tensor

def fit_temperature(logits: Tensor, labels: Tensor, grid: list[float] | None = None) -> float:
    if logits.numel() == 0:
        return 1.0
    candidates = grid or [0.5, 0.7, 1.0, 1.3, 1.7, 2.2, 3.0]
    best_t, best_nll = 1.0, float("inf")
    for temperature in candidates:
        nll = float(F.cross_entropy(logits.float() / temperature, labels).item())
        if nll < best_nll:
            best_t, best_nll = float(temperature), nll
    return best_t
''',
        '''logits = torch.tensor([[8.0, 0.0], [0.3, 0.0]])
labels = torch.tensor([1, 0])
chosen = fit_temperature(logits, labels, grid=[0.5, 1.0, 2.0])
assert chosen == 2.0
assert fit_temperature(torch.empty(0, 2), torch.empty(0, dtype=torch.long)) == 1.0
print("T", chosen)
''',
    )
    lesson(
        "11_serve.ipynb",
        """# 11 — Capstone

Wire the three answers into the payload the page expects, then train and open it.

The sample file is synthetic on purpose. After this works, point `--data` at a **local** JSONL of the same schema (a labeled slice of your chats). Do not commit that file.

```powershell
python -m scripts.train --config configs/tiny.yaml --calibrate
python -m scripts.serve --checkpoint checkpoints/buka_latest.pt
```

Open http://127.0.0.1:7860 and paste a message. You get a topic, an urgency score, and P(needs a reply). The page does not ask the model to write a reply. That job is buka / buka-evo.

**You implement:** `response_payload`.""",
        '''from __future__ import annotations


def response_payload(topic: dict, urgency: dict, reply: dict) -> dict:
    raise NotImplementedError("response_payload")
''',
        '''def response_payload(topic: dict, urgency: dict, reply: dict) -> dict:
    return {"answers": {"topic": topic, "urgency": urgency, "needs_reply": reply}}
''',
        '''body = response_payload({"choice": "ops"}, {"score": 1.5}, {"noul": 0.8})
assert set(body["answers"]) == {"topic", "urgency", "needs_reply"}
print(body)
''',
    )


if __name__ == "__main__":
    main()
