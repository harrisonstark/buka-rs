"""Train the decision model and write checkpoints/buka_latest.pt."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import yaml

from buka_rs.config import config_from_preset
from buka_rs.data import Example, load_examples
from buka_rs.train import calibrate, evaluate, run, save_checkpoint


def split_by_state(examples: list[Example], seed: int) -> tuple[list[Example], list[Example]]:
    """Hold out whole states so calibration never sees a line the optimizer trained on."""
    grouped: dict[str, list[Example]] = {}
    for row in examples:
        grouped.setdefault(row.state, []).append(row)
    states = list(grouped)
    if len(states) < 5:
        return examples, []
    random.Random(seed).shuffle(states)
    n_hold = max(1, len(states) // 5)
    held = set(states[:n_hold])
    train_rows: list[Example] = []
    hold_rows: list[Example] = []
    for state, rows in grouped.items():
        (hold_rows if state in held else train_rows).extend(rows)
    return train_rows, hold_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small System One model")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="checkpoints/buka_latest.pt")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    file_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    preset = args.preset or file_cfg.get("preset", "tiny")
    data = args.data or file_cfg.get("data", "data/samples/demo_decisions.jsonl")
    steps = args.steps if args.steps is not None else int(file_cfg.get("steps", 400))
    batch = args.batch if args.batch is not None else int(file_cfg.get("batch", 16))
    seq = args.seq if args.seq is not None else int(file_cfg.get("seq", 96))
    lr = args.lr if args.lr is not None else float(file_cfg.get("lr", 1e-3))

    config = config_from_preset(preset)
    if seq > config.max_seq_len:
        print(f"seq {seq} is above max_seq_len {config.max_seq_len}; using {config.max_seq_len}")
        seq = config.max_seq_len
    loaded = load_examples(data, config.topics, len(config.urgency))
    examples, holdout = split_by_state(loaded, seed=args.seed)
    if not holdout:
        print("not enough distinct states to hold any out; metrics are train-set only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"preset {config.name}  params-to-build  train {len(examples)}  "
        f"holdout {len(holdout)}  seq {seq}  device {device}  dtype float32"
    )
    model, metrics = run(
        config,
        examples,
        steps=steps,
        batch_size=batch,
        seq_len=seq,
        lr=lr,
        seed=args.seed,
        device=device,
    )
    temps = {"topic": 1.0, "urgency": 1.0}
    if args.calibrate and holdout:
        temps = calibrate(model, holdout, seq, device)
        print(f"temperatures fit on held-out states {temps}")
    elif args.calibrate:
        print("temperatures stay 1.0")
    save_checkpoint(args.out, model, temps, seq_len=seq)
    print(
        f"loss {metrics['first_loss']:.4f} -> {metrics['last_loss']:.4f}  "
        f"train-set topic {metrics['topic_acc']:.2f}  reply {metrics['reply_acc']:.2f}  "
        f"urgency_mae {metrics['urgency_mae']:.2f}  params {int(metrics['params'])}"
    )
    if holdout:
        held = evaluate(model, holdout, seq, device, temperatures=temps)
        print(
            f"holdout topic {held['topic_acc']:.2f}  reply {held['reply_acc']:.2f}  "
            f"urgency_mae {held['urgency_mae']:.2f}"
        )
    print(f"wrote {args.out}")
    print(f"serve: python -m scripts.serve --checkpoint {args.out}")


if __name__ == "__main__":
    main()
