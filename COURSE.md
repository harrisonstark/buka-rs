# Course guide

## Pacing

| Plan | Schedule |
|------|----------|
| **Weekend crash** | Sat: notebooks 01–06. Sun: 07–10, then 11 (train + page). |
| **Two weeks** | Days 1–3: what a decision is (01–03). 4–7: encoder (04–07). 8–10: data, step, temperature. 11–12: train longer on your own labels. |
| **Notebook only** | Each `.ipynb` is the whole lesson. |

## Outcomes

By the end you can:

1. Say what Choice, Score, and Noul return, and what confidence is measuring
2. Implement bidirectional attention, RMSNorm, SwiGLU, and a pad-aware pool
3. Train three heads from scratch on labeled states
4. Run the local decision page

## Rules of the road

- Struggle before expanding the **Solution** `<details>` block
- `buka_rs/` is the library answer key, not the first stop
- Capstone uses **sample data**, not a personal Discord dump in git
- Train in **float32**

## Where the other repos fit

| Repo | You get |
|------|---------|
| [buka](https://github.com/harrisonstark/buka) | A model that continues text |
| [buka-evo](https://github.com/harrisonstark/buka-evo) | A 3B chat model nudged with LoRA |
| **buka-rs** | A model that scores a message and stops |

## Capstone

```powershell
python -m scripts.train --config configs/tiny.yaml --calibrate
python -m scripts.serve --checkpoint checkpoints/buka_latest.pt
```

Open http://127.0.0.1:7860. Paste a message. Read the three answers.
