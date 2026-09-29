# Bring your own data

The notebooks and the default train command use `data/samples/demo_decisions.jsonl`. You do not need a Discord export to finish the course.

## Schema

One JSON object per line:

```json
{"state": "pager is open, confirm you are taking it", "topic": "ops", "urgency": 2, "needs_reply": true}
```

| Field | Values |
|-------|--------|
| `state` | the message, or a short thread you flattened yourself |
| `topic` | `chat`, `ops`, or `learning` |
| `urgency` | `0` calm, `1` soon, `2` now |
| `needs_reply` | `true` or `false` |

Put the file under `data/processed/` (gitignored) and train:

```powershell
python -m scripts.train --data data\processed\my_decisions.jsonl --preset tiny --steps 800 --calibrate
```

## From a Discord export

buka already parses a GDPR Messages tree into monologue text. Keep that export outside git. Then label states yourself. A heuristic (question mark → `needs_reply`) is a start; the model will learn the heuristic, including its mistakes. Hand labels on a few hundred lines teach you more about the heads than a giant unlabeled dump.

Do not commit `package/`, `packageL`, raw exports, or `data/processed/`. See [SECURITY.md](../SECURITY.md).

## Changing the questions

The three heads are the course. To ask something else (for example "is this about a person" or a different topic list), change `TOPICS` / `URGENCY` in `buka_rs/config.py` and the labels together, then retrain. The page will show whatever names the checkpoint was built with.
