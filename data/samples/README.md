# Sample decisions

`demo_decisions.jsonl` is synthetic. Each line is a state plus three labels. The same lines repeat so a tiny from-scratch model can memorize them during the capstone.

```powershell
python -m scripts.train --config configs/tiny.yaml
```

Your own labeled file: [docs/bring_your_own_data.md](../../docs/bring_your_own_data.md). Do not commit real chats.
