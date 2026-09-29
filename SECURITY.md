# Security & privacy

This repository is a public course. Never commit:

- Discord GDPR exports (`package/`, `packageL/`, or similar)
- Raw or labeled chat logs of real people
- Checkpoints (`.pt`) trained on private conversations
- API keys or `.env` files

`.gitignore` covers the usual paths. You still choose what you `git add`.

Discord exports include other people's messages. Do not publish those exports or models trained on them without a real legal review and consent.

The default train path is `data/samples/demo_decisions.jsonl`. `scripts/serve` binds to `127.0.0.1`.
