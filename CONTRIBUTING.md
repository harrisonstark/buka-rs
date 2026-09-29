# Contributing

## Dev setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[dev]"
pytest
```

## Notebooks

- Lessons live in `notebooks/*.ipynb` (stub → collapsed solution → check).
- Rebuild with `python scripts/build_course_notebooks.py`.
- Keep stub cells raising `NotImplementedError`.
- Solutions belong in the `<details>` block and in `buka_rs/`.

## Privacy

Do not commit Discord exports, `data/processed/`, or checkpoints. See [SECURITY.md](SECURITY.md).
