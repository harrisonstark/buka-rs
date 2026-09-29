"""Local page: paste a state, read Choice / Score / Noul."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel


class DecideRequest(BaseModel):
    state: str


def create_app(checkpoint: str, device: str = "cpu") -> Any:
    from fastapi import FastAPI
    from fastapi.responses import FileResponse

    from buka_rs.infer import decide
    from buka_rs.train import load_checkpoint

    app = FastAPI(title="buka-rs", version="0.1.0")
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model, temperatures, seq_len = load_checkpoint(checkpoint, dev)
    ui = Path(__file__).resolve().parent.parent / "ui" / "index.html"

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(ui)

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "params": model.num_parameters(),
            "params_m": round(model.num_parameters() / 1e6, 2),
            "device": str(dev),
            "dtype": "float32",
            "preset": model.config.name,
        }

    @app.post("/decide")
    def decide_route(body: DecideRequest) -> dict:
        from fastapi import HTTPException

        try:
            return decide(model, body.state, temperatures, seq_len=seq_len)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the decision model")
    parser.add_argument("--checkpoint", default="checkpoints/buka_latest.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    import uvicorn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    uvicorn.run(create_app(args.checkpoint, device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
