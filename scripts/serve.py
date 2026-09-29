"""Local page: paste a state, read Choice / Score / Noul."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field


class DecideRequest(BaseModel):
    state: str = Field(max_length=8_000)


_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def ensure_bind(host: str, allow_network: bool) -> None:
    if host in _LOOPBACK or allow_network:
        return
    raise SystemExit(
        f"Refusing to bind {host}. Pass --allow-network if you really want a non-loopback host."
    )


def create_app(checkpoint: str, device: str = "cpu", api_key: str | None = None) -> Any:
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse

    from buka_rs.infer import decide
    from buka_rs.train import load_checkpoint

    app = FastAPI(title="buka-rs", version="0.1.0")
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model, temperatures, seq_len = load_checkpoint(checkpoint, dev)
    ui = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    lock = threading.Lock()

    def _check_key(x_api_key: str | None) -> None:
        if api_key is None:
            return
        if not x_api_key or x_api_key != api_key:
            raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")

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
    def decide_route(
        body: DecideRequest,
        x_api_key: str | None = Header(default=None),
    ) -> dict:
        _check_key(x_api_key)
        try:
            with lock:
                return decide(model, body.state, temperatures, seq_len=seq_len)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the decision model")
    parser.add_argument("--checkpoint", default="checkpoints/buka_latest.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow binding beyond loopback. Localhost only by default.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="If set, require header X-API-Key on /decide.",
    )
    args = parser.parse_args()
    ensure_bind(args.host, args.allow_network)
    if args.allow_network and not args.api_key:
        print("[serve] WARNING: --allow-network without --api-key")
    import uvicorn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    uvicorn.run(
        create_app(args.checkpoint, device, api_key=args.api_key),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
