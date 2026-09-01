"""FastAPI app (prefix /api/v1). CORS configured now — the frontend dev server
runs on Windows against this WSL-bound service (bind 0.0.0.0; WSL2 localhost
forwarding is flaky after sleep/VPN, documented in ops notes)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mrilab.api.routers import arena, browse, ci, jobs, meta, reports
from mrilab.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="mri-lab", version="0.1.0",
                  description="The Model MRI backend: frozen instruments, "
                              "reference surfaces, evidence-disciplined reports.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in (meta.router, jobs.router, reports.router, browse.router,
              ci.router, arena.router):
        app.include_router(r, prefix="/api/v1")
    return app


app = create_app()
