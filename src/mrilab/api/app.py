"""FastAPI app: JSON API under /api/v1 + the server-rendered UI at the root
(same process, same face builders — no second source of truth). CORS stays for
external API consumers; the UI itself is same-origin."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mrilab.api.routers import arena, browse, ci, jobs, meta, reports
from mrilab.settings import get_settings
from mrilab.ui import routes as ui_routes


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
    app.mount("/ui/static",
              StaticFiles(directory=Path(ui_routes.__file__).parent / "static"),
              name="ui-static")
    app.include_router(ui_routes.router)
    return app


app = create_app()
