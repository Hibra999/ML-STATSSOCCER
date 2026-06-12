from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.web import mundial_services as services
from src.web.config import ALLOWED_HOSTS, LOCAL_HOST
from src.web.jobs import jobs
from src.web.static_assets import PublicStorageAssets


MUNDIAL_PORT = 5052
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = PROJECT_ROOT / "src" / "web" / "static"
STORAGE_ROOT = PROJECT_ROOT / "storage"


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_host = request.client.host if request.client else ""
        if client_host != "testclient" and client_host not in {"127.0.0.1", "::1"}:
            return JSONResponse(status_code=403, content={"ok": False, "data": {}, "error": "Acceso permitido solo desde localhost."})

        host = _host_without_port(request.headers.get("host", ""))
        if client_host != "testclient" and host and host not in ALLOWED_HOSTS:
            return JSONResponse(status_code=403, content={"ok": False, "data": {}, "error": "Host local obligatorio."})
        return await call_next(request)


def create_mundial_app() -> FastAPI:
    app = FastAPI(title="Mundial 2026 ML-STATSSOCCER", docs_url="/api/docs", redoc_url=None)
    app.add_middleware(LocalOnlyMiddleware)

    @app.middleware("http")
    async def no_cache_static_assets(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/mundial"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")
    app.mount("/assets", PublicStorageAssets(directory=str(STORAGE_ROOT)), name="assets")

    @app.get("/")
    def index():
        return FileResponse(STATIC_ROOT / "mundial.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/api/health")
    def health():
        return _ok({"status": "ok", "host": LOCAL_HOST, "port": int(os.environ.get("MLSTATSSOCCER_MUNDIAL_PORT", MUNDIAL_PORT))})

    @app.get("/api/mundial/overview")
    def overview(refresh: bool = False):
        return _wrap(services.overview, refresh)

    @app.get("/api/mundial/groups")
    def groups(refresh: bool = False):
        return _wrap(services.groups, refresh)

    @app.get("/api/mundial/fixtures")
    def fixtures(refresh: bool = False):
        return _wrap(services.fixtures, refresh)

    @app.get("/api/mundial/teams")
    def teams(refresh: bool = False):
        return _wrap(services.teams, refresh)

    @app.get("/api/mundial/players")
    def players(refresh: bool = False):
        return _wrap(services.players, refresh)

    @app.post("/api/mundial/training/download-kaggle")
    def training_download(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.training_download, payload)

    @app.post("/api/mundial/training/prepare-etl")
    def training_prepare(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.training_prepare, payload)

    @app.post("/api/mundial/training/player-snapshots")
    def training_player_snapshots(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.refresh_player_snapshots, payload)

    @app.get("/api/mundial/training/dataset")
    def training_dataset():
        return _wrap(services.training_dataset)

    @app.post("/api/mundial/training/train")
    def training_train(payload: Dict[str, Any] = Body(default={})):
        return _submit("Entrenando modelo Mundial", services.training_train, payload, with_progress=True, lock_key="mundial-training")

    @app.get("/api/mundial/training/status")
    def training_status():
        return _wrap(services.training_status)

    @app.get("/api/mundial/training/options")
    def training_options():
        return _wrap(services.training_options)

    @app.get("/api/mundial/models")
    def models_catalog():
        return _wrap(services.models_catalog)

    @app.post("/api/mundial/models/train")
    def models_train(payload: Dict[str, Any] = Body(default={})):
        return _submit("Entrenando modelo Mundial", services.training_train, payload, with_progress=True, lock_key="mundial-training")

    @app.post("/api/mundial/models/select")
    def models_select(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.select_model, payload)

    @app.delete("/api/mundial/models/{model_id}")
    def models_delete(model_id: str):
        return _wrap(services.delete_model, model_id)

    @app.post("/api/mundial/maintenance/clear")
    def maintenance_clear(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.maintenance_clear, payload)

    @app.post("/api/mundial/predict-match")
    def predict_match(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.predict_match, payload)

    @app.post("/api/mundial/predict-upcoming")
    def predict_upcoming(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.predict_upcoming, payload)

    @app.post("/api/mundial/simulate")
    def simulate(payload: Dict[str, Any] = Body(default={})):
        return _submit("Simulando Monte Carlo Mundial", services.simulate, payload, with_progress=True)

    @app.get("/api/mundial/procedure")
    def procedure():
        return _wrap(services.procedure)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"ok": False, "data": {}, "error": "Proceso no encontrado."})
        return _ok(job)

    return app


def _submit(message: str, fn, *args, with_progress: bool = False, **kwargs):
    try:
        return _ok(jobs.submit(message, fn, *args, with_progress=with_progress, **kwargs))
    except Exception as exc:
        return _error(exc)


def _wrap(fn, *args, **kwargs):
    try:
        return _ok(fn(*args, **kwargs))
    except Exception as exc:
        return _error(exc)


def _ok(data: Any):
    return {"ok": True, "data": services.jsonable(data), "error": ""}


def _error(exc: Exception):
    status_code = 400 if isinstance(exc, (ValueError, RuntimeError)) else 500
    return JSONResponse(status_code=status_code, content={"ok": False, "data": {}, "error": f"{exc.__class__.__name__}: {exc}"})


def _host_without_port(host_header: str) -> str:
    if host_header.startswith("[::1]"):
        return "::1"
    if ":" in host_header:
        return host_header.rsplit(":", maxsplit=1)[0]
    return host_header


app = create_mundial_app()
