from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.cli.common import CLIError
from src.web import services
from src.web.config import ALLOWED_HOSTS, LOCAL_HOST, LOCAL_PORT
from src.web.jobs import jobs


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_host = request.client.host if request.client else ""
        if client_host != "testclient" and client_host not in {"127.0.0.1", "::1"}:
            return JSONResponse(status_code=403, content={"ok": False, "data": {}, "error": "Acceso permitido solo desde localhost."})

        host = _host_without_port(request.headers.get("host", ""))
        if client_host != "testclient" and host and host not in ALLOWED_HOSTS:
            return JSONResponse(status_code=403, content={"ok": False, "data": {}, "error": "Host local obligatorio."})
        return await call_next(request)


def create_app() -> FastAPI:
    app = FastAPI(title="ML-STATSSOCCER Web Local", docs_url="/api/docs", redoc_url=None)
    app.add_middleware(LocalOnlyMiddleware)

    services.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
    app.mount("/assets", StaticFiles(directory="storage"), name="assets")
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

    @app.get("/")
    def index():
        return FileResponse("src/web/static/index.html")

    @app.get("/api/health")
    def health():
        return _ok({"status": "ok", "host": LOCAL_HOST, "port": int(os.environ.get("MLSTATSSOCCER_PORT", LOCAL_PORT))})

    @app.get("/api/dashboard")
    def dashboard():
        return _wrap(services.dashboard)

    @app.get("/api/dashboard/fixtures")
    def dashboard_fixtures(limit: int = 5, days: int = 7):
        return _wrap(services.dashboard_fixtures, limit, days)

    @app.get("/api/model-specs")
    def model_specs():
        return _wrap(services.model_specs)

    @app.get("/api/leagues/catalog")
    def catalog_leagues():
        return _wrap(services.catalog_leagues)

    @app.get("/api/leagues")
    def saved_leagues():
        return _wrap(services.saved_leagues)

    @app.post("/api/leagues")
    def create_league(payload: Dict[str, Any] = Body(...)):
        return _submit("Creando liga", services.create_league, payload)

    @app.get("/api/leagues/{league_id}")
    def league_detail(league_id: str, rows: int = 25):
        return _wrap(services.league_detail, league_id, rows)

    @app.post("/api/leagues/{league_id}/update")
    def update_league(league_id: str):
        return _submit(f"Actualizando {league_id}", services.update_league, league_id)

    @app.delete("/api/leagues/{league_id}")
    def delete_league(league_id: str):
        return _wrap(services.delete_league, league_id)

    @app.get("/api/leagues/{league_id}/data")
    def league_data(
            league_id: str,
            page: int = 1,
            page_size: int = 50,
            query: Optional[str] = None,
            column: Optional[str] = None,
            exact: bool = False,
            hide_missing: bool = False,
            columns: Optional[str] = None,
    ):
        return _wrap(services.league_data, league_id, page, page_size, query, column, exact, hide_missing, columns)

    @app.get("/api/leagues/{league_id}/data/export")
    def export_league_data(
            league_id: str,
            fmt: str = "csv",
            query: Optional[str] = None,
            column: Optional[str] = None,
            exact: bool = False,
            hide_missing: bool = False,
            columns: Optional[str] = None,
    ):
        return _wrap(services.export_league_data, league_id, fmt, query, column, exact, hide_missing, columns)

    @app.get("/api/leagues/{league_id}/models")
    def list_models(league_id: str):
        return _wrap(services.list_models, league_id)

    @app.post("/api/leagues/{league_id}/models/train")
    def train_model(league_id: str, payload: Dict[str, Any] = Body(...)):
        return _submit(f"Entrenando modelo para {league_id}", services.train_model, league_id, payload, with_progress=True)

    @app.post("/api/leagues/{league_id}/models/{model_id}/evaluate")
    def evaluate_model(league_id: str, model_id: str, payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.evaluate_model, league_id, model_id, payload)

    @app.delete("/api/leagues/{league_id}/models/{model_id}")
    def delete_model(league_id: str, model_id: str):
        return _wrap(services.delete_model, league_id, model_id)

    @app.post("/api/leagues/{league_id}/fixtures/upcoming")
    def upcoming_fixtures(league_id: str, payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.upcoming_fixtures, league_id, payload)

    @app.post("/api/leagues/{league_id}/predict/fixtures")
    def fixture_prediction(league_id: str, payload: Dict[str, Any] = Body(...)):
        return _submit(f"Prediciendo partidos para {league_id}", services.fixture_prediction, league_id, payload)

    @app.post("/api/leagues/{league_id}/analysis/{analysis_type}")
    def analysis_plot(league_id: str, analysis_type: str, payload: Dict[str, Any] = Body(default={})):
        return _submit(f"Generando {analysis_type}", services.analysis_plot, league_id, analysis_type, payload)

    @app.post("/api/leagues/{league_id}/models/{model_id}/explain/{plot_type}")
    def explain_plot(league_id: str, model_id: str, plot_type: str, payload: Dict[str, Any] = Body(default={})):
        return _submit(f"Generando {plot_type}", services.explain_plot, league_id, model_id, plot_type, payload)

    @app.get("/api/config/browser")
    def browser_config():
        return _wrap(services.browser_config)

    @app.put("/api/config/browser")
    def update_browser_config(payload: Dict[str, Any] = Body(...)):
        return _wrap(services.update_browser_config, payload)

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
    status_code = 400 if isinstance(exc, CLIError) else 500
    return JSONResponse(status_code=status_code, content={"ok": False, "data": {}, "error": f"{exc.__class__.__name__}: {exc}"})


def _host_without_port(host_header: str) -> str:
    if host_header.startswith("[::1]"):
        return "::1"
    if ":" in host_header:
        return host_header.rsplit(":", maxsplit=1)[0]
    return host_header


app = create_app()
