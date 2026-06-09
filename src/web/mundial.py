from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.web import mundial_services as services
from src.web.config import ALLOWED_HOSTS, LOCAL_HOST


MUNDIAL_PORT = 5052


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

    app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
    app.mount("/assets", StaticFiles(directory="storage"), name="assets")

    @app.get("/")
    def index():
        return FileResponse("src/web/static/mundial.html")

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

    @app.get("/api/mundial/lineups")
    def lineups(refresh: bool = False):
        return _wrap(services.lineups, refresh)

    @app.get("/api/mundial/fixtures/{fixture_id}/lineups")
    def fixture_lineup(fixture_id: str, refresh: bool = False):
        return _wrap(services.fixture_lineup, fixture_id, refresh)

    @app.post("/api/mundial/fixtures/{fixture_id}/autodetect")
    def autodetect_fixture(fixture_id: str, payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.autodetect_fixture, fixture_id, payload)

    @app.post("/api/mundial/fixtures/{fixture_id}/lineups/refresh")
    def refresh_fixture_lineup(fixture_id: str, payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.refresh_fixture_lineup, fixture_id, payload)

    @app.post("/api/mundial/fixtures/{fixture_id}/lineups/link")
    def link_lineup(fixture_id: str, payload: Dict[str, Any] = Body(...)):
        return _wrap(services.link_lineup, fixture_id, payload)

    @app.post("/api/mundial/lineups/auto-refresh")
    def auto_refresh_lineups(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.auto_refresh, payload)

    @app.get("/api/mundial/fixtures/{fixture_id}/player-stats")
    def fixture_player_stats(fixture_id: str, refresh: bool = False):
        return _wrap(services.fixture_player_stats, fixture_id, refresh)

    @app.get("/api/mundial/player-features")
    def player_features(refresh: bool = False):
        return _wrap(services.player_features, refresh)

    @app.post("/api/mundial/training/download-kaggle")
    def training_download(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.training_download, payload)

    @app.get("/api/mundial/training/dataset")
    def training_dataset():
        return _wrap(services.training_dataset)

    @app.post("/api/mundial/training/train")
    def training_train(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.training_train, payload)

    @app.get("/api/mundial/training/status")
    def training_status():
        return _wrap(services.training_status)

    @app.get("/api/mundial/training/options")
    def training_options():
        return _wrap(services.training_options)

    @app.post("/api/mundial/predict-match")
    def predict_match(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.predict_match, payload)

    @app.post("/api/mundial/simulate")
    def simulate(payload: Dict[str, Any] = Body(default={})):
        return _wrap(services.simulate, payload)

    @app.get("/api/mundial/procedure")
    def procedure():
        return _wrap(services.procedure)

    return app


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
