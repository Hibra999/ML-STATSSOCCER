import ast
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_web_server_binds_localhost_only():
    from src.web.config import LOCAL_HOST, LOCAL_PORT

    assert LOCAL_HOST == "127.0.0.1"
    assert LOCAL_PORT == 5050


def test_server_source_does_not_bind_all_interfaces():
    source = open("src/web/server.py", "r", encoding="utf-8").read()
    tree = ast.parse(source)
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}

    assert "0.0.0.0" not in constants


def test_fastapi_app_imports_when_dependency_available():
    pytest.importorskip("fastapi")
    from src.web.app import create_app

    app = create_app()
    paths = {route.path for route in app.routes}

    assert "/" in paths
    assert "/api/health" in paths
    assert "/api/leagues" in paths
    assert "/api/dashboard/fixtures" in paths
    assert "/api/leagues/{league_id}/fixtures/upcoming" in paths
    assert "/api/leagues/{league_id}/predict/manual" not in paths
    assert "/favicon.ico" in paths
    assert "/assets" in paths


def test_web_catalog_has_flags_and_defaults():
    from src.web import services

    catalog = services.catalog_leagues()

    assert catalog
    assert all(item["flag_url"].startswith("/assets/graphics/countries/") for item in catalog)
    assert all(item["default_league_id"] for item in catalog)
    assert all(value is not None for item in catalog for value in item.values())


def test_web_browser_config_supports_brave():
    from src.web import services

    config = services.browser_config()

    assert "brave" in services.SUPPORTED_BROWSERS
    assert "brave_binary" in config
    assert config["brave_binary"] is not None


def test_web_model_specs_only_expose_boosting_models():
    from src.web import services

    specs = services.model_specs()
    keys = {spec["key"] for spec in specs}

    assert keys == {"ngboost", "catboost", "lightgbm", "xgboost"}
    assert all(spec["tunables"] for spec in specs)


def test_training_progress_payload_shape():
    from src.web import services

    payloads = []
    services.emit_training_progress(payloads.append, "tuning", 2, 5, "Optuna en ejecucion", best_value=0.8)

    assert payloads == [{
        "stage": "tuning",
        "current": 2,
        "total": 5,
        "current_trial": 2,
        "total_trials": 5,
        "percent": 40,
        "message": "Optuna en ejecucion",
        "best_value": 0.8,
    }]


def test_predict_ui_uses_automatic_fixtures_only():
    index_source = open("src/web/static/index.html", "r", encoding="utf-8").read()
    app_source = open("src/web/static/app.js", "r", encoding="utf-8").read()

    assert index_source.index("dashboard-fixtures") < index_source.index("metric-grid")
    assert "manual-form" not in index_source
    assert "predict/manual" not in app_source
    assert 'type="file"' not in index_source
    assert "fixtures-browser" in index_source
    assert "fixtures-picker" in index_source
    assert "dashboard-fixtures" in index_source
    assert "/static/app.js?v=" in index_source
    assert "renderJobs();" in app_source
    assert "dashboardFixtureSummaryHtml" in app_source


def test_confusion_matrix_payload_for_result_target():
    from src.preprocessing.utils.target import TargetType
    from src.web import services

    matrix = services.confusion_matrix_dataframe(
        target_type=TargetType.RESULT,
        y_true=np.array([0, 0, 1, 2, 2]),
        y_pred=np.array([0, 1, 1, 2, 0]),
    )

    assert matrix.to_dict(orient="records") == [
        {"Real": "H", "Pred H": 1, "Pred D": 1, "Pred A": 0},
        {"Real": "D", "Pred H": 0, "Pred D": 1, "Pred A": 0},
        {"Real": "A", "Pred H": 1, "Pred D": 0, "Pred A": 1},
    ]


def test_fixture_rows_from_payload_requires_selected_rows():
    from src.web import services

    with pytest.raises(Exception, match="Selecciona"):
        services.fixture_rows_from_payload([])


def test_dashboard_fixtures_uses_catalog_leagues_and_limits(monkeypatch):
    from src.web import services

    class FakeLeagueDatabase:
        def __init__(self):
            self.leagues = [
                SimpleNamespace(country="Mexico", name="Liga-MX", fixture="https://example.test/fixtures")
            ]

    def fake_scrape_dashboard_upcoming_fixtures(**kwargs):
        return pd.DataFrame([
            {"Date": "2026-06-05", "Dia": "Viernes", "Hora MX": "18:00", "Home": "A", "Away": "B", "Fuente": "FotMob"},
            {"Date": "2026-06-06", "Dia": "Sabado", "Hora MX": "20:00", "Home": "C", "Away": "D", "Fuente": "FotMob"},
        ])

    monkeypatch.setattr(services, "LeagueDatabase", FakeLeagueDatabase)
    monkeypatch.setattr(services, "scrape_dashboard_upcoming_fixtures", fake_scrape_dashboard_upcoming_fixtures)

    result = services.dashboard_fixtures(limit=1, days=7)

    assert result["fixtures"]["total"] == 1
    assert result["summary"]["catalog_total"] == 1
    assert result["summary"]["attempted"] == 1
    assert result["summary"]["with_fixtures"] == 1
    assert result["summary"]["shown"] == 1
    assert result["fixtures"]["rows"][0]["Catalogo"] == 1
    assert result["fixtures"]["rows"][0]["Liga"] == "Mexico / Liga-MX"
    assert result["fixtures"]["rows"][0]["Hora MX"] == "18:00"
    assert result["fixtures"]["rows"][0]["Fuente"] == "FotMob"


def test_fotmob_provider_parses_upcoming_matches_in_mx_time():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.network.fixtures.fotmob import parse_fotmob_upcoming_fixtures

    payload = {
        "fixtures": {
            "allMatches": [
                {
                    "home": {"name": "FC Tokyo"},
                    "away": {"name": "Cerezo Osaka"},
                    "status": {"utcTime": "2026-06-06T05:00:00Z", "finished": False, "cancelled": False},
                },
                {
                    "home": {"name": "Finished"},
                    "away": {"name": "Match"},
                    "status": {"utcTime": "2026-06-06T03:00:00Z", "finished": True},
                },
            ],
        },
    }

    result = parse_fotmob_upcoming_fixtures(
        payload=payload,
        days=7,
        now=datetime(2026, 6, 5, 12, 0, tzinfo=ZoneInfo("America/Mexico_City")),
        source_name="FotMob: J. League",
    )

    assert result.to_dict(orient="records") == [{
        "Date": "2026-06-05",
        "Dia": "Viernes",
        "Hora MX": "23:00",
        "Home": "FC Tokyo",
        "Away": "Cerezo Osaka",
        "Fuente": "FotMob: J. League",
    }]


def test_fotmob_resolves_catalog_alias_from_all_leagues():
    from types import SimpleNamespace

    from src.network.fixtures import fotmob

    class FakeSession:
        def get(self, url, headers=None, timeout=None):
            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "countries": [{
                            "ccode": "JPN",
                            "leagues": [{
                                "id": 223,
                                "name": "J. League",
                                "localizedName": "J. League",
                                "pageUrl": "/leagues/223/overview/j-league",
                                "ccode": "JPN",
                            }],
                        }],
                    }

            return Response()

    source = fotmob.resolve_fotmob_league(
        league=SimpleNamespace(country="Japan", name="J-1"),
        session=FakeSession(),
    )

    assert source["id"] == 223
    assert source["name"] == "J. League"


def test_dashboard_error_notes_are_grouped():
    from src.web import services

    notes = services.compact_dashboard_errors([
        {"league": "Argentina / Primera-Division", "message": "No se pudo cargar FootyStats."},
        {"league": "Belgium / Jupiler-League", "message": "No se pudo cargar FootyStats."},
        {"league": "Brazil / Serie-A", "message": "No se pudo cargar FootyStats."},
        {"league": "China / Super-League", "message": "No se pudo cargar FootyStats."},
    ])

    assert len(notes) == 1
    assert notes[0].startswith("4 ligas fallaron")
    assert "Argentina / Primera-Division" in notes[0]
    assert "y 1 mas" in notes[0]


def test_dashboard_error_cleaning_groups_request_object_addresses():
    from src.web import services

    cleaned = [
        services.clean_error_text(RuntimeError("No se pudo cargar FotMob: <urllib3.connection.HTTPSConnection object at 0xabc123>")),
        services.clean_error_text(RuntimeError("No se pudo cargar FotMob: <urllib3.connection.HTTPSConnection object at 0xdef456>")),
    ]
    notes = services.compact_dashboard_errors([
        {"league": "Japan / J-1", "message": cleaned[0]},
        {"league": "USA / MLS", "message": cleaned[1]},
    ])

    assert len(notes) == 1
    assert notes[0].startswith("2 ligas fallaron")
