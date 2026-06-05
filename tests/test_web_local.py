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

    assert "manual-form" not in index_source
    assert "predict/manual" not in app_source
    assert 'type="file"' not in index_source
    assert "fixtures-browser" in index_source
    assert "fixtures-picker" in index_source
    assert "dashboard-fixtures" in index_source


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


def test_dashboard_fixtures_uses_saved_leagues_and_limits(monkeypatch):
    from src.web import services

    class FakeLeagueDatabase:
        def __init__(self):
            self.index = {
                "mx": SimpleNamespace(country="Mexico", fixture="https://example.test/fixtures")
            }

        def get_league_ids(self):
            return ["mx"]

        def load_league(self, league_id):
            return pd.DataFrame({"Home": ["A"], "Away": ["B"]})

    def fake_scrape_upcoming_fixtures(**kwargs):
        return pd.DataFrame([
            {"Date": "2026-06-05", "Hora MX": "18:00", "Home": "A", "Away": "B", "1": 1.8, "X": 3.2, "2": 4.0},
            {"Date": "2026-06-06", "Hora MX": "20:00", "Home": "C", "Away": "D", "1": 2.1, "X": 3.1, "2": 3.4},
        ])

    monkeypatch.setattr(services, "LeagueDatabase", FakeLeagueDatabase)
    monkeypatch.setattr(services, "scrape_upcoming_fixtures", fake_scrape_upcoming_fixtures)

    result = services.dashboard_fixtures(limit=1, days=7)

    assert result["fixtures"]["total"] == 1
    assert result["fixtures"]["rows"][0]["Liga"] == "mx"
    assert result["fixtures"]["rows"][0]["Hora MX"] == "18:00"
