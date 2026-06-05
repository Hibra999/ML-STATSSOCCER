import ast

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
