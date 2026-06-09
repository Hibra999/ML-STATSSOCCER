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
    assert "/api/worldcup/overview" not in paths
    assert "/api/worldcup/simulate" not in paths
    assert "/api/worldcup/lineups" not in paths
    assert "/api/leagues/{league_id}/fixtures/upcoming" in paths
    assert "/api/leagues/{league_id}/predict/manual" not in paths
    assert "/favicon.ico" in paths
    assert "/assets" in paths


def test_mundial_app_imports_as_independent_fastapi_app():
    pytest.importorskip("fastapi")
    from src.web.mundial import create_mundial_app

    app = create_mundial_app()
    paths = {route.path for route in app.routes}

    assert "/" in paths
    assert "/api/health" in paths
    assert "/api/mundial/overview" in paths
    assert "/api/mundial/simulate" in paths
    assert "/api/mundial/lineups" in paths
    assert "/api/mundial/fixtures/{fixture_id}/lineups" in paths
    assert "/api/mundial/fixtures/{fixture_id}/autodetect" in paths
    assert "/api/mundial/lineups/auto-refresh" in paths
    assert "/api/mundial/fixtures/{fixture_id}/player-stats" in paths
    assert "/api/mundial/player-features" in paths
    assert "/api/mundial/training/download-kaggle" in paths
    assert "/api/mundial/training/prepare-etl" in paths
    assert "/api/mundial/training/dataset" in paths
    assert "/api/mundial/training/train" in paths
    assert "/api/mundial/training/status" in paths
    assert "/api/mundial/training/options" in paths
    assert "/api/mundial/models" in paths
    assert "/api/mundial/models/train" in paths
    assert "/api/mundial/models/select" in paths
    assert "/api/mundial/models/{model_id}" in paths
    assert "/api/mundial/maintenance/clear" in paths
    assert "/api/mundial/predict-match" in paths
    assert "/api/mundial/predict-upcoming" in paths
    assert "/api/mundial/procedure" not in paths
    assert "/api/jobs/{job_id}" in paths
    assert "/api/worldcup/overview" not in paths
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


def test_requirements_use_windows_tensorflow_io_marker():
    source = open("requirements.txt", "r", encoding="utf-8").read()

    assert 'tensorflow-io-gcs-filesystem==0.37.1; platform_system != "Windows"' in source
    assert 'tensorflow-io-gcs-filesystem==0.31.0; platform_system == "Windows"' in source


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
    assert "Mundial 2026" not in index_source
    assert "/api/worldcup/simulate" not in app_source
    assert "worldcup-lineup-fixture" not in index_source
    assert "worldcup-use-lineups" not in index_source
    assert "/static/app.js?v=" in index_source
    assert "renderJobs();" in app_source
    assert "dashboardFixtureSummaryHtml" in app_source


def test_mundial_ui_is_standalone_and_personalizable():
    html_source = open("src/web/static/mundial.html", "r", encoding="utf-8").read()
    app_source = open("src/web/static/mundial.js", "r", encoding="utf-8").read()

    assert "Mundial 2026" in html_source
    assert "Datos y procedimiento" not in html_source
    assert "procedure-list" not in html_source
    assert "renderProcedure" not in app_source
    assert "worldcup-view active" in html_source
    nav_source = html_source.split("<nav>", 1)[1].split("</nav>", 1)[0]
    nav_order = [
        "Resumen",
        "Grupos",
        "Calendario",
        "11 Iniciales",
        "Entrenamiento y Modelo",
        "Predicciones Futuras",
        "Datos",
    ]
    assert [nav_source.index(label) for label in nav_order] == sorted(nav_source.index(label) for label in nav_order)
    assert "scrollIntoView" not in app_source
    assert "switchWorldcupView" in app_source
    assert "data-section=\"predicciones\"" in html_source
    assert "Calendario" in html_source
    assert "Modelo existente" in html_source
    assert "Predicciones Futuras" in html_source
    assert "Entrenamiento y Modelo" in html_source
    assert "Algoritmo boosting" in html_source
    assert "mundial-xgb-hibrido" in html_source
    assert "ML híbrido" in html_source
    assert "model-active-select" in html_source
    assert "worldcup-new-model" in html_source
    assert "model-load" in html_source
    assert "worldcup-clear-cache" in html_source
    assert "worldcup-model-id" in html_source
    assert "upcoming-model-select" in html_source
    assert "hero-hardware" in html_source
    assert "training-hardware" not in html_source
    assert "sim-history-weight" in html_source
    assert "sim-recency-weight" in html_source
    assert "sim-host-advantage" in html_source
    assert "lineup-fixture" in html_source
    assert "lineup-stage" in html_source
    assert "lineup-autodetect" in html_source
    assert "sim-use-player-features" in html_source
    assert "sim-use-ml-model" in html_source
    assert "training-train" in html_source
    assert "worldcup-training-progress" in html_source
    assert "worldcup-simulation-progress" in html_source
    assert "training-retrain-base" in html_source
    assert "training-retrain-players" in html_source
    assert "training-walkforward-notice" in html_source
    assert "worldcup-model-type" in html_source
    assert "worldcup-tuning-enabled" in html_source
    assert "worldcup-device" in html_source
    assert "worldcup-n-jobs" in html_source
    assert "training-prepare-etl" in html_source
    assert "training-tuning-lock-status" in html_source
    assert "training-model-params" in html_source
    assert "training-model-state" in html_source
    assert "dataset-summary" in html_source
    assert "training-etl-flow" in html_source
    assert "training-confusion-matrix" in html_source
    assert "training-tuning-flow" in html_source
    assert "Siempre 1X2 + O/U 2.5" in html_source
    assert "upcoming-predict-limit" in html_source
    assert "upcoming-predictions" in html_source
    assert "hero-countdown" in html_source
    assert "hero-next-grid" in html_source
    assert "simulation-summary" in html_source
    assert "upcoming-team" in app_source
    assert "flagHtml(homeAsset)" in app_source
    assert "flagHtml(awayAsset)" in app_source
    assert "market_sources" in app_source
    assert "marketBadgeText" in app_source
    assert "source-strip" in app_source
    assert "predict-match-btn" not in html_source
    assert "worldcup-target" not in html_source
    assert "lineup-features-table" in html_source
    assert "/api/mundial/simulate" in app_source
    assert "/api/mundial/player-features" in app_source
    assert "/api/mundial/models" in app_source
    assert "/api/mundial/models/train" in app_source
    assert "/api/mundial/training/prepare-etl" in app_source
    assert "/api/mundial/models/select" in app_source
    assert "/api/mundial/maintenance/clear" in app_source
    assert "trainingPayload" in app_source
    assert "paramsTable" in app_source
    assert "evalStrategyLabel" in app_source
    assert "renderConfusionMatrix" in app_source
    assert "renderEtlFlow" in app_source
    assert "renderTuningFlow" in app_source
    assert "dual_markets" in app_source
    assert "hibrido" in app_source
    assert "market-panel" in app_source
    assert "renderWalkForwardNotice" in app_source
    assert "renderHeroHardware" in app_source
    assert "trackWorldcupJob" in app_source
    assert "/api/jobs/${jobId}" in app_source
    assert "runUpcomingPredictions" in app_source
    assert "/api/mundial/predict-upcoming" in app_source
    assert "/api/mundial/procedure" not in app_source
    assert "renderHeroCountdown" in app_source
    assert "heroNextCardHtml" in app_source
    assert "/api/mundial/fixtures/${encodeURIComponent(fixtureId)}/autodetect" in app_source
    assert "/api/worldcup/simulate" not in app_source
    assert "player-photo" in app_source


def test_worldcup_training_options_expose_boosting_models_and_hardware():
    from src.worldcup.training import default_model_id, training_options

    options = training_options()
    keys = {model["key"] for model in options["models"]}

    assert keys == {"ngboost", "catboost", "lightgbm", "xgboost"}
    assert options["hardware"]["cpu_count"] >= 1
    assert options["hardware"]["default_n_jobs"] == -1
    assert options["defaults"]["model_type"] == "xgboost"
    assert options["defaults"]["training_target"] == "result"
    assert options["defaults"]["market_mode"] == "dual_markets"
    assert [target["key"] for target in options["targets"]] == ["dual_markets"]
    assert default_model_id("xgboost", "dual_markets") == "mundial-xgb-hibrido"


def test_worldcup_accelerator_cpu_sanitizes_float32_matrix():
    from src.worldcup.accelerators import prepare_numeric_matrix, training_acceleration_backend

    raw = np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float64)
    matrix = prepare_numeric_matrix(raw, use_cuda=False)
    backend = training_acceleration_backend(prefer_cuda=False)

    assert matrix.dtype == np.float32
    assert matrix.tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert backend["backend"] == "cpu"
    assert "numba_cuda_available" in backend


def test_worldcup_xgboost_cpu_hist_training_and_cuda_fallback(monkeypatch):
    from src.worldcup import training

    monkeypatch.setattr(training, "detect_hardware", lambda: {
        "cpu_count": 2,
        "default_n_jobs": -1,
        "cuda_available": False,
        "cuda_devices": [],
        "cuda_error": "sin cuda en test",
        "numba_cuda_available": False,
        "device_default": "cpu",
    })
    x = pd.DataFrame({
        "a": [0.0, 1.0, 2.0, 3.0, np.nan, np.inf],
        "b": [1.0, 0.0, 1.0, 0.0, 2.0, -np.inf],
    })
    y = pd.Series([0, 1, 2, 0, 1, 2])
    result = training.fit_configured_classifier(
        x_train=x,
        y_train=y,
        model_key="xgboost",
        params={"n_estimators": 5, "max_depth": 2, "learning_rate": 0.2},
        n_jobs=1,
        requested_device="cuda",
        seed=7,
        num_classes=3,
    )
    params = result["classifier"].get_params()

    assert result["device"] == "cpu"
    assert result["accelerator"]["backend"] == "cpu"
    assert params["tree_method"] == "hist"
    assert params["device"] == "cpu"
    assert any("CUDA no disponible" in warning for warning in result["warnings"])


def test_worldcup_fallback_has_2026_groups_opener_and_bracket():
    from src.worldcup.data import fallback_tournament_2026, group_stage_matches, groups_dataframe, knockout_matches

    tournament = fallback_tournament_2026()
    groups = groups_dataframe(tournament)
    group_matches = group_stage_matches(tournament)
    knockouts = knockout_matches(tournament)

    assert groups.shape[0] == 48
    assert group_matches[0]["date"] == "2026-06-11"
    assert group_matches[0]["team1"] == "Mexico"
    assert group_matches[0]["team2"] == "South Africa"
    assert len(group_matches) == 72
    assert len(knockouts) == 31


def test_worldcup_match_probabilities_are_normalized():
    from src.worldcup.data import FALLBACK_2026_GROUPS
    from src.worldcup.model import WorldCupModel

    teams = [team for group in FALLBACK_2026_GROUPS.values() for team in group]
    model = WorldCupModel.from_history(pd.DataFrame(), teams=teams)
    probabilities = model.match_probabilities("Mexico", "South Africa")

    assert probabilities["home"] + probabilities["draw"] + probabilities["away"] == pytest.approx(1, abs=0.01)
    assert probabilities["over25"] + probabilities["under25"] == pytest.approx(1, abs=0.01)
    assert probabilities["lambda1"] > 0
    assert probabilities["lambda2"] > 0


def test_worldcup_simulation_returns_advancement_probabilities():
    from src.worldcup.data import FALLBACK_2026_GROUPS, fallback_tournament_2026
    from src.worldcup.model import WorldCupModel
    from src.worldcup.simulation import simulate_worldcup

    tournament = fallback_tournament_2026()
    teams = [team for group in FALLBACK_2026_GROUPS.values() for team in group]
    model = WorldCupModel.from_history(pd.DataFrame(), teams=teams)
    result = simulate_worldcup(tournament, model, iterations=100, seed=7)

    assert result["advancement"].shape[0] == 48
    assert result["matches"].shape[0] == 72
    assert "Pasa grupo %" in result["advancement"].columns
    assert "Over 2.5 %" in result["matches"].columns
    assert result["advancement"]["Campeon %"].sum() == pytest.approx(100, abs=0.01)


def test_worldcup_simulation_emits_progress_payloads():
    from src.worldcup.data import FALLBACK_2026_GROUPS, fallback_tournament_2026
    from src.worldcup.model import WorldCupModel
    from src.worldcup.simulation import simulate_worldcup

    tournament = fallback_tournament_2026()
    teams = [team for group in FALLBACK_2026_GROUPS.values() for team in group]
    model = WorldCupModel.from_history(pd.DataFrame(), teams=teams)
    payloads = []

    simulate_worldcup(tournament, model, iterations=100, seed=7, progress_callback=payloads.append)

    assert payloads
    assert payloads[-1]["stage"] == "simulation"
    assert payloads[-1]["current"] == 100
    assert payloads[-1]["total"] == 100
    assert payloads[-1]["percent"] == 100
    assert payloads[-1]["message"] == "Monte Carlo completado"


def test_worldcup_lanus_lineup_normalization_extracts_starting_elevens():
    from src.worldcup.lanus_provider import LINEUP_STATUSES, normalize_lanus_lineups

    fixture = pd.Series({"No.": 1, "Fecha": "2026-06-11", "Grupo": "Group A", "Equipo 1": "Mexico", "Equipo 2": "South Africa"})
    raw = {
        "confirmed": True,
        "home": {"formation": "4-3-3", "players": [_fake_lanus_player(index, False) for index in range(1, 12)] + [_fake_lanus_player(12, True)]},
        "away": {"formation": "4-2-3-1", "players": [_fake_lanus_player(index, False) for index in range(21, 32)]},
    }

    result = normalize_lanus_lineups(raw, fixture=fixture, fixture_key="1", match_url="https://www.sofascore.com/test#id:1", fetched_at="2026-06-10T18:00:00+00:00")

    assert result["status"] == LINEUP_STATUSES["official"]
    assert result["starters_home"] == 11
    assert result["starters_away"] == 11
    assert len([player for player in result["players"] if player["team"] == "Mexico" and player["starter"]]) == 11
    assert result["formation_home"] == "4-3-3"
    assert all(isinstance(player["stats"], dict) for player in result["players"])


def test_worldcup_sofascore_event_matching_finds_fixture():
    from src.worldcup.lanus_provider import best_event_match, sofa_event_url, team_similarity

    events = [
        {"id": 123, "homeTeam": {"name": "Mexico"}, "awayTeam": {"name": "South Africa"}},
        {"id": 456, "homeTeam": {"name": "Canada"}, "awayTeam": {"name": "Qatar"}},
    ]

    event, confidence, reverse = best_event_match(events, "Mexico", "South Africa")

    assert event["id"] == 123
    assert confidence >= 0.99
    assert reverse is False
    assert team_similarity("USA", "United States") == 1.0
    assert sofa_event_url("123", "Mexico", "South Africa").endswith("#id:123")


def test_worldcup_autodetect_fixture_event_caches_match(tmp_path, monkeypatch):
    from src.worldcup import lanus_provider
    from src.worldcup.data import fallback_tournament_2026

    monkeypatch.setattr(lanus_provider, "LINEUPS_ROOT", tmp_path / "lineups")
    monkeypatch.setattr(lanus_provider, "LINEUP_LINKS_FILE", tmp_path / "lineups" / "links.json")
    monkeypatch.setattr(lanus_provider, "SOFASCORE_ROOT", tmp_path / "sofascore")
    monkeypatch.setattr(lanus_provider, "SOFASCORE_EVENTS_FILE", tmp_path / "sofascore" / "events.json")

    def fake_fetch(fixture):
        return {
            "fixture_id": str(fixture["No."]),
            "date": fixture["Fecha"],
            "home": "Mexico",
            "away": "South Africa",
            "event_id": "987",
            "match_url": "https://www.sofascore.com/football/match/mexico-south-africa#id:987",
            "confidence": 1.0,
            "status": "Detectado",
        }

    monkeypatch.setattr(lanus_provider, "fetch_best_sofascore_event", fake_fetch)

    result = lanus_provider.autodetect_fixture_event(fallback_tournament_2026(), fixture_id=1)
    cached = lanus_provider.autodetect_fixture_event(fallback_tournament_2026(), fixture_id=1)

    assert result["event_id"] == "987"
    assert cached == result
    assert lanus_provider.read_lineup_links()["1"].endswith("#id:987")


def test_worldcup_autodetect_fixture_event_handles_provider_failures(tmp_path, monkeypatch):
    from src.worldcup import lanus_provider
    from src.worldcup.data import fallback_tournament_2026

    monkeypatch.setattr(lanus_provider, "LINEUPS_ROOT", tmp_path / "lineups")
    monkeypatch.setattr(lanus_provider, "LINEUP_LINKS_FILE", tmp_path / "lineups" / "links.json")
    monkeypatch.setattr(lanus_provider, "SOFASCORE_ROOT", tmp_path / "events")
    monkeypatch.setattr(lanus_provider, "SOFASCORE_EVENTS_FILE", tmp_path / "events" / "events.json")
    monkeypatch.setattr(lanus_provider, "fetch_best_fotmob_event", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fotmob down")))
    monkeypatch.setattr(lanus_provider, "fetch_best_sofascore_event", lambda fixture: (_ for _ in ()).throw(RuntimeError("sofascore down")))

    result = lanus_provider.autodetect_fixture_event(fallback_tournament_2026(), fixture_id=1, refresh=True)

    assert result["status"] == "Pendiente"
    assert result["provider"] == "none"
    assert result["event_id"] == ""
    assert "fotmob down" in result["error"]
    assert "sofascore down" in result["error"]
    assert len(result["source_attempts"]) == 2


def test_worldcup_fotmob_provider_extracts_match_and_lineup():
    from src.worldcup.fotmob_provider import extract_fotmob_matches, normalize_fotmob_players
    from src.worldcup.lanus_provider import team_similarity

    payload = {
        "matches": [
            {"id": 11, "home": {"name": "Mexico"}, "away": {"name": "South Africa"}},
            {"id": 12, "home": {"name": "Canada"}, "away": {"name": "Qatar"}},
        ]
    }
    matches = extract_fotmob_matches(payload)
    assert len(matches) == 2
    assert team_similarity("Mexico", matches[0]["home"]["name"]) == 1.0

    details = {
        "content": {
            "lineup": {
                "confirmed": True,
                "lineup": [
                    {"teamName": "Mexico", "formation": "4-3-3", "players": [{"id": index, "name": f"MEX {index}", "position": "M", "shirt": index, "rating": 7.0} for index in range(1, 12)]},
                    {"teamName": "South Africa", "formation": "4-4-2", "players": [{"id": 100 + index, "name": f"RSA {index}", "position": "D", "shirt": index, "rating": 6.5} for index in range(1, 12)]},
                ],
            }
        }
    }
    home_players, away_players, formation_home, formation_away, confirmed = normalize_fotmob_players(details, "Mexico", "South Africa")

    assert confirmed is True
    assert formation_home == "4-3-3"
    assert formation_away == "4-4-2"
    assert len([player for player in home_players if player["starter"]]) == 11
    assert len([player for player in away_players if player["starter"]]) == 11


def test_worldcup_lineup_fallback_pending_without_match_url(tmp_path, monkeypatch):
    from src.worldcup import lanus_provider
    from src.worldcup.data import fallback_tournament_2026

    monkeypatch.setattr(lanus_provider, "LINEUPS_ROOT", tmp_path)
    monkeypatch.setattr(lanus_provider, "LINEUP_LINKS_FILE", tmp_path / "links.json")

    result = lanus_provider.lineup_payload_for_fixture(fallback_tournament_2026(), fixture_id=1)

    assert result["status"] == lanus_provider.LINEUP_STATUSES["pending"]
    assert result["starters_home"] == 0
    assert result["starters_away"] == 0
    assert result["source"] == "unavailable:lineups"


def test_worldcup_lineup_rating_adjustments_use_safe_cached_lineups(tmp_path, monkeypatch):
    from src.worldcup import lanus_provider
    from src.worldcup.data import fallback_tournament_2026

    monkeypatch.setattr(lanus_provider, "LINEUPS_ROOT", tmp_path)
    monkeypatch.setattr(lanus_provider, "LINEUP_LINKS_FILE", tmp_path / "links.json")
    payload = {
        "fixture_id": "1",
        "date": "2026-06-11",
        "status": lanus_provider.LINEUP_STATUSES["official"],
        "home": "Mexico",
        "away": "South Africa",
        "players": [
            {"team": "Mexico", "starter": True, "rating": 7.2} for _ in range(11)
        ] + [
            {"team": "South Africa", "starter": True, "rating": 6.2} for _ in range(11)
        ],
        "fetched_at": "2026-06-10T18:00:00+00:00",
    }
    lanus_provider.write_lineup_cache(lanus_provider.lineup_cache_path("1"), payload)

    adjustments, notes = lanus_provider.lineup_rating_adjustments(fallback_tournament_2026())

    assert adjustments["Mexico"] > 0
    assert adjustments["South Africa"] < 0
    assert notes


def test_worldcup_player_features_dataframe_uses_safe_cached_lineups(tmp_path, monkeypatch):
    from src.worldcup import lanus_provider
    from src.worldcup.data import fallback_tournament_2026

    monkeypatch.setattr(lanus_provider, "LINEUPS_ROOT", tmp_path / "lineups")
    monkeypatch.setattr(lanus_provider, "LINEUP_LINKS_FILE", tmp_path / "lineups" / "links.json")
    monkeypatch.setattr(lanus_provider, "PLAYER_STATS_ROOT", tmp_path / "player_stats")
    positions = ["G", "D", "D", "D", "D", "M", "M", "M", "F", "F", "F"]
    payload = {
        "fixture_id": "1",
        "date": "2026-06-11",
        "group": "Group A",
        "status": lanus_provider.LINEUP_STATUSES["official"],
        "source": "LanusStats/SofaScore",
        "home": "Mexico",
        "away": "South Africa",
        "formation_home": "4-3-3",
        "formation_away": "4-3-3",
        "players": [
            {"team": "Mexico", "starter": True, "rating": 7.2, "position": position, "stats": {"minutesPlayed": 90}}
            for position in positions
        ] + [
            {"team": "South Africa", "starter": True, "rating": 6.2, "position": position, "stats": {"minutesPlayed": 90}}
            for position in positions
        ],
        "fetched_at": "2026-06-10T18:00:00+00:00",
    }
    lanus_provider.write_lineup_cache(lanus_provider.lineup_cache_path("1"), payload)

    features = lanus_provider.player_features_dataframe(fallback_tournament_2026())
    mexico = features[features["Equipo"] == "Mexico"].iloc[0]
    adjustments, notes = lanus_provider.player_feature_rating_adjustments(fallback_tournament_2026(), weight=1.0)

    assert mexico["Prediction safe"] == "Si"
    assert mexico["Titulares"] == 11
    assert mexico["Stats conocidos"] == 11
    assert mexico["XI rating prom"] == pytest.approx(7.2)
    assert mexico["DEF rating"] == pytest.approx(7.2)
    assert adjustments["Mexico"] > 0
    assert adjustments["South Africa"] < 0
    assert notes


def test_mundial_simulation_config_is_clamped():
    from src.web.mundial_services import simulation_config

    config = simulation_config({
        "iterations": 999999,
        "history_weight": 0,
        "recency_weight": 2,
        "host_advantage": -5,
        "max_goals": 99,
        "lineup_weight": 9,
        "player_feature_weight": 9,
        "ml_weight": 9,
        "use_lineups": True,
        "use_player_features": True,
        "use_ml_model": True,
    })

    assert config["iterations"] == 20000
    assert config["history_weight"] == 0.2
    assert config["recency_weight"] == 1.0
    assert config["host_advantage"] == 0.0
    assert config["max_goals"] == 14
    assert config["lineup_weight"] == 2.0
    assert config["player_feature_weight"] == 2.0
    assert config["ml_weight"] == 1.0
    assert config["use_lineups"] is True
    assert config["use_player_features"] is True
    assert config["use_ml_model"] is True


def test_worldcup_training_normalizes_trains_and_predicts(tmp_path, monkeypatch):
    from src.worldcup import training
    from src.worldcup.data import fallback_tournament_2026
    from src.worldcup.model import WorldCupModel

    monkeypatch.setattr(training, "KAGGLE_ROOT", tmp_path / "kaggle")
    monkeypatch.setattr(training, "WORLD_CUP_MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(training, "HYBRID_MODEL_FILE", tmp_path / "models" / "hybrid.pkl")
    monkeypatch.setattr(training, "HYBRID_MODEL_META_FILE", tmp_path / "models" / "hybrid.json")
    monkeypatch.setattr(training, "PREPARED_DATASET_FILE", tmp_path / "cache" / "prepared.pkl")
    monkeypatch.setattr(training, "PREPARED_DATASET_META_FILE", tmp_path / "cache" / "prepared.json")
    training.KAGGLE_ROOT.mkdir(parents=True)
    pd.DataFrame([
        {"home_team": "Mexico", "away_team": "South Africa", "home_goals": 2, "away_goals": 0},
        {"home_team": "South Africa", "away_team": "Mexico", "home_goals": 1, "away_goals": 1},
        {"home_team": "Mexico", "away_team": "Canada", "home_goals": 1, "away_goals": 2},
        {"home_team": "Canada", "away_team": "South Africa", "home_goals": 0, "away_goals": 1},
    ]).to_csv(training.KAGGLE_ROOT / "train.csv", index=False)
    pd.DataFrame([
        {"home_team": "Mexico", "away_team": "South Africa", "home_goals": 3, "away_goals": 1},
        {"home_team": "Canada", "away_team": "Mexico", "home_goals": 0, "away_goals": 0},
    ]).to_csv(training.KAGGLE_ROOT / "test.csv", index=False)
    pd.DataFrame([
        {"Team": "Mexico", "Rank": 14, "Goals": 10},
        {"Team": "South Africa", "Rank": 60, "Goals": 5},
        {"Team": "Canada", "Rank": 35, "Goals": 7},
    ]).to_csv(training.KAGGLE_ROOT / "teams.csv", index=False)

    status = training.dataset_status()
    prepared = training.prepare_training_dataset(force=True)
    status = training.dataset_status()
    result = training.train_hybrid_model(fallback_tournament_2026(), payload={"seed": 7, "n_estimators": 20, "model_id": "mex-test"})
    model = WorldCupModel.from_history(pd.DataFrame(), teams=["Mexico", "South Africa", "Canada"])
    prediction = training.predict_match_payload(fallback_tournament_2026(), model, fixture_id=1, use_ml_model=True, ml_weight=0.5)
    catalog = training.list_worldcup_models()

    assert status["trainable"] is True
    assert prepared["etl_ready"] is True
    assert status["etl_ready"] is True
    assert status["test_rows"] == 2
    assert status["prediction_rows"] == 0
    assert status["eval_strategy"] == "test_file"
    assert result["model"]["trained"] is True
    assert result["model"]["model_id"] == "mex-test"
    assert result["model"]["model_type"] == "xgboost"
    assert result["model"]["eval_strategy"] == "test_file"
    assert result["model"]["confusion_matrix"]["matrix"]
    assert result["model"]["etl_steps"]
    assert result["model"]["tuning_trace"]["enabled"] is False
    assert result["model"]["hardware"]["actual_device"] in {"cpu", "cuda"}
    assert result["eval_rows"] == 2
    assert prediction["fixture"]["home"] == "Mexico"
    assert set(prediction["probabilities"]) >= {"home", "draw", "away", "over25", "under25"}
    assert prediction["model_probs"]["ml_weight"] == 0.5
    assert prediction["model_probs"]["model_id"] == "mex-test"
    assert prediction["market_sources"]["result"]["source"] == "ML + Poisson"
    assert prediction["market_sources"]["over_under_25"]["source"] == "ML + Poisson"
    assert prediction["market_sources"]["over_under_25"]["uses_ml"] is True
    assert set(prediction["model_probs"]) >= {"poisson", "poisson_totals", "ml", "over_under_ml"}
    assert catalog["active_model_id"] == "mex-test"
    assert any(item["model_id"] == "mex-test" for item in catalog["models"])

    dual_result = training.train_hybrid_model(
        fallback_tournament_2026(),
        payload={"seed": 7, "n_estimators": 5, "market_mode": "dual_markets", "model_id": "mex-dual"},
    )
    dual_prediction = training.predict_match_payload(fallback_tournament_2026(), model, fixture_id=1, use_ml_model=True, ml_weight=0.5)
    dual_catalog = training.list_worldcup_models()

    assert dual_result["model"]["bundle"] is True
    assert dual_result["model"]["market_mode"] == "dual_markets"
    assert set(dual_result["model"]["market_models"]) == {"result", "over_under_25"}
    assert dual_result["model"]["markets"]["result"]["confusion_matrix"]["matrix"]
    assert dual_result["model"]["markets"]["over_under_25"]["confusion_matrix"]["matrix"]
    assert dual_prediction["model_probs"]["model_id"] == "mex-dual"
    assert dual_prediction["model_probs"]["ml"]
    assert dual_prediction["model_probs"]["over_under_ml"]
    assert dual_prediction["market_sources"]["result"]["source"] == "ML + Poisson"
    assert dual_prediction["market_sources"]["over_under_25"]["source"] == "ML + Poisson"
    assert dual_catalog["active_model_id"] == "mex-dual"
    assert any(item["model_id"] == "mex-dual" and item["bundle"] for item in dual_catalog["models"])
    assert not any(str(item["model_id"]).endswith("__result") for item in dual_catalog["models"])
    assert not any(str(item["model_id"]).endswith("__uo25") for item in dual_catalog["models"])


def test_worldcup_training_rejects_single_market_requests(tmp_path, monkeypatch):
    from src.worldcup import training
    from src.worldcup.data import fallback_tournament_2026

    monkeypatch.setattr(training, "KAGGLE_ROOT", tmp_path / "kaggle")
    monkeypatch.setattr(training, "WORLD_CUP_MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(training, "HYBRID_MODEL_FILE", tmp_path / "models" / "hybrid.pkl")
    monkeypatch.setattr(training, "HYBRID_MODEL_META_FILE", tmp_path / "models" / "hybrid.json")
    monkeypatch.setattr(training, "PREPARED_DATASET_FILE", tmp_path / "cache" / "prepared.pkl")
    monkeypatch.setattr(training, "PREPARED_DATASET_META_FILE", tmp_path / "cache" / "prepared.json")
    training.KAGGLE_ROOT.mkdir(parents=True)
    pd.DataFrame([
        {"home_team": "Mexico", "away_team": "South Africa", "home_goals": 2, "away_goals": 0},
        {"home_team": "South Africa", "away_team": "Mexico", "home_goals": 1, "away_goals": 1},
        {"home_team": "Mexico", "away_team": "Canada", "home_goals": 1, "away_goals": 2},
        {"home_team": "Canada", "away_team": "South Africa", "home_goals": 0, "away_goals": 1},
    ]).to_csv(training.KAGGLE_ROOT / "train.csv", index=False)
    training.prepare_training_dataset(force=True)

    with pytest.raises(training.WorldCupTrainingError, match="bundle dual"):
        training.train_hybrid_model(
            fallback_tournament_2026(),
            payload={"seed": 7, "n_estimators": 5, "training_target": "over_under_25", "market_mode": "over_under_25"},
        )


def test_worldcup_training_uses_team_strength_dataset_shape(tmp_path, monkeypatch):
    from src.worldcup import training
    from src.worldcup.data import fallback_tournament_2026
    from src.worldcup.model import WorldCupModel

    monkeypatch.setattr(training, "KAGGLE_ROOT", tmp_path / "kaggle")
    monkeypatch.setattr(training, "WORLD_CUP_MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(training, "HYBRID_MODEL_FILE", tmp_path / "models" / "hybrid.pkl")
    monkeypatch.setattr(training, "HYBRID_MODEL_META_FILE", tmp_path / "models" / "hybrid.json")
    monkeypatch.setattr(training, "PREPARED_DATASET_FILE", tmp_path / "cache" / "prepared.pkl")
    monkeypatch.setattr(training, "PREPARED_DATASET_META_FILE", tmp_path / "cache" / "prepared.json")
    training.KAGGLE_ROOT.mkdir(parents=True)
    pd.DataFrame([
        {"version": 2022, "team": "Mexico", "fifa_rank_pre_tournament": 12, "fifa_points_pre_tournament": 1600, "wins_last_4y": 20, "quarter_finalist": 1},
        {"version": 2022, "team": "South Africa", "fifa_rank_pre_tournament": 60, "fifa_points_pre_tournament": 1350, "wins_last_4y": 8, "quarter_finalist": 0},
        {"version": 2022, "team": "Brazil", "fifa_rank_pre_tournament": 1, "fifa_points_pre_tournament": 1800, "wins_last_4y": 30, "quarter_finalist": 1},
        {"version": 2022, "team": "Qatar", "fifa_rank_pre_tournament": 50, "fifa_points_pre_tournament": 1400, "wins_last_4y": 9, "quarter_finalist": 0},
        {"version": 2018, "team": "France", "fifa_rank_pre_tournament": 7, "fifa_points_pre_tournament": 1700, "wins_last_4y": 25, "quarter_finalist": 1},
        {"version": 2018, "team": "Panama", "fifa_rank_pre_tournament": 55, "fifa_points_pre_tournament": 1320, "wins_last_4y": 6, "quarter_finalist": 0},
    ]).to_csv(training.KAGGLE_ROOT / "train.csv", index=False)
    pd.DataFrame([
        {"version": 2026, "team": "Mexico", "fifa_rank_pre_tournament": 14, "fifa_points_pre_tournament": 1650, "wins_last_4y": 22, "quarter_finalist": ""},
        {"version": 2026, "team": "South Africa", "fifa_rank_pre_tournament": 58, "fifa_points_pre_tournament": 1360, "wins_last_4y": 10, "quarter_finalist": ""},
    ]).to_csv(training.KAGGLE_ROOT / "test.csv", index=False)

    status = training.dataset_status()
    prepared = training.prepare_training_dataset(force=True)
    status = training.dataset_status()
    result = training.train_hybrid_model(fallback_tournament_2026(), payload={"seed": 11, "n_estimators": 20})
    model = WorldCupModel.from_history(pd.DataFrame(), teams=["Mexico", "South Africa"])
    prediction = training.predict_match_payload(fallback_tournament_2026(), model, fixture_id=1, use_ml_model=True, ml_weight=0.5)

    assert prepared["etl_ready"] is True
    assert status["raw_training_mode"] == "team_strength"
    assert status["training_mode"] == "match_result"
    assert status["prepared_label_source"] == "historical_worldcup"
    assert status["target_column"] == "Label + OverUnder25"
    assert status["test_rows"] == 0
    assert status["prediction_rows"] == 2
    assert status["eval_rows"] > 0
    assert status["eval_strategy"] == "holdout_from_train"
    assert result["mode"] == "match_result"
    assert result["eval_strategy"] == "holdout_from_train"
    assert result["prediction_rows"] == 2
    assert result["model"]["target_column"] == "Label + OverUnder25"
    assert result["model"]["eval_strategy"] == "holdout_from_train"
    assert result["model"]["markets"]["result"]["confusion_matrix"]["labels"] == ["1 Local", "X Empate", "2 Visita"]
    assert result["model"]["markets"]["over_under_25"]["confusion_matrix"]["labels"] == ["Under 2.5", "Over 2.5"]
    assert result["model"]["etl_steps"]
    assert result["model"]["tuning_trace"]["steps"]
    assert result["model"]["model_label"] == "XGBoost"
    assert result["model"]["hardware"]["effective_n_jobs"] >= 1
    assert prediction["model_probs"]["ml"]
    assert prediction["model_probs"]["over_under_ml"]


def test_worldcup_predict_upcoming_returns_future_predictions(tmp_path, monkeypatch):
    from src.web import mundial_services

    result = mundial_services.predict_upcoming({"limit": 3, "use_ml_model": False})

    assert result["summary"]["requested"] == 3
    assert result["summary"]["returned"] == 3
    assert len(result["predictions"]) == 3
    assert result["table"]["total"] == 3
    assert set(result["predictions"][0]["probabilities"]) >= {"home", "draw", "away", "over25", "under25"}


def test_mundial_overview_exposes_highlight_countdown_and_next_grid():
    from src.web import mundial_services

    result = mundial_services.overview()

    assert result["opener"]["home"]["name"] == "Mexico"
    assert result["highlight"]["home"]["name"] == "Mexico"
    assert result["countdown_state"] in {"ready", "pending"}
    assert isinstance(result["next_matches"], list)
    assert len(result["next_matches"]) >= 1


def test_mundial_maintenance_clear_resets_runtime_and_preserves_base_sources(tmp_path, monkeypatch):
    from src.web import mundial_services
    from src.worldcup import training

    cache_root = tmp_path / "cache"
    models_root = tmp_path / "models"
    kaggle_root = tmp_path / "kaggle"
    lineups_root = tmp_path / "lineups"
    stats_root = tmp_path / "player_stats"
    sofascore_root = tmp_path / "sofascore"
    walk_root = tmp_path / "walk_forward"
    for root in (cache_root, models_root, kaggle_root, lineups_root, stats_root, sofascore_root, walk_root):
        root.mkdir(parents=True, exist_ok=True)
    (cache_root / "worldcup_2026.json").write_text("{}", encoding="utf-8")
    (cache_root / "players_2026.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (cache_root / "worldcup_training_prepared.pkl").write_text("x", encoding="utf-8")
    (cache_root / "worldcup_training_prepared.json").write_text("{}", encoding="utf-8")
    (kaggle_root / "train.csv").write_text("home_team,away_team,home_goals,away_goals\nMexico,South Africa,2,1\n", encoding="utf-8")
    (models_root / "dummy.json").write_text("{}", encoding="utf-8")
    (models_root / "dummy.pkl").write_text("x", encoding="utf-8")
    (lineups_root / "fixture_1.json").write_text("{}", encoding="utf-8")
    (stats_root / "fixture_1.json").write_text("{}", encoding="utf-8")
    (sofascore_root / "events.json").write_text("{}", encoding="utf-8")
    (walk_root / "matches.csv").write_text("fixture_id\n1\n", encoding="utf-8")

    monkeypatch.setattr(mundial_services, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(mundial_services, "WORLD_CUP_MODELS_ROOT", models_root)
    monkeypatch.setattr(mundial_services, "KAGGLE_ROOT", kaggle_root)
    monkeypatch.setattr(mundial_services, "LINEUPS_ROOT", lineups_root)
    monkeypatch.setattr(mundial_services, "PLAYER_STATS_ROOT", stats_root)
    monkeypatch.setattr(mundial_services, "SOFASCORE_ROOT", sofascore_root)
    monkeypatch.setattr(mundial_services, "WALK_FORWARD_ROOT", walk_root)
    monkeypatch.setattr(training, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(training, "KAGGLE_ROOT", kaggle_root)
    monkeypatch.setattr(training, "WORLD_CUP_MODELS_ROOT", models_root)
    monkeypatch.setattr(training, "HYBRID_MODEL_FILE", models_root / "hybrid.pkl")
    monkeypatch.setattr(training, "HYBRID_MODEL_META_FILE", models_root / "hybrid.json")
    monkeypatch.setattr(training, "WALK_FORWARD_ROOT", walk_root)
    monkeypatch.setattr(training, "WALK_FORWARD_MATCHES_FILE", walk_root / "matches.csv")
    monkeypatch.setattr(training, "WALK_FORWARD_PLAYERS_FILE", walk_root / "player_match_stats.csv")
    monkeypatch.setattr(training, "WALK_FORWARD_TEAM_FEATURES_FILE", walk_root / "team_match_features.csv")
    monkeypatch.setattr(training, "PREPARED_DATASET_FILE", cache_root / "worldcup_training_prepared.pkl")
    monkeypatch.setattr(training, "PREPARED_DATASET_META_FILE", cache_root / "worldcup_training_prepared.json")

    result = mundial_services.maintenance_clear({"clear_cache": True})

    assert (cache_root / "worldcup_2026.json").exists()
    assert not (cache_root / "players_2026.csv").exists()
    assert not (cache_root / "worldcup_training_prepared.pkl").exists()
    assert not (models_root / "dummy.json").exists()
    assert result["training"]["available"] is True
    assert result["models"]["models"] == []


def test_worldcup_match_feature_row_includes_history_trend_and_h2h_features():
    from src.worldcup import training
    from src.worldcup.model import WorldCupModel

    history = pd.DataFrame([
        {"Date": "2010-06-11", "Team 1": "Mexico", "Team 2": "South Africa", "G1": 1, "G2": 1, "Round": "Group", "Group": "A"},
        {"Date": "2014-06-17", "Team 1": "Mexico", "Team 2": "Cameroon", "G1": 1, "G2": 0, "Round": "Group", "Group": "A"},
        {"Date": "2018-06-17", "Team 1": "Mexico", "Team 2": "Germany", "G1": 1, "G2": 0, "Round": "Group", "Group": "F"},
        {"Date": "2002-06-02", "Team 1": "South Africa", "Team 2": "Paraguay", "G1": 2, "G2": 2, "Round": "Group", "Group": "B"},
        {"Date": "2010-06-22", "Team 1": "South Africa", "Team 2": "France", "G1": 2, "G2": 1, "Round": "Group", "Group": "A"},
    ])
    model = WorldCupModel.from_history(history, teams=["Mexico", "South Africa", "Cameroon", "Germany", "Paraguay", "France"])
    history_features = training.build_history_feature_table(history)
    matchup_features = training.build_matchup_feature_table(history)

    row = training.match_feature_row(
        model,
        pd.DataFrame(),
        "Mexico",
        "South Africa",
        history_team_features=history_features,
        matchup_features=matchup_features,
    )

    assert "poisson_home_win" in row
    assert "history_last_3_points_ppg_home" in row
    assert "history_trend_goal_diff_3_vs_10_diff" in row
    assert "h2h_matches" in row
    assert row["h2h_matches"] >= 1


def test_worldcup_feature_importance_vector_handles_nested_ngboost_shape():
    from src.worldcup import training

    raw = [
        np.array([0.12, -0.08]),
        np.array([0.03, 0.01]),
        np.array([-0.2, 0.15]),
    ]

    vector = training.feature_importance_vector(raw, 3)
    top = training.top_feature_importances(type("Fake", (), {"feature_importances_": raw})(), ["a", "b", "c"])

    assert vector.shape == (3,)
    assert np.all(vector >= 0)
    assert top[0]["feature"] == "c"
    assert top[0]["importance"] > 0


def test_worldcup_ngboost_dual_training_with_tuning_completes(tmp_path, monkeypatch):
    pytest.importorskip("ngboost")

    from src.worldcup import training
    from src.worldcup.data import fallback_tournament_2026

    monkeypatch.setattr(training, "KAGGLE_ROOT", tmp_path / "kaggle")
    monkeypatch.setattr(training, "WORLD_CUP_MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(training, "HYBRID_MODEL_FILE", tmp_path / "models" / "hybrid.pkl")
    monkeypatch.setattr(training, "HYBRID_MODEL_META_FILE", tmp_path / "models" / "hybrid.json")
    monkeypatch.setattr(training, "PREPARED_DATASET_FILE", tmp_path / "cache" / "prepared.pkl")
    monkeypatch.setattr(training, "PREPARED_DATASET_META_FILE", tmp_path / "cache" / "prepared.json")
    training.KAGGLE_ROOT.mkdir(parents=True)
    pd.DataFrame([
        {"home_team": "Mexico", "away_team": "South Africa", "home_goals": 2, "away_goals": 0},
        {"home_team": "South Africa", "away_team": "Mexico", "home_goals": 1, "away_goals": 1},
        {"home_team": "Mexico", "away_team": "Canada", "home_goals": 1, "away_goals": 2},
        {"home_team": "Canada", "away_team": "South Africa", "home_goals": 0, "away_goals": 1},
        {"home_team": "Canada", "away_team": "Mexico", "home_goals": 0, "away_goals": 0},
        {"home_team": "South Africa", "away_team": "Canada", "home_goals": 2, "away_goals": 2},
    ]).to_csv(training.KAGGLE_ROOT / "train.csv", index=False)
    pd.DataFrame([
        {"home_team": "Mexico", "away_team": "South Africa", "home_goals": 3, "away_goals": 1},
        {"home_team": "Canada", "away_team": "Mexico", "home_goals": 0, "away_goals": 0},
    ]).to_csv(training.KAGGLE_ROOT / "test.csv", index=False)
    pd.DataFrame([
        {"Team": "Mexico", "Rank": 14, "Goals": 10},
        {"Team": "South Africa", "Rank": 60, "Goals": 5},
        {"Team": "Canada", "Rank": 35, "Goals": 7},
    ]).to_csv(training.KAGGLE_ROOT / "teams.csv", index=False)
    training.prepare_training_dataset(force=True)

    result = training.train_hybrid_model(
        fallback_tournament_2026(),
        payload={
            "seed": 7,
            "model_type": "ngboost",
            "model_id": "ngb-dual",
            "market_mode": "dual_markets",
            "tuning_enabled": True,
            "n_trials": 4,
            "n_estimators": 10,
        },
    )

    assert result["model"]["trained"] is True
    assert result["model"]["bundle"] is True
    assert result["model"]["markets"]["result"]["top_features"]
    assert result["model"]["markets"]["over_under_25"]["top_features"]


def test_mundial_lineup_payload_adds_visual_positions_and_photos():
    from src.web.mundial_services import enrich_lineup_payload

    payload = {
        "home": "Mexico",
        "away": "South Africa",
        "formation_home": "4-3-3",
        "formation_away": "4-4-2",
        "players": [
            {"team": "Mexico", "name": f"Mexico {index}", "id": index, "starter": True, "shirt_number": index, "position": "M"}
            for index in range(1, 12)
        ] + [
            {"team": "South Africa", "name": f"SA {index}", "id": 100 + index, "starter": True, "shirt_number": index, "position": "D"}
            for index in range(1, 12)
        ],
    }

    result = enrich_lineup_payload(payload)
    starters = [player for player in result["players"] if player["team"] == "Mexico" and player["starter"]]

    assert result["home_asset"]["flag_url"]
    assert len(starters) == 11
    assert all(player["photo_url"].startswith("https://api.sofascore.app/api/v1/player/") for player in starters)
    assert all(player["x"] != "" and player["y"] != "" for player in starters)
    assert starters[0]["initials"] == "M1"


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


def _fake_lanus_player(index: int, substitute: bool):
    return {
        "player": {"id": index, "name": f"Player {index}", "position": "M"},
        "shirtNumber": index,
        "position": "M",
        "substitute": substitute,
        "captain": index == 1,
        "statistics": {"rating": 7.1},
    }


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
