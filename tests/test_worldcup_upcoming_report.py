import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _patch_worldcup_results_file(monkeypatch, tmp_path):
    import src.worldcup.data as worldcup_data

    results_path = tmp_path / "worldcup_2026_results.csv"
    original_load = worldcup_data.load_worldcup_results_override

    def load_override(path=None):
        return original_load(path or results_path)

    monkeypatch.setattr(worldcup_data, "WORLD_CUP_2026_RESULTS_FILE", results_path)
    monkeypatch.setattr(worldcup_data, "load_worldcup_results_override", load_override)
    return worldcup_data, results_path


def _freeze_worldcup_now(monkeypatch, services, worldcup_data, current):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return current.astimezone(tz) if tz else current.replace(tzinfo=None)

    monkeypatch.setattr(worldcup_data, "datetime", FrozenDateTime)
    monkeypatch.setattr(services, "_now_utc", lambda: current)


def _fotmob_event(match_id, home, away, home_goals, away_goals, *, finished=True):
    status = {"finished": bool(finished), "reason": {"short": "FT" if finished else "NS"}}
    return {
        "id": str(match_id),
        "home": {"name": home, "score": home_goals},
        "away": {"name": away, "score": away_goals},
        "status": status,
        "finished": bool(finished),
    }


def test_score_history_for_tournament_prefers_all_matches_since_2014(monkeypatch):
    from src.web import mundial_services as services
    from src.worldcup.international_provider import normalize_international_matches

    raw_matches = pd.DataFrame([
        {"date": "2013-12-31", "home_team": "Mexico", "away_team": "Canada", "home_score": 5, "away_score": 0, "tournament": "Friendly", "neutral": False},
        {"date": "2014-01-02", "home_team": "Mexico", "away_team": "Canada", "home_score": 1, "away_score": 0, "tournament": "Friendly", "neutral": False},
        {"date": "2025-06-01", "home_team": "USA", "away_team": "Czech Republic", "home_score": 2, "away_score": 1, "tournament": "Friendly", "neutral": False},
    ])
    normalized = normalize_international_matches(raw_matches)
    monkeypatch.setattr(services, "load_international_matches", lambda required=False: normalized)
    monkeypatch.setattr(
        services,
        "load_historical_matches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("historical fallback should not be used")),
    )
    tournament = {"groups": {"A": ["Mexico", "Canada", "USA", "Czech Republic"]}}

    history, source = services.score_history_for_tournament(tournament, services.DEFAULT_CONFIG)

    assert source.startswith("all_matches.csv:")
    assert history["Date"].min() == pd.Timestamp("2014-01-02")
    assert list(history["Team 1"]) == ["Mexico", "USA"]
    assert list(history["Team 2"]) == ["Canada", "Czech Republic"]


def test_sota_and_alternative_sequences_are_statistical_score_models():
    from src.web import mundial_services as services
    from src.worldcup.score_models import score_model_options

    disabled = {"bayesian_hierarchical_poisson", "bayesian_dynamic_poisson"}
    catalog_keys = {option["key"] for option in score_model_options()}

    assert disabled.isdisjoint(services.SOTA_SCORE_MODEL_SEQUENCE)
    assert disabled <= catalog_keys
    assert "xg_poisson_local" not in services.SOTA_SCORE_MODEL_SEQUENCE
    assert len(services.SOTA_SCORE_MODEL_SEQUENCE) == 5
    assert services.ALTERNATIVE_SCORE_MODEL_SEQUENCE == [
        "statsmodels_poisson_glm",
        "negative_binomial_glm",
        "dixon_coles_mle",
        "bivariate_poisson_mle",
    ]
    assert services.BENCHMARK_SCORE_MODEL_SEQUENCE == services.SOTA_SCORE_MODEL_SEQUENCE
    assert "independent_poisson" not in services.ALTERNATIVE_SCORE_MODEL_SEQUENCE
    removed = {
        "diagonal_inflated_bivariate_poisson",
        "zero_inflated_generalized_poisson",
        "negative_binomial_mle",
        "conway_maxwell_poisson",
        "skellam_margin",
        "copula_weibull_count",
    }
    assert removed.isdisjoint(catalog_keys)
    assert removed.isdisjoint(services.SOTA_SCORE_MODEL_SEQUENCE)
    assert removed.isdisjoint(services.ALTERNATIVE_SCORE_MODEL_SEQUENCE)
    assert removed.isdisjoint(services.BENCHMARK_SCORE_MODEL_SEQUENCE)
    assert services.XG_LIGHTGBM_PIPELINE_MODE not in services.SOTA_SCORE_MODEL_SEQUENCE
    assert services.XG_LIGHTGBM_PIPELINE_MODE not in services.BENCHMARK_SCORE_MODEL_SEQUENCE


def test_alternatives_benchmark_aliases_and_statistical_registry():
    from src.web import mundial_services as services
    from src.worldcup.sota_alternatives import sota_alternatives_catalog

    assert services.normalize_report_pipeline_mode("benchmark_alternativas") == "alternatives_benchmark"
    assert services.normalize_report_pipeline_mode("sota_alternatives") == "alternatives_benchmark"
    assert services.normalize_report_pipeline_mode("modelos mejores") == "alternatives_benchmark"
    assert services.normalize_report_pipeline_mode("xg_lightgbm") == "xg_lightgbm"
    assert services.normalize_report_pipeline_mode("xg-lightgbm-cuda") == "xg_lightgbm"
    assert services.normalize_report_pipeline_mode("modelos_avanzados") == "advanced_models"
    assert services.normalize_report_pipeline_mode("todo documento") == "advanced_models"
    assert services.normalize_report_pipeline_mode("poisson_sota") == "poisson_sota"
    assert services.normalize_report_pipeline_mode("modo_desconocido") == "poisson_sota"
    assert services.SOTA_SCORE_MODEL_SEQUENCE == [
        "independent_poisson",
        "statsmodels_poisson_glm",
        "negative_binomial_glm",
        "dixon_coles_mle",
        "bivariate_poisson_mle",
    ]
    alternatives = sota_alternatives_catalog()
    assert [item["key"] for item in alternatives] == services.ALTERNATIVE_SCORE_MODEL_SEQUENCE
    forbidden = {"catboost", "xgboost", "lightgbm", "random forest", "mlp", "machine learning"}
    registry_text = json.dumps(alternatives).lower()
    assert not any(term in registry_text for term in forbidden)
    assert all(item["model_name"] and item["description"] for item in alternatives)


def test_advanced_models_pipeline_registry_and_config():
    from src.web import mundial_services as services

    config = services.report_pipeline_config({}, services.ADVANCED_MODELS_PIPELINE_MODE)
    assert config["bayes_profile"] == "light"
    assert config["backtest_scope"] == "worldcup_2026_confirmed_auto"
    assert services.active_advanced_score_model_sequence(config) == [
        "xg_dixon_coles",
        "negative_binomial_dixon_coles",
        "dynamic_strength_kalman",
        "stacked_meta_mnlogit",
    ]

    deep_config = services.report_pipeline_config(
        {"bayes_profile": "deep", "advanced_include_bayesian": True},
        services.ADVANCED_MODELS_PIPELINE_MODE,
    )
    ignored_deep_config = services.report_pipeline_config(
        {"bayes_profile": "deep"},
        services.ADVANCED_MODELS_PIPELINE_MODE,
    )
    assert "bayesian_dynamic_poisson" not in services.active_advanced_score_model_sequence(ignored_deep_config)
    assert "bayesian_dynamic_poisson" in services.active_advanced_score_model_sequence(deep_config)
    catalog = services.advanced_models_catalog({"families": [{"key": "xg_shot_quality", "status": "active"}]})
    assert {item["key"] for item in catalog} >= {
        "xg_dixon_coles",
        "negative_binomial_dixon_coles",
        "dynamic_strength_kalman",
        "stacked_meta_mnlogit",
        "bayesian_dynamic_poisson",
    }


def test_bayesian_fit_progress_emits_heartbeat_and_cuda_context(monkeypatch):
    from src.web import mundial_services as services

    class FakeModel:
        pass

    captured_config = {}

    def fake_build_score_model(base_model, history_df, teams, config):
        captured_config.update(config)
        time.sleep(0.04)
        return FakeModel()

    monkeypatch.setattr(services, "build_score_model", fake_build_score_model)
    progress = []

    model = services.build_score_model_with_fit_progress(
        FakeModel(),
        history_df=pd.DataFrame([{"Team 1": "A", "Team 2": "B", "G1": 1, "G2": 0}]),
        teams=["A", "B"],
        config={
            "score_backend": "cupy",
            "sota_device": "cuda",
            "bayes_draws": 2000,
            "bayes_tune": 2000,
            "bayes_chains": 4,
        },
        model_key="bayesian_dynamic_poisson",
        model_index=5,
        model_total=5,
        fixture_total=4,
        start_time=time.monotonic(),
        hardware={"score_backend": "cupy", "actual_device": "cuda", "requested_device": "cuda"},
        progress_callback=progress.append,
        heartbeat_interval=0.01,
    )

    assert isinstance(model, FakeModel)
    assert captured_config["score_model"] == "bayesian_dynamic_poisson"
    heartbeats = [item for item in progress if item.get("progress_mode") == "fit_heartbeat"]
    assert heartbeats
    assert any(item.get("pulse_index", 0) > 0 for item in heartbeats)
    assert any(item.get("last_state") == "muestreo NUTS activo" for item in heartbeats)
    assert heartbeats[-1]["last_state"] == "ajuste listo"
    assert heartbeats[-1]["score_backend"] == "cupy"
    assert heartbeats[-1]["actual_device"] == "cuda"
    assert heartbeats[-1]["bayes_draws"] == 2000
    assert "PyMC/NUTS tune 2000" in heartbeats[-1]["progress_detail"]


def test_report_warning_payload_normalizes_optional_limitations():
    from src.web import mundial_services as services

    payload = services.public_warning_payload([
        "Football-Data XLSX no disponible en cache: storage\\worldcup\\market\\WorldCup2026.xlsx.",
        "socceraction no instalado; xT/VAEP quedan como features opcionales.",
        "Potencia limitada: 20 partidos evaluados.",
    ], pipeline_mode=services.ADVANCED_MODELS_PIPELINE_MODE)

    assert payload["visible_warnings"][0] == "Fuentes avanzadas opcionales pendientes; el reporte usó fallback estadístico donde hizo falta."
    assert "Backtest con muestra limitada" in payload["visible_warnings"][1]
    assert all("\\" not in warning for warning in payload["technical_warnings"])
    assert "storage/worldcup/market/WorldCup2026.xlsx" in payload["technical_warnings"][0]


def test_advanced_source_preflight_prepares_data_and_uses_downloadable_sources(monkeypatch):
    from src.web import mundial_services as services

    calls = []

    def fake_market(**kwargs):
        calls.append(("market", kwargs))
        return {"status": "ok", "market_rows": 3, "qualifier_rows": 2, "sources": ["storage\\worldcup\\market\\WorldCup2026.xlsx"], "warnings": []}

    def fake_api(**kwargs):
        calls.append(("api", kwargs))
        return {
            "status": "ok",
            "fixtures": pd.DataFrame([{"FixtureId": 1}]),
            "team_stats": pd.DataFrame([{"Team": "Mexico"}]),
            "lineups": pd.DataFrame(),
            "injuries": pd.DataFrame(),
            "odds": pd.DataFrame(),
            "market_rows": pd.DataFrame(),
            "sources": ["storage\\worldcup\\api_football\\raw\\fixture.json"],
            "downloaded": [],
            "warnings": [],
        }

    def fake_prepare(payload, progress_callback=None):
        calls.append(("prepare", payload))
        return {"prepared": True, "prepared_rows": 4, "active_sources": ["manual_xg"], "warnings": [], "families": [], "models": []}

    monkeypatch.setattr(services, "load_market_data", fake_market)
    monkeypatch.setattr(services, "load_api_football_data", fake_api)
    monkeypatch.setattr(services, "advanced_data_prepare", fake_prepare)
    config = {}

    preflight = services.resolve_worldcup_sources_for_pipeline(
        {},
        config,
        services.ADVANCED_MODELS_PIPELINE_MODE,
        progress_callback=None,
    )

    assert [item[0] for item in calls] == ["market", "api", "prepare"]
    assert calls[0][1]["allow_download"] is True
    assert calls[1][1]["allow_download"] is True
    assert calls[2][1]["force"] is True
    assert config["_advanced_data_status"]["prepared_rows"] == 4
    assert preflight["sources"]["advanced_data"]["status"] == "prepared"
    assert preflight["sources"]["football_data"]["sources"][0] == "storage/worldcup/market/WorldCup2026.xlsx"


def test_predict_upcoming_report_runs_source_preflight_before_advanced_report(monkeypatch):
    from src.web import mundial_services as services

    calls = []

    monkeypatch.setattr(services, "stat_report_hardware", lambda *args, **kwargs: {"warnings": [], "score_backend": "numpy"})

    def fake_preflight(payload, config, pipeline_mode, progress_callback=None):
        calls.append(("preflight", pipeline_mode))
        config["_advanced_data_status"] = {"prepared_rows": 2, "active_sources": ["manual_xg"], "warnings": []}
        return {"status": "ok", "status_label": "Fuentes revisadas", "sources": {}, "visible_warnings": [], "technical_warnings": []}

    def fake_advanced_report(payload, config, start_time, hardware, progress_callback=None):
        calls.append(("advanced", config.get("_advanced_data_status", {}).get("prepared_rows")))
        return {"summary": {"pipeline_mode": services.ADVANCED_MODELS_PIPELINE_MODE, "source_preflight": config["_source_preflight"]}}

    monkeypatch.setattr(services, "resolve_worldcup_sources_for_pipeline", fake_preflight)
    monkeypatch.setattr(services, "advanced_models_report", fake_advanced_report)

    result = services.predict_upcoming_report({"pipeline_mode": "advanced_models"})

    assert calls == [("preflight", services.ADVANCED_MODELS_PIPELINE_MODE), ("advanced", 2)]
    assert result["summary"]["source_preflight"]["status"] == "ok"


def test_alternatives_benchmark_default_backtest_and_ranking_policy():
    from src.web import mundial_services as services

    config = services.report_pipeline_config({}, services.ALTERNATIVES_BENCHMARK_PIPELINE_MODE)
    assert config["backtest_last_n"] == 0
    assert config["backtest_mode"] == "auto_since_opening"
    assert config["backtest_start_date"] == "2026-06-11"
    assert config["backtest_cutoff_delay_minutes"] == 1
    assert config["backtest_scope"] == "worldcup_2026_confirmed_auto"
    assert config["sota_device"] == "cuda"

    ranked = services.rank_backtest_models([
        {
            "model_key": "a",
            "model_label": "A",
            "available": True,
            "evaluated_matches": 7,
            "log_loss": 1.0,
            "rps": 0.30,
            "expected_calibration_error": 0.10,
            "brier": 0.22,
            "pick_accuracy": 0.4,
            "top3_score_accuracy": 0.3,
            "over_under_accuracy": 0.6,
            "ou25_log_loss": 0.7,
            "score_accuracy": 0.1,
            "score_log_loss": 3.0,
        },
        {
            "model_key": "b",
            "model_label": "B",
            "available": True,
            "evaluated_matches": 7,
            "log_loss": 1.0,
            "rps": 0.20,
            "expected_calibration_error": 0.12,
            "brier": 0.30,
            "pick_accuracy": 0.3,
            "top3_score_accuracy": 0.2,
            "over_under_accuracy": 0.5,
            "ou25_log_loss": 0.8,
            "score_accuracy": 0.2,
            "score_log_loss": 4.0,
        },
        {
            "model_key": "c",
            "model_label": "C",
            "available": True,
            "evaluated_matches": 7,
            "log_loss": 0.9,
            "rps": 0.40,
            "expected_calibration_error": 0.20,
            "brier": 0.40,
            "pick_accuracy": 0.2,
            "top3_score_accuracy": 0.1,
            "over_under_accuracy": 0.4,
            "ou25_log_loss": 1.0,
            "score_accuracy": 0.0,
            "score_log_loss": 5.0,
        },
    ], {"holdout_start": "2022-12-09", "holdout_end": "2022-12-18"})

    assert [item["model_key"] for item in ranked] == ["a", "b", "c"]
    assert [item["rank"] for item in ranked] == [1, 2, 3]
    assert ranked[0]["score_resultados"] > ranked[1]["score_resultados"] > ranked[2]["score_resultados"]
    assert ranked[0]["reliability_score"] > ranked[1]["reliability_score"] > ranked[2]["reliability_score"]
    assert all(item["ranking_metric"] == "score_resultados" for item in ranked)
    assert "log-loss" in ranked[0]["ranking_reason"]
    assert all(item["holdout_start"] == "2022-12-09" for item in ranked)


def test_xg_lightgbm_report_is_separate_prediction_pipeline(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    class FakeModel:
        max_goals = 10

        def match_probabilities(self, home, away, max_goals=None):
            return {
                "home": 0.50,
                "draw": 0.25,
                "away": 0.25,
                "over05": 0.85,
                "under05": 0.15,
                "over15": 0.62,
                "under15": 0.38,
                "over25": 0.42,
                "under25": 0.58,
                "over35": 0.18,
                "under35": 0.82,
                "lambda1": 1.3,
                "lambda2": 0.9,
                "modal_g1": 1,
                "modal_g2": 0,
            }

    fixtures = pd.DataFrame([
        {"No.": 1, "Fecha": "2026-06-11", "Hora": "18:00 UTC+0", "Grupo": "Group A", "Equipo 1": "Mexico", "Equipo 2": "Canada", "Sede": "A"},
    ])
    model_meta = {
        "trained": True,
        "bundle": True,
        "model_id": "mundial-xg-lightgbm-hibrido",
        "model_name": "xG LightGBM test",
        "model_type": "lightgbm",
        "model_profile": "xg_lightgbm",
        "model_label": "xG-LightGBM",
        "market_mode": "dual_markets",
        "train_rows": 80,
        "validation_rows": 10,
        "test_rows": 10,
        "hardware": {"requested_device": "auto", "actual_device": "cuda", "cuda_available": True},
        "tuning": {"enabled": True, "sampler": "tpe", "best_value": 0.42},
        "warnings": [],
    }

    monkeypatch.setattr(services, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: ({}, "fixture-test"))
    monkeypatch.setattr(services, "ensure_worldcup_results_autorefreshed_once", lambda tournament: {"attempted": False})
    monkeypatch.setattr(services, "build_model", lambda tournament, config: (FakeModel(), "history-test"))
    monkeypatch.setattr(services, "apply_recent_context_model", lambda model, config: model)
    monkeypatch.setattr(services, "upcoming_fixture_rows", lambda tournament, group_filter="": fixtures)
    monkeypatch.setattr(services, "read_model_metadata", lambda model_id=None: model_meta)
    monkeypatch.setattr(services, "upcoming_sota_fixture_reports", lambda *args, **kwargs: pytest.fail("xg pipeline must not use SOTA reports"))
    monkeypatch.setattr(services, "score_history_for_tournament", lambda tournament, config: (pd.DataFrame([
        {"Date": "2022-01-01", "Year": 2022, "Team 1": "Mexico", "Team 2": "Canada", "G1": 2, "G2": 0, "Round": "Friendly", "Group": "Test"},
    ]), "history-test"))
    monkeypatch.setattr(services, "benchmark_feature_source", lambda tournament, history_df, config: type("FeatureSource", (), {"warnings": []})())

    def fake_backtest_model(key, label, score):
        return {
            "model_key": key,
            "model_label": label,
            "available": True,
            "evaluated_matches": 1,
            "rank": 1,
            "score_resultados": score,
            "reliability_score": score,
            "log_loss": 0.3,
            "brier": 0.1,
            "rps": 0.2,
            "expected_calibration_error": 0.04,
            "pick_accuracy": 1.0,
            "score_accuracy": 0.0,
            "top3_score_accuracy": 1.0,
            "over_under_accuracy": 1.0,
            "ou25_log_loss": 0.2,
            "matches": [],
            "vs_poisson": {"summary": "test"},
        }

    def fake_xg_backtest_report(**kwargs):
        return {
            "summary": {"available": True, "evaluated_matches": 1, "confirmed_matches": 1, "requested_matches": 5, "generated_at": "2026-06-18T00:00:00+00:00", "backtest_range": {"evaluated_matches": 1}},
            "models": [fake_backtest_model("xg_lightgbm", "Goles esperados (xG) + LightGBM", 82.0)],
            "warnings": [],
        }

    def fake_sota_backtest_report(**kwargs):
        return {
            "summary": {"available": True, "evaluated_matches": 1, "confirmed_matches": 1, "requested_matches": 5, "generated_at": "2026-06-18T00:00:00+00:00", "backtest_range": {"evaluated_matches": 1}},
            "models": [fake_backtest_model("independent_poisson", "Poisson", 70.0)],
            "warnings": [],
        }

    monkeypatch.setattr(services, "xg_lightgbm_backtest_report", fake_xg_backtest_report)
    monkeypatch.setattr(services, "alternatives_backtest_report", fake_sota_backtest_report)

    def fake_predict_match_payload(tournament, base_model, **kwargs):
        assert kwargs["use_ml_model"] is True
        assert kwargs["ml_weight"] == 1.0
        assert kwargs["model_id"] == "mundial-xg-lightgbm-hibrido"
        return {
            "fixture": {
                "id": "1",
                "date": "2026-06-11",
                "time": "18:00 UTC+0",
                "group": "Group A",
                "home": "Mexico",
                "away": "Canada",
                "venue": "A",
            },
            "probabilities": {
                "home": 61.0,
                "draw": 22.0,
                "away": 17.0,
                "over05": 91.0,
                "under05": 9.0,
                "over15": 68.0,
                "under15": 32.0,
                "over25": 48.0,
                "under25": 52.0,
                "over35": 21.0,
                "under35": 79.0,
            },
            "prediction": "1 Mexico",
            "expected_goals": {"home": 1.3, "away": 0.9},
            "modal_score": "1-0",
            "model_probs": {
                "ml": {"H": 61.0, "D": 22.0, "A": 17.0},
                "over_under_ml": {"over05": 91.0, "under05": 9.0},
                "result_weight": 1.0,
                "over_under_weight": 1.0,
                "model_id": "mundial-xg-lightgbm-hibrido",
                "model_name": "xG LightGBM test",
            },
            "market_readout": {},
            "contextual_poisson": {},
            "notes": ["Modelo xG aplicado."],
        }

    monkeypatch.setattr(services, "predict_match_payload", fake_predict_match_payload)

    progress = []
    result = services.predict_upcoming_report(
        {"pipeline_mode": "xg_lightgbm", "limit": 1},
        progress_callback=progress.append,
    )

    assert result["summary"]["pipeline_mode"] == "xg_lightgbm"
    assert result["summary"]["pipeline_label"] == "xG-LightGBM"
    assert result["summary"]["score_models"] == ["xg_lightgbm"]
    assert result["summary"]["backtest_auto_n"] == 1
    assert result["summary"]["xg_backtest"]["evaluated_matches"] == 1
    assert result["summary"]["sota_backtest"]["evaluated_matches"] == 1
    assert {item["model_key"] for item in result["model_backtests"]} == {"xg_lightgbm", "independent_poisson"}
    assert result["summary"]["model"]["model_profile"] == "xg_lightgbm"
    assert result["fixture_reports"][0]["decision"]["label"] == "1"
    assert "models" not in result["fixture_reports"][0]
    assert result["table"]["rows"][0]["Pipeline"] == "xG-LightGBM"
    assert result["table"]["rows"][0]["Peso ML 1X2"] == 1.0
    assert any(item.get("model_key") == "xg_lightgbm" for item in progress)
    assert (tmp_path / "latest.json").exists()


def test_alternatives_benchmark_report_returns_predictions_backtest_and_no_consensus(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    class FakeModel:
        max_goals = 10

        def __init__(self, key="independent_poisson"):
            self.key = key

        def match_probabilities(self, home, away, max_goals=None):
            return {
                "home": 0.56,
                "draw": 0.24,
                "away": 0.20,
                "over05": 0.9,
                "under05": 0.1,
                "over15": 0.7,
                "under15": 0.3,
                "over25": 0.45,
                "under25": 0.55,
                "over35": 0.2,
                "under35": 0.8,
                "lambda1": 1.4,
                "lambda2": 0.9,
                "modal_g1": 1,
                "modal_g2": 0,
            }

        def score_model_metadata(self):
            return {"key": self.key, "label": self.key, "available": True, "params": {}, "warnings": []}

    class FakeWorldCupModel:
        @classmethod
        def from_history(cls, historical_df, teams, **kwargs):
            return FakeModel()

    tournament = {
        "name": "World Cup 2026",
        "matches": [
            {"num": 1, "date": "2026-06-11", "time": "12:00 UTC+0", "team1": "Argentina", "team2": "France", "group": "Group A", "ground": "Test", "score": {"ft": [2, 1]}},
            {"num": 2, "date": "2026-06-12", "time": "12:00 UTC+0", "team1": "Brazil", "team2": "England", "group": "Group A", "ground": "Test", "score": {"ft": [1, 1]}},
            {"num": 3, "date": "2026-06-13", "time": "12:00 UTC+0", "team1": "France", "team2": "Brazil", "group": "Group A", "ground": "Test", "score": {"ft": [0, 1]}},
            {"num": 4, "date": "2026-06-20", "time": "18:00 UTC+0", "team1": "Argentina", "team2": "England", "group": "Group A", "ground": "Test"},
        ],
    }
    history = pd.DataFrame([
        {"Date": f"20{10 + index // 6:02d}-{(index % 12) + 1:02d}-01", "Year": 2010 + index // 6, "Team 1": home, "Team 2": away, "G1": g1, "G2": g2, "Round": "Group", "Group": "Test"}
        for index, (home, away, g1, g2) in enumerate([
            ("Argentina", "France", 2, 1),
            ("Brazil", "England", 1, 1),
            ("Argentina", "Brazil", 1, 0),
            ("France", "England", 0, 0),
            ("Brazil", "France", 2, 2),
            ("England", "Argentina", 0, 1),
            ("Argentina", "England", 3, 1),
            ("France", "Brazil", 1, 2),
            ("Brazil", "Argentina", 0, 0),
            ("England", "France", 2, 1),
            ("Argentina", "France", 1, 1),
            ("Brazil", "England", 2, 0),
        ])
    ])
    upcoming = pd.DataFrame([{
        "No.": 4,
        "Fecha": "2026-06-20",
        "Hora": "18:00 UTC+0",
        "Grupo": "Group A",
        "Equipo 1": "Argentina",
        "Equipo 2": "England",
        "Sede": "Test",
        "Finalizado": "No",
    }])

    monkeypatch.setattr(services, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(services, "BENCHMARK_SCORE_MODEL_SEQUENCE", ["independent_poisson", "dixon_coles_mle"])
    monkeypatch.setattr(services, "WorldCupModel", FakeWorldCupModel)
    monkeypatch.setattr(services, "build_score_model", lambda base_model, history_df, teams, config: FakeModel(config["score_model"]))
    refresh_calls = []

    def fake_refresh_worldcup_results(tournament, refresh=False):
        refresh_calls.append(refresh)
        return {
            "source": "test-results",
            "warnings": [],
            "refresh_attempted": bool(refresh),
            "fotmob_final_rows": 3,
            "refresh_added": 0,
            "confirmed_results": 3,
        }

    monkeypatch.setattr(services, "refresh_worldcup_2026_results", fake_refresh_worldcup_results)
    monkeypatch.setattr(services, "fixture_results_status", lambda fixture_df=None: {"source": "test-results"})
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: (tournament, "test:tournament"))
    monkeypatch.setattr(services, "load_historical_matches", lambda refresh=False: (history, "test:history"))
    monkeypatch.setattr(services, "upcoming_fixture_rows", lambda tournament, group_filter="": upcoming)
    monkeypatch.setattr(services, "contextual_poisson_for_match", lambda *args, **kwargs: {"available": False, "reason": "test"})
    monkeypatch.setattr(services, "detect_hardware", lambda: {
        "cpu_count": 2,
        "default_n_jobs": -1,
        "cuda_available": False,
        "cuda_devices": [],
        "cuda_device_names": [],
        "cuda_detection_source": "none",
        "cuda_detection_sources": [],
        "cuda_error": "sin dispositivos",
        "cuda_warning": "sin dispositivos",
        "device_default": "cpu",
    })

    progress = []
    result = services.predict_upcoming_report(
        {"pipeline_mode": "benchmark_alternativas", "limit": 1, "backtest_last_n": 5, "stat_model_cache": False},
        progress_callback=progress.append,
    )

    assert result["summary"]["pipeline_mode"] == "alternatives_benchmark"
    assert result["summary"]["pipeline_label"] == "Benchmark alternativas"
    assert result["summary"]["evidence_policy"] == "local_backtest_vs_poisson"
    assert result["summary"]["score_models"] == [item["model_key"] for item in result["model_backtests"]]
    assert set(result["summary"]["score_models"]) == set(services.BENCHMARK_SCORE_MODEL_SEQUENCE)
    assert result["summary"]["baseline_model"]["key"] == "independent_poisson"
    assert result["summary"]["backtest"]["evaluated_matches"] == 3
    assert result["summary"]["backtest_auto_n"] == 3
    assert result["summary"]["backtest_last_n"] == 3
    assert result["summary"]["backtest_mode"] == "auto_since_opening"
    assert result["summary"]["backtest_start_date"] == "2026-06-11"
    assert result["summary"]["backtest_scope"] == "worldcup_2026_confirmed_auto"
    assert result["summary"]["backtest_source"] == "test-results"
    assert result["summary"]["statistical_audit"]["available"] is True
    assert result["statistical_audit"]["baseline_model_key"] == "independent_poisson"
    assert result["statistical_audit"]["market_comparisons"]
    assert result["summary"]["generated_at"]
    assert result["summary"]["backtest_range"]["evaluated_matches"] == 3
    assert result["summary"]["backtest_range"]["first_match"]["home"] == "Argentina"
    assert result["summary"]["backtest_range"]["last_match"]["away"] == "Brazil"
    assert refresh_calls
    assert all(refresh_calls)
    assert result["summary"]["results_refresh"]["refresh_attempted"] is True
    assert result["summary"]["results_refresh"]["fotmob_final_rows"] == 3
    assert len(result["summary"]["backtest_confirmed_matches"]) == 3
    assert "posteriores" in result["summary"]["anti_leakage"]
    assert len(result["fixture_reports"]) == 1
    fixture_report = result["fixture_reports"][0]
    assert fixture_report["fixture"]["kickoff_iso"] == "2026-06-20T18:00:00+00:00"
    assert fixture_report["fixture"]["countdown_state"] == "ready"
    assert [model["model_key"] for model in fixture_report["models"]] == result["summary"]["score_models"]
    assert all("feature_context" in model for model in fixture_report["models"])
    assert fixture_report["baseline_poisson"]["model_key"] == "independent_poisson"
    assert "consensus" not in fixture_report
    assert "consensus_score_distribution" not in fixture_report
    assert "consensus_eligible" not in fixture_report["baseline_poisson"]
    assert all("consensus_eligible" not in model and "signature" not in model for model in fixture_report["models"])
    assert len(result["model_backtests"]) == len(services.BENCHMARK_SCORE_MODEL_SEQUENCE)
    assert [item["rank"] for item in result["model_backtests"]] == list(range(1, len(services.BENCHMARK_SCORE_MODEL_SEQUENCE) + 1))
    assert all(len(item["matches"]) == 3 for item in result["model_backtests"])
    assert all("over_under_accuracy" in item for item in result["model_backtests"])
    assert all("rps" in item for item in result["model_backtests"])
    assert all("expected_calibration_error" in item for item in result["model_backtests"])
    assert all("top3_score_accuracy" in item for item in result["model_backtests"])
    assert all("feature_usage_counts" in item for item in result["model_backtests"])
    assert all("ou25_log_loss" in item for item in result["model_backtests"])
    assert all("score_resultados" in item for item in result["model_backtests"])
    first_backtest_row = result["model_backtests"][0]["matches"][0]
    assert first_backtest_row["home"] == "Argentina"
    assert first_backtest_row["away"] == "France"
    assert first_backtest_row["home_asset"]["name"] == "Argentina"
    assert first_backtest_row["away_asset"]["name"] == "France"
    assert first_backtest_row["pick"] in {"1", "X", "2"}
    assert first_backtest_row["actual_pick"] in {"1", "X", "2"}
    assert isinstance(first_backtest_row["pick_hit"], bool)
    assert first_backtest_row["most_probable_score"] == first_backtest_row["modal_score"]
    assert "most_probable_score_probability" in first_backtest_row
    assert isinstance(first_backtest_row["most_probable_score_hit"], bool)
    assert isinstance(first_backtest_row["top3_score_hit"], bool)
    assert len(first_backtest_row["top_scores"]) == 5
    assert "rps" in first_backtest_row
    assert [item["line"] for item in first_backtest_row["over_under"]] == ["0.5", "1.5", "2.5", "3.5"]
    assert all(item["prediction_label"] in {"Over", "Under"} for item in first_backtest_row["over_under"])
    assert all(item["actual_label"] in {"Over", "Under"} for item in first_backtest_row["over_under"])
    assert all("log_loss" in item and "brier" in item for item in first_backtest_row["over_under"])
    assert result["best_model"]["model_key"] == result["model_backtests"][0]["model_key"]
    assert result["best_model"]["model_key"] in services.BENCHMARK_SCORE_MODEL_SEQUENCE
    assert "ensemble" not in result["fixture_reports"][0]
    assert result["fixture_reports"][0]["primary_model"]["available"] is True
    assert result["fixture_reports"][0]["primary_model"]["model_key"] == result["best_model"]["model_key"]
    assert result["table"]["total"] == len(result["fixture_reports"])
    assert result["summary"]["feature_research"]["families"]
    assert result["downloads"]["predictions_html"].endswith("kind=predictions&format=html")
    assert result["downloads"]["predictions_csv"].endswith("kind=predictions&format=csv")
    assert result["downloads"]["backtest_html"].endswith("kind=backtest&format=html")
    assert result["downloads"]["backtest_csv"].endswith("kind=backtest&format=csv")
    assert (tmp_path / f"{result['report_id']}_predictions.html").exists()
    assert (tmp_path / f"{result['report_id']}_predictions.csv").exists()
    assert (tmp_path / f"{result['report_id']}_backtest.html").exists()
    assert "Marcador #1" in (tmp_path / f"{result['report_id']}_backtest.html").read_text(encoding="utf-8")
    assert (tmp_path / f"{result['report_id']}_backtest.csv").exists()
    assert (tmp_path / "latest.json").exists()
    assert progress[-1]["stage"] == "complete"


def test_benchmark_optuna_tunes_poisson_recent_matches(monkeypatch):
    from src.web import mundial_services as services

    class FakeState:
        name = "COMPLETE"

    class FakeTrial:
        def __init__(self, number, recent_n):
            self.number = number
            self.recent_n = recent_n
            self.params = {}
            self.user_attrs = {}
            self.value = None
            self.state = FakeState()

        def suggest_int(self, name, low, high):
            assert name == "poisson_recent_matches"
            assert low == 3
            assert high == 50
            self.params[name] = self.recent_n
            return self.recent_n

        def set_user_attr(self, name, value):
            self.user_attrs[name] = value

    class FakeStudy:
        def __init__(self):
            self.best_trial = None
            self.best_value = None

        def optimize(self, objective, n_trials, callbacks, show_progress_bar=False):
            for number, recent_n in enumerate([5, 12, 30][:n_trials]):
                trial = FakeTrial(number, recent_n)
                trial.value = objective(trial)
                if self.best_value is None or trial.value > self.best_value:
                    self.best_value = trial.value
                    self.best_trial = trial
                for callback in callbacks:
                    callback(self, trial)

    class FakeSampler:
        def __init__(self, seed=None):
            self.seed = seed

    class FakeOptuna:
        class samplers:
            TPESampler = FakeSampler
            RandomSampler = FakeSampler

        @staticmethod
        def create_study(direction, sampler):
            assert direction == "maximize"
            return FakeStudy()

    history = pd.DataFrame([
        {"Date": "2022-11-20", "Year": 2022, "Team 1": "A", "Team 2": "B", "G1": 1, "G2": 0, "Round": "Group", "Group": "A"},
    ])
    confirmed = pd.DataFrame([
        {"No.": 1, "Date": "2026-06-11", "Year": 2026, "Team 1": "A", "Team 2": "B", "G1": 2, "G2": 0, "Round": "Group", "Group": "A", "Source": "test"},
    ])

    def fake_evaluate(**kwargs):
        recent_n = int(kwargs["config"]["poisson_recent_matches"])
        return {"available": True, "score_resultados": 100 - abs(recent_n - 12), "log_loss": abs(recent_n - 12) + 0.25}

    monkeypatch.setitem(sys.modules, "optuna", FakeOptuna)
    monkeypatch.setattr(services, "confirmed_worldcup_2026_backtest_rows", lambda tournament: confirmed)
    monkeypatch.setattr(services, "evaluate_score_model_walk_forward_2026", fake_evaluate)

    summary = services.tune_benchmark_poisson_recent_matches(
        history_df=history,
        tournament={"matches": []},
        config={
            **services.report_pipeline_config(
                {"benchmark_tuning_enabled": True, "benchmark_tuning_trials": 3},
                services.ALTERNATIVES_BENCHMARK_PIPELINE_MODE,
            ),
            "seed": 2026,
        },
        model_sequence=["independent_poisson", "dixon_coles_mle"],
        start_time=0.0,
        hardware={},
    )

    assert summary["enabled"] is True
    assert summary["available"] is True
    assert summary["scope"] == "all_active_models"
    assert summary["model_sequence"] == ["independent_poisson", "dixon_coles_mle"]
    assert summary["best_poisson_recent_matches"] == 12
    assert summary["objective"] == "mean_score_resultados"
    assert summary["best_value"] == 100
    assert [trial["poisson_recent_matches"] for trial in summary["trials"]] == [5, 12, 30]
    assert all(trial["available_models"] == 2 for trial in summary["trials"])


def test_alternatives_benchmark_report_applies_tuned_recent_matches(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    class FakeModel:
        max_goals = 10

        def match_probabilities(self, home, away, max_goals=None):
            return {
                "home": 0.5,
                "draw": 0.25,
                "away": 0.25,
                "over05": 0.8,
                "under05": 0.2,
                "over15": 0.6,
                "under15": 0.4,
                "over25": 0.4,
                "under25": 0.6,
                "over35": 0.2,
                "under35": 0.8,
                "lambda1": 1.2,
                "lambda2": 0.8,
                "modal_g1": 1,
                "modal_g2": 0,
            }

        def score_model_metadata(self):
            return {"key": "independent_poisson", "label": "Poisson", "available": True, "params": {}, "warnings": []}

    class FakeWorldCupModel:
        @classmethod
        def from_history(cls, historical_df, teams, **kwargs):
            return FakeModel()

    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-20", "time": "18:00 UTC+0", "team1": "A", "team2": "B", "group": "Group A"},
        ],
    }
    history = pd.DataFrame([
        {"Date": "2022-11-20", "Year": 2022, "Team 1": "A", "Team 2": "B", "G1": 1, "G2": 0, "Round": "Group", "Group": "A"},
    ])
    upcoming = pd.DataFrame([{
        "No.": 1,
        "Fecha": "2026-06-20",
        "Hora": "18:00 UTC+0",
        "Grupo": "Group A",
        "Equipo 1": "A",
        "Equipo 2": "B",
        "Sede": "Test",
        "Finalizado": "No",
    }])
    captured = {}

    def fake_fixture_reports(**kwargs):
        captured["fixture_poisson_recent_matches"] = kwargs["config"]["poisson_recent_matches"]
        return [{
            "fixture": {"id": "1", "date": "2026-06-20", "group": "Group A", "label": "A vs B", "home": "A", "away": "B"},
            "models": [{
                "model_key": "independent_poisson",
                "model_label": "Poisson",
                "available": True,
                "decision": {"outcome": "home", "label": "1", "team": "A"},
                "probabilities": {"home": 50, "draw": 25, "away": 25},
                "expected_goals": {"home": 1.2, "away": 0.8},
                "top_score": "1-0",
                "top_scores": [],
                "warnings": [],
            }],
        }]

    def fake_backtest_report(**kwargs):
        captured["backtest_poisson_recent_matches"] = kwargs["config"]["poisson_recent_matches"]
        return {
            "summary": {
                "available": True,
                "scope": "worldcup_2026_confirmed_auto",
                "source": "test-results",
                "generated_at": "2026-06-14T12:00:00+00:00",
                "backtest_range": {
                    "evaluated_matches": 1,
                    "first_match": {"home": "A", "away": "B"},
                    "last_match": {"home": "A", "away": "B"},
                    "first_date": "2026-06-11",
                    "last_date": "2026-06-11",
                    "generated_at": "2026-06-14T12:00:00+00:00",
                },
                "confirmed_matches": 1,
                "confirmed_matches_detail": [],
                "evaluated_matches": 1,
                "train_matches": 1,
                "anti_leakage": "posteriores",
            },
            "models": [{
                "model_key": "independent_poisson",
                "model_label": "Poisson",
                "available": True,
                "rank": 1,
                "reliability_score": 100,
                "score_resultados": 100,
                "ranking_metric": "score_resultados",
                "evaluated_matches": 1,
                "log_loss": 0.25,
                "brier": 0.1,
                "score_log_loss": 1.0,
                "pick_accuracy": 1.0,
                "score_accuracy": 0.0,
                "over_under_accuracy": 1.0,
                "matches": [],
                "vs_poisson": {"summary": "baseline"},
            }],
            "warnings": [],
        }

    monkeypatch.setattr(services, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(services, "BENCHMARK_SCORE_MODEL_SEQUENCE", ["independent_poisson"])
    monkeypatch.setattr(services, "WorldCupModel", FakeWorldCupModel)
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: (tournament, "test:tournament"))
    monkeypatch.setattr(services, "refresh_worldcup_2026_results", lambda tournament, refresh=False: {"source": "test-results", "warnings": [], "refresh_attempted": True, "confirmed_results": 1, "conflicts": []})
    monkeypatch.setattr(services, "load_historical_matches", lambda refresh=False: (history, "test:history"))
    monkeypatch.setattr(services, "upcoming_fixture_rows", lambda tournament, group_filter="": upcoming)
    monkeypatch.setattr(services, "upcoming_sota_fixture_reports", fake_fixture_reports)
    monkeypatch.setattr(services, "alternatives_backtest_report", fake_backtest_report)
    monkeypatch.setattr(services, "poisson_baseline_report_for_fixture", lambda base_model, fixture, config: {"model_key": "independent_poisson"})
    monkeypatch.setattr(services, "tune_benchmark_poisson_recent_matches", lambda **kwargs: {
        "enabled": True,
        "available": True,
        "best_poisson_recent_matches": 7,
        "best_value": 88.2,
        "n_trials": 3,
        "sampler": "tpe",
        "objective": "mean_score_resultados",
        "trials": [],
        "warnings": [],
    })

    result = services.alternatives_benchmark_report(
        payload={"limit": 1},
        config=services.report_pipeline_config({"benchmark_tuning_enabled": True}, services.ALTERNATIVES_BENCHMARK_PIPELINE_MODE),
        start_time=0.0,
        hardware={"warnings": []},
    )

    assert captured["fixture_poisson_recent_matches"] == 7
    assert captured["backtest_poisson_recent_matches"] == 7
    assert result["summary"]["poisson_recent_matches"] == 7
    assert result["summary"]["benchmark_tuning"]["best_poisson_recent_matches"] == 7


def test_alternatives_benchmark_refreshes_results_automatically_and_uses_all_finals(tmp_path, monkeypatch):
    from src.web import mundial_services as services
    import src.worldcup.fotmob_provider as fotmob_provider

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 23, 0, tzinfo=timezone.utc))

    pd.DataFrame([
        {"date": "2026-06-11", "home": "Argentina", "away": "France", "home_goals": 2, "away_goals": 1, "status": "final", "source": "manual", "updated_at": "2026-06-11T23:00:00+00:00"},
        {"date": "2026-06-12", "home": "Brazil", "away": "England", "home_goals": 1, "away_goals": 1, "status": "final", "source": "manual", "updated_at": "2026-06-12T23:00:00+00:00"},
    ], columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)

    class FakeModel:
        max_goals = 10

        def match_probabilities(self, home, away, max_goals=None):
            return {
                "home": 0.56,
                "draw": 0.24,
                "away": 0.20,
                "over05": 0.9,
                "under05": 0.1,
                "over15": 0.7,
                "under15": 0.3,
                "over25": 0.45,
                "under25": 0.55,
                "over35": 0.2,
                "under35": 0.8,
                "lambda1": 1.4,
                "lambda2": 0.9,
                "modal_g1": 1,
                "modal_g2": 0,
            }

        def score_model_metadata(self):
            return {"key": "independent_poisson", "label": "Poisson", "available": True, "params": {}, "warnings": []}

    class FakeWorldCupModel:
        @classmethod
        def from_history(cls, historical_df, teams, **kwargs):
            return FakeModel()

    tournament = {
        "name": "World Cup 2026",
        "matches": [
            {"num": 1, "date": "2026-06-11", "time": "12:00 UTC+0", "team1": "Argentina", "team2": "France", "group": "Group A", "ground": "Test"},
            {"num": 2, "date": "2026-06-12", "time": "12:00 UTC+0", "team1": "Brazil", "team2": "England", "group": "Group A", "ground": "Test"},
            {"num": 3, "date": "2026-06-13", "time": "12:00 UTC+0", "team1": "France", "team2": "Brazil", "group": "Group A", "ground": "Test"},
            {"num": 4, "date": "2026-06-14", "time": "12:00 UTC+0", "team1": "England", "team2": "Argentina", "group": "Group A", "ground": "Test"},
            {"num": 5, "date": "2026-06-20", "time": "18:00 UTC+0", "team1": "Argentina", "team2": "England", "group": "Group A", "ground": "Test"},
        ],
    }
    history = pd.DataFrame([
        {"Date": f"20{10 + index // 6:02d}-{(index % 12) + 1:02d}-01", "Year": 2010 + index // 6, "Team 1": home, "Team 2": away, "G1": g1, "G2": g2, "Round": "Group", "Group": "Test"}
        for index, (home, away, g1, g2) in enumerate([
            ("Argentina", "France", 2, 1),
            ("Brazil", "England", 1, 1),
            ("Argentina", "Brazil", 1, 0),
            ("France", "England", 0, 0),
            ("Brazil", "France", 2, 2),
            ("England", "Argentina", 0, 1),
        ])
    ])
    fotmob_payloads = {
        "20260611": [_fotmob_event(101, "Argentina", "France", 2, 1)],
        "20260612": [_fotmob_event(102, "Brazil", "England", 1, 1)],
        "20260613": [_fotmob_event(103, "France", "Brazil", 0, 1)],
        "20260614": [_fotmob_event(104, "England", "Argentina", 0, 2)],
    }

    def fake_fotmob_get_json(url, params=None):
        return {"matches": fotmob_payloads.get((params or {}).get("date"), [])}

    monkeypatch.setattr(fotmob_provider, "fotmob_get_json", fake_fotmob_get_json)
    monkeypatch.setattr(services, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(services, "BENCHMARK_SCORE_MODEL_SEQUENCE", ["independent_poisson"])
    monkeypatch.setattr(services, "WorldCupModel", FakeWorldCupModel)
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: (tournament, "test:tournament"))
    monkeypatch.setattr(services, "load_historical_matches", lambda refresh=False: (history, "test:history"))
    monkeypatch.setattr(services, "contextual_poisson_for_match", lambda *args, **kwargs: {"available": False, "reason": "test"})
    monkeypatch.setattr(services, "detect_hardware", lambda: {
        "cpu_count": 2,
        "default_n_jobs": -1,
        "cuda_available": False,
        "cuda_devices": [],
        "cuda_device_names": [],
        "cuda_detection_source": "none",
        "cuda_detection_sources": [],
        "cuda_error": "",
        "cuda_warning": "",
        "device_default": "cpu",
        "warnings": [],
    })

    result = services.predict_upcoming_report({
        "pipeline_mode": "benchmark_alternativas",
        "limit": 1,
        "refresh": False,
        "stat_model_cache": False,
    })

    refresh = result["summary"]["results_refresh"]
    assert refresh["refresh_attempted"] is True
    assert refresh["fotmob_final_rows"] == 4
    assert refresh["confirmed_results"] == 4
    assert result["summary"]["backtest_auto_n"] == 4
    assert len(result["summary"]["backtest_confirmed_matches"]) == 4
    assert result["summary"]["backtest_confirmed_matches"][-1]["home"] == "England"
    assert result["model_backtests"][0]["evaluated_matches"] == 4
    assert len(result["model_backtests"][0]["matches"]) == 4
    assert pd.read_csv(results_path).shape[0] == 4


def test_backtest_auto_n_crece_al_avanzar_fecha_sin_reinicio(tmp_path, monkeypatch):
    from src.worldcup import data as worldcup_data
    from src.web import mundial_services as services

    class FakeModel:
        max_goals = 10

        def match_probabilities(self, home, away, max_goals=None):
            return {
                "home": 0.56,
                "draw": 0.24,
                "away": 0.20,
                "over05": 0.9,
                "under05": 0.1,
                "over15": 0.7,
                "under15": 0.3,
                "over25": 0.45,
                "under25": 0.55,
                "over35": 0.2,
                "under35": 0.8,
                "lambda1": 1.4,
                "lambda2": 0.9,
            }

        def score_model_metadata(self):
            return {"key": "independent_poisson", "label": "Poisson", "available": True, "params": {}, "warnings": []}

    class FakeWorldCupModel:
        @classmethod
        def from_history(cls, historical_df, teams, **kwargs):
            return FakeModel()

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 13, 0, tzinfo=timezone.utc))

    tournament = {
        "name": "World Cup 2026",
        "matches": [
            {"num": 1, "date": "2026-06-14", "time": "12:00 UTC+0", "team1": "England", "team2": "Argentina", "group": "Group A", "ground": "Test"},
            {"num": 2, "date": "2026-06-16", "time": "12:00 UTC+0", "team1": "France", "team2": "Brazil", "group": "Group A", "ground": "Test"},
            {"num": 3, "date": "2026-06-20", "time": "18:00 UTC+0", "team1": "Mexico", "team2": "USA", "group": "Group A", "ground": "Test"},
        ],
    }
    history = pd.DataFrame([
        {"Date": f"20{14 + index // 6:02d}-{(index % 12) + 1:02d}-01", "Year": 2014 + index // 6, "Team 1": home, "Team 2": away, "G1": g1, "G2": g2, "Round": "Group", "Group": "Test"}
        for index, (home, away, g1, g2) in enumerate([
            ("England", "Argentina", 2, 1),
            ("France", "Brazil", 1, 1),
            ("Mexico", "USA", 3, 0),
        ])
    ])
    upcoming = pd.DataFrame([{
        "No.": 3,
        "Fecha": "2026-06-20",
        "Hora": "18:00 UTC+0",
        "Grupo": "Group A",
        "Equipo 1": "Mexico",
        "Equipo 2": "USA",
        "Sede": "Test",
        "Finalizado": "No",
    }])

    pd.DataFrame([
        {"date": "2026-06-14", "home": "England", "away": "Argentina", "home_goals": 2, "away_goals": 1, "status": "final", "source": "manual", "updated_at": "2026-06-14T13:00:00+00:00"},
    ], columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)

    def fake_refresh_worldcup_2026_results(refresh_tournament, refresh=False):
        confirmed = int(len(pd.read_csv(results_path)))
        return {
            "source": "test-results",
            "provider": "test-provider",
            "refresh_attempted": bool(refresh),
            "refresh_added": confirmed,
            "refresh_updated": 0,
            "fotmob_final_rows": 0,
            "sofascore_final_rows": 0,
            "verified_final_rows": 0,
            "conflicts": [],
            "warnings": [],
            "provider_warnings": [],
            "missing_result_fixtures": [],
            "confirmed_results": confirmed,
        }

    def fake_upcoming_sota_fixture_reports(
        tournament,
        base_model,
        fixtures,
        config,
        start_time,
        hardware,
        model_sequence=None,
        history_df=None,
        feature_source=None,
        progress_callback=None,
    ):
        probabilities = {
            "home": 56,
            "draw": 24,
            "away": 20,
            "over05": 90,
            "under05": 10,
            "over15": 70,
            "under15": 30,
            "over25": 45,
            "under25": 55,
            "over35": 20,
            "under35": 80,
        }
        reports = []
        for fixture in fixtures:
            reports.append({
                "fixture": services.report_fixture_payload({
                    "id": str(fixture.get("No.", "")),
                    "date": fixture.get("Fecha", ""),
                    "time": fixture.get("Hora", ""),
                    "group": fixture.get("Grupo", ""),
                    "home": str(fixture.get("Equipo 1", "")),
                    "away": str(fixture.get("Equipo 2", "")),
                    "venue": fixture.get("Sede", ""),
                }),
                "contextual_poisson": {"available": False, "reason": "test"},
                "models": [
                    {
                        "model_key": "independent_poisson",
                        "model_label": "Poisson",
                        "available": True,
                        "decision": {"outcome": "home", "label": "1", "team": str(fixture.get("Equipo 1", ""))},
                        "probabilities": probabilities,
                        "expected_goals": {"home": 1.4, "away": 0.9},
                        "top_score": "1-0",
                        "top_scores": [
                            {"score": "1-0", "probability": 55.0},
                            {"score": "0-0", "probability": 20.0},
                        ],
                    },
                ],
                "warnings": [],
            })
        return reports

    def fake_poisson_baseline_report_for_fixture(base_model, fixture, cfg, feature_source=None, history_df=None):
        return {
            "model_key": "independent_poisson",
            "model_label": "Poisson",
            "available": True,
            "probabilities": {
                "home": 56,
                "draw": 24,
                "away": 20,
                "over05": 90,
                "under05": 10,
                "over15": 70,
                "under15": 30,
                "over25": 45,
                "under25": 55,
                "over35": 20,
                "under35": 80,
            },
            "top_scores": [{"score": "1-0", "probability": 55.0}],
            "top_score": "1-0",
            "top_score_probability": 55.0,
            "source": "Poisson baseline",
        }

    monkeypatch.setattr(services, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(services, "BENCHMARK_SCORE_MODEL_SEQUENCE", ["independent_poisson"])
    monkeypatch.setattr(services, "WorldCupModel", FakeWorldCupModel)
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: (tournament, "test:tournament"))
    monkeypatch.setattr(services, "load_historical_matches", lambda refresh=False: (history, "test:history"))
    monkeypatch.setattr(services, "upcoming_fixture_rows", lambda tournament, group_filter="": upcoming)
    monkeypatch.setattr(services, "refresh_worldcup_2026_results", fake_refresh_worldcup_2026_results)
    monkeypatch.setattr(services, "upcoming_sota_fixture_reports", fake_upcoming_sota_fixture_reports)
    monkeypatch.setattr(services, "poisson_baseline_report_for_fixture", fake_poisson_baseline_report_for_fixture)
    monkeypatch.setattr(services, "load_international_matches", lambda required=False: pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score"]))
    monkeypatch.setattr(services, "_WORLD_CUP_RESULTS_AUTO_REFRESHED", False)
    monkeypatch.setattr(services, "_WORLD_CUP_RESULTS_AUTO_REFRESH_SUMMARY", {})
    monkeypatch.setattr(services, "_WORLD_CUP_RESULTS_AUTO_REFRESH_EXPIRES_AT", None)

    config = services.report_pipeline_config({}, services.ALTERNATIVES_BENCHMARK_PIPELINE_MODE)
    result_dia14 = services.alternatives_benchmark_report(
        payload={"limit": 1},
        config=config,
        start_time=0.0,
        hardware={"warnings": []},
    )
    confirmed_dia14 = services.confirmed_worldcup_2026_backtest_rows(tournament)

    assert result_dia14["summary"]["backtest_auto_n"] == 1
    assert len(confirmed_dia14) == 1
    assert confirmed_dia14.iloc[0]["Team 1"] == "England"

    pd.concat([
        pd.read_csv(results_path),
        pd.DataFrame([
            {"date": "2026-06-16", "home": "France", "away": "Brazil", "home_goals": 3, "away_goals": 2, "status": "final", "source": "manual", "updated_at": "2026-06-16T13:00:00+00:00"},
        ]),
    ], ignore_index=True).to_csv(results_path, index=False)

    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 16, 13, 0, tzinfo=timezone.utc))
    result_dia16 = services.alternatives_benchmark_report(
        payload={"limit": 1},
        config=config,
        start_time=0.0,
        hardware={"warnings": []},
    )
    confirmed_dia16 = services.confirmed_worldcup_2026_backtest_rows(tournament)

    assert result_dia16["summary"]["backtest_auto_n"] == 2
    assert len(confirmed_dia16) == 2
    assert confirmed_dia16.iloc[-1]["Team 1"] == "France"


def test_backtest_alternatives_refresca_fixtures_sin_reinicio_si_cambia_dia(tmp_path, monkeypatch):
    from src.worldcup import data as worldcup_data
    from src.web import mundial_services as services

    class FakeModel:
        max_goals = 10

        def match_probabilities(self, home, away, max_goals=None):
            return {
                "home": 0.56,
                "draw": 0.24,
                "away": 0.20,
                "over05": 0.9,
                "under05": 0.1,
                "over15": 0.7,
                "under15": 0.3,
                "over25": 0.45,
                "under25": 0.55,
                "over35": 0.2,
                "under35": 0.8,
                "lambda1": 1.4,
                "lambda2": 0.9,
            }

        def score_model_metadata(self):
            return {"key": "independent_poisson", "label": "Poisson", "available": True, "params": {}, "warnings": []}

    class FakeWorldCupModel:
        @classmethod
        def from_history(cls, historical_df, teams, **kwargs):
            return FakeModel()

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 13, 0, tzinfo=timezone.utc))
    load_calls: list[bool] = []

    def fake_load_tournament(refresh=False):
        load_calls.append(bool(refresh))
        if refresh:
            return {
                "name": "World Cup 2026",
                "matches": [
                    {"num": 1, "date": "2026-06-14", "time": "12:00 UTC+0", "team1": "England", "team2": "Argentina", "group": "Group A", "ground": "Test"},
                    {"num": 2, "date": "2026-06-16", "time": "12:00 UTC+0", "team1": "France", "team2": "Brazil", "group": "Group A", "ground": "Test"},
                    {"num": 3, "date": "2026-06-20", "time": "18:00 UTC+0", "team1": "Mexico", "team2": "USA", "group": "Group A", "ground": "Test"},
                ],
            }, "test:tournament"
        return {
            "name": "World Cup 2026",
            "matches": [
                {"num": 1, "date": "2026-06-14", "time": "12:00 UTC+0", "team1": "England", "team2": "Argentina", "group": "Group A", "ground": "Test"},
            ],
        }, "test:tournament"

    history = pd.DataFrame([
        {"Date": "2014-01-01", "Year": 2014, "Team 1": "England", "Team 2": "Argentina", "G1": 2, "G2": 1, "Round": "Group", "Group": "Test"},
    ])

    upcoming = pd.DataFrame([{
        "No.": 3,
        "Fecha": "2026-06-20",
        "Hora": "18:00 UTC+0",
        "Grupo": "Group A",
        "Equipo 1": "Mexico",
        "Equipo 2": "USA",
        "Sede": "Test",
        "Finalizado": "No",
    }])

    pd.DataFrame([
        {"date": "2026-06-14", "home": "England", "away": "Argentina", "home_goals": 2, "away_goals": 1, "status": "final", "source": "manual", "updated_at": "2026-06-14T10:00:00+00:00"},
        {"date": "2026-06-16", "home": "France", "away": "Brazil", "home_goals": 3, "away_goals": 2, "status": "final", "source": "manual", "updated_at": "2026-06-16T10:00:00+00:00"},
    ], columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)

    def fake_refresh_worldcup_2026_results(_, refresh=False):
        confirmed = int(len(pd.read_csv(results_path)))
        return {
            "source": "test-results",
            "provider": "test-provider",
            "refresh_attempted": bool(refresh),
            "refresh_added": 0,
            "refresh_updated": 0,
            "fotmob_final_rows": 0,
            "sofascore_final_rows": 0,
            "verified_final_rows": 0,
            "conflicts": [],
            "warnings": [],
            "provider_warnings": [],
            "missing_result_fixtures": [],
            "confirmed_results": confirmed,
        }

    def fake_upcoming_sota_fixture_reports(
        tournament,
        base_model,
        fixtures,
        config,
        start_time,
        hardware,
        model_sequence=None,
        history_df=None,
        feature_source=None,
        progress_callback=None,
    ):
        probabilities = {
            "home": 56,
            "draw": 24,
            "away": 20,
            "over05": 90,
            "under05": 10,
            "over15": 70,
            "under15": 30,
            "over25": 45,
            "under25": 55,
            "over35": 20,
            "under35": 80,
        }
        return [{
            "fixture": services.report_fixture_payload({
                "id": str(upcoming.iloc[0].get("No.", "")),
                "date": upcoming.iloc[0].get("Fecha", ""),
                "time": upcoming.iloc[0].get("Hora", ""),
                "group": upcoming.iloc[0].get("Grupo", ""),
                "home": str(upcoming.iloc[0].get("Equipo 1", "")),
                "away": str(upcoming.iloc[0].get("Equipo 2", "")),
                "venue": upcoming.iloc[0].get("Sede", ""),
            }),
            "contextual_poisson": {"available": False, "reason": "test"},
            "models": [
                {
                    "model_key": "independent_poisson",
                    "model_label": "Poisson",
                    "available": True,
                    "decision": {"outcome": "home", "label": "1", "team": str(upcoming.iloc[0].get("Equipo 1", ""))},
                    "probabilities": probabilities,
                    "expected_goals": {"home": 1.4, "away": 0.9},
                    "top_score": "1-0",
                    "top_scores": [{"score": "1-0", "probability": 55.0}],
                },
            ],
            "warnings": [],
        }]

    def fake_poisson_baseline_report_for_fixture(base_model, fixture, cfg, feature_source=None, history_df=None):
        return {
            "model_key": "independent_poisson",
            "model_label": "Poisson",
            "available": True,
            "probabilities": {
                "home": 56,
                "draw": 24,
                "away": 20,
                "over05": 90,
                "under05": 10,
                "over15": 70,
                "under15": 30,
                "over25": 45,
                "under25": 55,
                "over35": 20,
                "under35": 80,
            },
            "top_scores": [{"score": "1-0", "probability": 55.0}],
            "top_score": "1-0",
            "top_score_probability": 55.0,
            "source": "Poisson baseline",
        }

    monkeypatch.setattr(services, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(services, "BENCHMARK_SCORE_MODEL_SEQUENCE", ["independent_poisson"])
    monkeypatch.setattr(services, "WorldCupModel", FakeWorldCupModel)
    monkeypatch.setattr(services, "load_tournament_2026", fake_load_tournament)
    monkeypatch.setattr(services, "load_historical_matches", lambda refresh=False: (history, "test:history"))
    monkeypatch.setattr(services, "upcoming_fixture_rows", lambda tournament, group_filter="": upcoming)
    monkeypatch.setattr(services, "refresh_worldcup_2026_results", fake_refresh_worldcup_2026_results)
    monkeypatch.setattr(services, "upcoming_sota_fixture_reports", fake_upcoming_sota_fixture_reports)
    monkeypatch.setattr(services, "poisson_baseline_report_for_fixture", fake_poisson_baseline_report_for_fixture)
    monkeypatch.setattr(services, "load_international_matches", lambda required=False: pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score"]))
    monkeypatch.setattr(services, "_WORLD_CUP_FIXTURES_AUTO_REFRESH_EXPIRES_AT", None)

    config = services.report_pipeline_config({}, services.ALTERNATIVES_BENCHMARK_PIPELINE_MODE)
    result_dia14 = services.alternatives_benchmark_report(
        payload={"limit": 1},
        config=config,
        start_time=0.0,
        hardware={"warnings": []},
    )

    assert result_dia14["summary"]["backtest_auto_n"] == 1

    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 16, 13, 0, tzinfo=timezone.utc))
    result_dia16 = services.alternatives_benchmark_report(
        payload={"limit": 1},
        config=config,
        start_time=0.0,
        hardware={"warnings": []},
    )

    assert result_dia16["summary"]["backtest_auto_n"] == 2
    assert load_calls == [True, True]


def test_worldcup_results_refresh_ignores_future_fixture_dates(tmp_path, monkeypatch):
    from src.web import mundial_services as services
    import src.worldcup.fotmob_provider as fotmob_provider

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 13, 23, 0, tzinfo=timezone.utc))
    requested_dates = []
    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-13", "time": "12:00 UTC+0", "team1": "France", "team2": "Brazil", "group": "Group A"},
            {"num": 2, "date": "2026-06-14", "time": "12:00 UTC+0", "team1": "England", "team2": "Argentina", "group": "Group A"},
        ],
    }

    def fake_fotmob_get_json(url, params=None):
        requested_dates.append((params or {}).get("date"))
        return {
            "matches": [
                _fotmob_event(201, "France", "Brazil", 0, 1),
                _fotmob_event(202, "England", "Argentina", 0, 2),
            ],
        }

    monkeypatch.setattr(fotmob_provider, "fotmob_get_json", fake_fotmob_get_json)

    refresh = worldcup_data.refresh_worldcup_2026_results(tournament, refresh=True)
    confirmed = services.confirmed_worldcup_2026_backtest_rows(tournament)

    assert requested_dates == ["20260613"]
    assert refresh["fotmob_final_rows"] == 1
    assert pd.read_csv(results_path)["date"].tolist() == ["2026-06-13"]
    assert confirmed["Date"].tolist() == ["2026-06-13"]


def test_confirmed_backtest_rows_wait_until_real_kickoff_time(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 11, 0, tzinfo=timezone.utc))
    pd.DataFrame([
        {"date": "2026-06-14", "home": "England", "away": "Argentina", "home_goals": 0, "away_goals": 2, "status": "final", "source": "manual", "updated_at": "2026-06-14T11:00:00+00:00"},
    ], columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)
    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-14", "time": "12:00 UTC+0", "team1": "England", "team2": "Argentina", "group": "Group A"},
        ],
    }

    assert services.confirmed_worldcup_2026_backtest_rows(tournament).empty

    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc))
    assert services.confirmed_worldcup_2026_backtest_rows(tournament).empty

    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 12, 1, tzinfo=timezone.utc))
    confirmed = services.confirmed_worldcup_2026_backtest_rows(tournament)
    assert confirmed.shape[0] == 1
    assert confirmed.iloc[0]["Team 1"] == "England"


def test_no_typeerror_entre_timestamp_y_str_en_filtro_de_fecha_en_backtest(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 16, 8, 0, tzinfo=timezone.utc))
    pd.DataFrame([
        {"date": "2026-06-14", "home": "England", "away": "Argentina", "home_goals": 2, "away_goals": 1, "status": "final", "source": "manual", "updated_at": "2026-06-14T21:00:00+00:00"},
        {"date": "2026-06-15", "home": "France", "away": "Brazil", "home_goals": 0, "away_goals": 0, "status": "final", "source": "manual", "updated_at": "2026-06-15T21:00:00+00:00"},
    ], columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)

    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-14", "time": "12:00 UTC+0", "team1": "England", "team2": "Argentina", "group": "Group A"},
            {"num": 2, "date": pd.Timestamp("2026-06-15"), "time": "12:00 UTC+0", "team1": "France", "team2": "Brazil", "group": "Group B"},
        ],
    }

    confirmed = services.confirmed_worldcup_2026_backtest_rows(tournament)
    assert [row["Date"] for _, row in confirmed.iterrows()] == ["2026-06-14", "2026-06-15"]


def test_worldcup_results_refresh_skips_non_final_current_date(tmp_path, monkeypatch):
    from src.web import mundial_services as services
    import src.worldcup.fotmob_provider as fotmob_provider

    worldcup_data, _ = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 15, 0, tzinfo=timezone.utc))
    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-14", "time": "12:00 UTC+0", "team1": "England", "team2": "Argentina", "group": "Group A"},
        ],
    }

    monkeypatch.setattr(
        fotmob_provider,
        "fotmob_get_json",
        lambda url, params=None: {"matches": [_fotmob_event(301, "England", "Argentina", 0, 0, finished=False)]},
    )

    refresh = worldcup_data.refresh_worldcup_2026_results(tournament, refresh=True)
    confirmed = services.confirmed_worldcup_2026_backtest_rows(tournament)

    assert refresh["fotmob_final_rows"] == 0
    assert refresh["confirmed_results"] == 0
    assert confirmed.empty


def test_autorefresh_de_resultados_no_queda_atrapado(tmp_path, monkeypatch):
    from src.web import mundial_services as services
    import src.worldcup.data as worldcup_data

    _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 10, 0, tzinfo=timezone.utc))

    refresh_calls: list[str] = []

    def fake_refresh_worldcup_2026_results(tournament, refresh=False):
        refresh_calls.append("called")
        return {
            "source": "fake-refresh",
            "provider": "test-provider",
            "refresh_attempted": bool(refresh),
            "refresh_added": 0,
            "refresh_updated": 0,
            "fotmob_final_rows": 0,
            "sofascore_final_rows": 0,
            "verified_final_rows": 0,
            "conflicts": [],
            "warnings": [],
            "provider_warnings": [],
            "missing_result_fixtures": [],
        }

    monkeypatch.setattr(services, "refresh_worldcup_2026_results", fake_refresh_worldcup_2026_results)
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: ({}, "test:tournament"))
    monkeypatch.setattr(services, "_WORLD_CUP_RESULTS_AUTO_REFRESHED", False)
    monkeypatch.setattr(services, "_WORLD_CUP_RESULTS_AUTO_REFRESH_SUMMARY", {})
    monkeypatch.setattr(services, "_WORLD_CUP_RESULTS_AUTO_REFRESH_EXPIRES_AT", None)

    first = services.ensure_worldcup_results_autorefresh_once({"matches": []})
    second_same_day = services.ensure_worldcup_results_autorefresh_once({"matches": []})
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 15, 9, 0, tzinfo=timezone.utc))
    second_day = services.ensure_worldcup_results_autorefresh_once({"matches": []})

    assert len(refresh_calls) == 2
    assert first == second_same_day
    assert first["provider"] == "test-provider"
    assert second_day["provider"] == "test-provider"


def test_worldcup_results_refresh_falls_back_to_csv_when_fotmob_fails(tmp_path, monkeypatch):
    from src.web import mundial_services as services
    import src.worldcup.fotmob_provider as fotmob_provider

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 23, 0, tzinfo=timezone.utc))
    pd.DataFrame([
        {"date": "2026-06-14", "home": "England", "away": "Argentina", "home_goals": 0, "away_goals": 2, "status": "final", "source": "manual", "updated_at": "2026-06-14T23:00:00+00:00"},
    ], columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)
    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-14", "time": "12:00 UTC+0", "team1": "England", "team2": "Argentina", "group": "Group A"},
        ],
    }

    def fake_fotmob_get_json(url, params=None):
        raise RuntimeError("down")

    monkeypatch.setattr(fotmob_provider, "fotmob_get_json", fake_fotmob_get_json)
    monkeypatch.setattr(worldcup_data, "fetch_sofascore_worldcup_result_rows", lambda working, warnings: [])

    refresh = worldcup_data.refresh_worldcup_2026_results(tournament, refresh=True)
    confirmed = services.confirmed_worldcup_2026_backtest_rows(tournament)

    assert refresh["refresh_attempted"] is True
    assert refresh["provider"] == "local_csv"
    assert refresh["fotmob_final_rows"] == 0
    assert refresh["confirmed_results"] == 1
    assert "backtest parcial por fuente no disponible" not in refresh["warnings"]


def test_worldcup_results_refresh_uses_verified_local_rows_when_remote_sources_are_empty(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc))
    pd.DataFrame([
        {"date": "2026-06-11", "home": "Mexico", "away": "South Africa", "home_goals": 2, "away_goals": 0, "status": "final", "source": "guardian", "updated_at": "2026-06-12T00:30:00+00:00"},
        {"date": "2026-06-11", "home": "South Korea", "away": "Czech Republic", "home_goals": 2, "away_goals": 1, "status": "final", "source": "guardian", "updated_at": "2026-06-12T00:30:00+00:00"},
    ], columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)
    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-11", "time": "13:00 UTC-6", "team1": "Mexico", "team2": "South Africa", "group": "Group A"},
            {"num": 2, "date": "2026-06-11", "time": "20:00 UTC-6", "team1": "South Korea", "team2": "Czech Republic", "group": "Group A"},
            {"num": 5, "date": "2026-06-12", "time": "18:00 UTC-4", "team1": "Canada", "team2": "Bosnia & Herzegovina", "group": "Group B"},
            {"num": 9, "date": "2026-06-12", "time": "18:00 UTC-7", "team1": "USA", "team2": "Paraguay", "group": "Group D"},
            {"num": 6, "date": "2026-06-13", "time": "15:00 UTC-4", "team1": "Qatar", "team2": "Switzerland", "group": "Group B"},
            {"num": 13, "date": "2026-06-13", "time": "18:00 UTC-4", "team1": "Brazil", "team2": "Morocco", "group": "Group C"},
            {"num": 14, "date": "2026-06-13", "time": "21:00 UTC-4", "team1": "Haiti", "team2": "Scotland", "group": "Group C"},
            {"num": 10, "date": "2026-06-13", "time": "21:00 UTC-5", "team1": "Australia", "team2": "Turkey", "group": "Group D"},
        ],
    }

    monkeypatch.setattr(worldcup_data, "fetch_fotmob_worldcup_result_rows", lambda working, warnings: [])
    monkeypatch.setattr(worldcup_data, "fetch_sofascore_worldcup_result_rows", lambda working, warnings: [])

    refresh = worldcup_data.refresh_worldcup_2026_results(tournament, refresh=True)
    confirmed = services.confirmed_worldcup_2026_backtest_rows(tournament)

    assert refresh["refresh_attempted"] is True
    assert refresh["fotmob_final_rows"] == 0
    assert refresh["sofascore_final_rows"] == 0
    assert refresh["verified_final_rows"] == 8
    assert refresh["refresh_added"] == 6
    assert refresh["refresh_updated"] == 2
    assert refresh["confirmed_results"] == 8
    assert confirmed.shape[0] == 8
    assert confirmed["Date"].tolist() == [
        "2026-06-11",
        "2026-06-11",
        "2026-06-12",
        "2026-06-12",
        "2026-06-13",
        "2026-06-13",
        "2026-06-13",
        "2026-06-13",
    ]
    haiti = confirmed[confirmed["Team 1"].eq("Haiti") & confirmed["Team 2"].eq("Scotland")].iloc[0]
    assert haiti["G1"] == 1
    assert haiti["G2"] == 2
    assert pd.read_csv(results_path).shape[0] == 8


def test_worldcup_results_refresh_and_backtest_auto_use_nine_verified_finals(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 20, 15, tzinfo=timezone.utc))
    pd.DataFrame(columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)
    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-11", "time": "13:00 UTC-6", "team1": "Mexico", "team2": "South Africa", "group": "Group A"},
            {"num": 2, "date": "2026-06-11", "time": "20:00 UTC-6", "team1": "South Korea", "team2": "Czech Republic", "group": "Group A"},
            {"num": 5, "date": "2026-06-12", "time": "18:00 UTC-4", "team1": "Canada", "team2": "Bosnia & Herzegovina", "group": "Group B"},
            {"num": 9, "date": "2026-06-12", "time": "18:00 UTC-7", "team1": "USA", "team2": "Paraguay", "group": "Group D"},
            {"num": 6, "date": "2026-06-13", "time": "15:00 UTC-4", "team1": "Qatar", "team2": "Switzerland", "group": "Group B"},
            {"num": 13, "date": "2026-06-13", "time": "18:00 UTC-4", "team1": "Brazil", "team2": "Morocco", "group": "Group C"},
            {"num": 14, "date": "2026-06-13", "time": "21:00 UTC-4", "team1": "Haiti", "team2": "Scotland", "group": "Group C"},
            {"num": 10, "date": "2026-06-13", "time": "21:00 UTC-5", "team1": "Australia", "team2": "Turkey", "group": "Group D"},
            {"num": 17, "date": "2026-06-14", "time": "18:00 UTC-4", "team1": "Germany", "team2": "Curaçao", "group": "Group E"},
        ],
    }
    def fake_fotmob_rows(working, warnings):
        warnings.append("FotMob 2026-06-14: HTTPError.")
        return []

    def fake_sofascore_rows(working, warnings):
        warnings.append("SofaScore 2026-06-14: TypeError.")
        return []

    monkeypatch.setattr(worldcup_data, "fetch_fotmob_worldcup_result_rows", fake_fotmob_rows)
    monkeypatch.setattr(worldcup_data, "fetch_sofascore_worldcup_result_rows", fake_sofascore_rows)

    refresh = worldcup_data.refresh_worldcup_2026_results(tournament, refresh=True)
    confirmed = services.confirmed_worldcup_2026_backtest_rows(tournament)

    assert refresh["verified_final_rows"] == 9
    assert refresh["confirmed_results"] == 9
    assert refresh["warnings"] == []
    assert "FotMob 2026-06-14: HTTPError." in refresh["provider_warnings"]
    assert "SofaScore 2026-06-14: TypeError." in refresh["provider_warnings"]
    assert refresh["missing_result_fixtures"] == []
    assert confirmed.shape[0] == 9
    germany = confirmed[confirmed["Team 1"].eq("Germany") & confirmed["Team 2"].eq("Curaçao")].iloc[0]
    assert germany["G1"] == 7
    assert germany["G2"] == 1
    stored = pd.read_csv(results_path)
    assert stored.shape[0] == 9
    stored_germany = stored[stored["home"].eq("Germany") & stored["away"].eq("Curaçao")].iloc[0]
    assert int(stored_germany["home_goals"]) == 7
    assert int(stored_germany["away_goals"]) == 1

    class FakeModel:
        max_goals = 10

        def match_probabilities(self, home, away, max_goals=None):
            return {
                "home": 0.52,
                "draw": 0.26,
                "away": 0.22,
                "over05": 0.9,
                "under05": 0.1,
                "over15": 0.7,
                "under15": 0.3,
                "over25": 0.45,
                "under25": 0.55,
                "over35": 0.2,
                "under35": 0.8,
                "lambda1": 1.3,
                "lambda2": 0.9,
            }

        def score_model_metadata(self):
            return {"key": "independent_poisson", "label": "Poisson", "available": True, "params": {}, "warnings": []}

    class FakeWorldCupModel:
        @classmethod
        def from_history(cls, historical_df, teams, **kwargs):
            return FakeModel()

    history = pd.DataFrame([
        {"Date": "2022-11-20", "Year": 2022, "Team 1": "Mexico", "Team 2": "Canada", "G1": 1, "G2": 0, "Round": "Group", "Group": "A"},
        {"Date": "2022-11-21", "Year": 2022, "Team 1": "Germany", "Team 2": "Brazil", "G1": 2, "G2": 1, "Round": "Group", "Group": "A"},
    ])
    monkeypatch.setattr(services, "WorldCupModel", FakeWorldCupModel)
    monkeypatch.setattr(services, "fixture_results_status", lambda fixture_df=None: {"source": "verified-test"})

    result = services.alternatives_backtest_report(
        history_df=history,
        tournament=tournament,
        config=services.report_pipeline_config({}, services.ALTERNATIVES_BENCHMARK_PIPELINE_MODE),
        model_sequence=["independent_poisson"],
        start_time=0.0,
        hardware={},
    )

    assert result["summary"]["evaluated_matches"] == 9
    assert result["models"][0]["evaluated_matches"] == 9
    assert len(result["models"][0]["matches"]) == 9


def test_worldcup_results_refresh_matches_cote_divoire_alias_from_fotmob(tmp_path, monkeypatch):
    from src.web import mundial_services as services
    import src.worldcup.fotmob_provider as fotmob_provider

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 15, 2, 15, tzinfo=timezone.utc))
    pd.DataFrame(columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)
    tournament = {
        "matches": [
            {"num": 26, "date": "2026-06-14", "time": "19:00 UTC-4", "team1": "Ivory Coast", "team2": "Ecuador", "group": "Group E"},
        ],
    }

    monkeypatch.setattr(
        fotmob_provider,
        "fotmob_get_json",
        lambda url, params=None: {"matches": [_fotmob_event(2601, "Côte d’Ivoire", "Ecuador", 1, 0)]},
    )
    monkeypatch.setattr(worldcup_data, "fetch_sofascore_worldcup_result_rows", lambda working, warnings: [])

    refresh = worldcup_data.refresh_worldcup_2026_results(tournament, refresh=True)
    stored = pd.read_csv(results_path).iloc[0]

    assert refresh["fotmob_final_rows"] == 1
    assert refresh["confirmed_results"] == 1
    assert stored["home"] == "Ivory Coast"
    assert stored["away"] == "Ecuador"
    assert int(stored["home_goals"]) == 1
    assert int(stored["away_goals"]) == 0
    assert str(stored["source"]).startswith("fotmob:2601")


def test_worldcup_results_refresh_and_backtest_auto_use_ten_verified_finals(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 15, 2, 15, tzinfo=timezone.utc))
    pd.DataFrame(columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)
    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-11", "time": "13:00 UTC-6", "team1": "Mexico", "team2": "South Africa", "group": "Group A"},
            {"num": 2, "date": "2026-06-11", "time": "20:00 UTC-6", "team1": "South Korea", "team2": "Czech Republic", "group": "Group A"},
            {"num": 5, "date": "2026-06-12", "time": "18:00 UTC-4", "team1": "Canada", "team2": "Bosnia & Herzegovina", "group": "Group B"},
            {"num": 9, "date": "2026-06-12", "time": "18:00 UTC-7", "team1": "USA", "team2": "Paraguay", "group": "Group D"},
            {"num": 6, "date": "2026-06-13", "time": "15:00 UTC-4", "team1": "Qatar", "team2": "Switzerland", "group": "Group B"},
            {"num": 13, "date": "2026-06-13", "time": "18:00 UTC-4", "team1": "Brazil", "team2": "Morocco", "group": "Group C"},
            {"num": 14, "date": "2026-06-13", "time": "21:00 UTC-4", "team1": "Haiti", "team2": "Scotland", "group": "Group C"},
            {"num": 10, "date": "2026-06-13", "time": "21:00 UTC-5", "team1": "Australia", "team2": "Turkey", "group": "Group D"},
            {"num": 17, "date": "2026-06-14", "time": "18:00 UTC-4", "team1": "Germany", "team2": "Curaçao", "group": "Group E"},
            {"num": 26, "date": "2026-06-14", "time": "19:00 UTC-4", "team1": "Ivory Coast", "team2": "Ecuador", "group": "Group E"},
        ],
    }

    monkeypatch.setattr(worldcup_data, "fetch_fotmob_worldcup_result_rows", lambda working, warnings: [])
    monkeypatch.setattr(worldcup_data, "fetch_sofascore_worldcup_result_rows", lambda working, warnings: [])

    refresh = worldcup_data.refresh_worldcup_2026_results(tournament, refresh=True)
    confirmed = services.confirmed_worldcup_2026_backtest_rows(tournament)

    assert refresh["verified_final_rows"] == 10
    assert refresh["confirmed_results"] == 10
    assert refresh["missing_result_fixtures"] == []
    assert confirmed.shape[0] == 10
    ivory = confirmed[confirmed["Team 1"].eq("Ivory Coast") & confirmed["Team 2"].eq("Ecuador")].iloc[0]
    assert ivory["G1"] == 1
    assert ivory["G2"] == 0
    stored = pd.read_csv(results_path)
    assert stored.shape[0] == 10
    stored_ivory = stored[stored["home"].eq("Ivory Coast") & stored["away"].eq("Ecuador")].iloc[0]
    assert int(stored_ivory["home_goals"]) == 1
    assert int(stored_ivory["away_goals"]) == 0
    assert stored_ivory["source"] == "verified:guardian"


def test_worldcup_results_refresh_uses_verified_netherlands_japan_result(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 22, 30, tzinfo=timezone.utc))
    pd.DataFrame(columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)
    tournament = {
        "matches": [
            {"num": 31, "date": "2026-06-14", "time": "15:00 UTC-5", "team1": "Netherlands", "team2": "Japan", "group": "Group F"},
        ],
    }

    monkeypatch.setattr(worldcup_data, "fetch_fotmob_worldcup_result_rows", lambda working, warnings: [])
    monkeypatch.setattr(worldcup_data, "fetch_sofascore_worldcup_result_rows", lambda working, warnings: [])

    refresh = worldcup_data.refresh_worldcup_2026_results(tournament, refresh=True)
    stored = pd.read_csv(results_path).iloc[0]

    assert refresh["verified_final_rows"] == 1
    assert refresh["confirmed_results"] == 1
    assert stored["home"] == "Netherlands"
    assert stored["away"] == "Japan"
    assert int(stored["home_goals"]) == 2
    assert int(stored["away_goals"]) == 2


def test_mundial_fixtures_service_autorefreshes_results_on_first_load(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 15, 2, 15, tzinfo=timezone.utc))
    pd.DataFrame(columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)
    tournament = {
        "matches": [
            {"num": 26, "date": "2026-06-14", "time": "19:00 UTC-4", "team1": "Ivory Coast", "team2": "Ecuador", "group": "Group E"},
        ],
    }

    monkeypatch.setattr(worldcup_data, "fetch_fotmob_worldcup_result_rows", lambda working, warnings: [])
    monkeypatch.setattr(worldcup_data, "fetch_sofascore_worldcup_result_rows", lambda working, warnings: [])
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: (tournament, "test:tournament"))
    monkeypatch.setattr(services, "_WORLD_CUP_RESULTS_AUTO_REFRESHED", False)
    monkeypatch.setattr(services, "_WORLD_CUP_RESULTS_AUTO_REFRESH_SUMMARY", {})

    payload = services.fixtures()
    fixture = payload["fixtures"][0]

    assert payload["confirmed_results"] == 1
    assert fixture["label"] == "Ivory Coast vs Ecuador"
    assert fixture["finished"] is True
    assert fixture["score_home"] == 1
    assert fixture["score_away"] == 0
    assert fixture["result_source"] == "verified:guardian"


def test_worldcup_results_refresh_updates_conflicting_local_result(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc))
    pd.DataFrame([
        {"date": "2026-06-13", "home": "Haiti", "away": "Scotland", "home_goals": 0, "away_goals": 1, "status": "final", "source": "guardian", "updated_at": "2026-06-14T08:00:00+00:00"},
    ], columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)
    tournament = {
        "matches": [
            {"num": 14, "date": "2026-06-13", "time": "21:00 UTC-4", "team1": "Haiti", "team2": "Scotland", "group": "Group C"},
        ],
    }

    monkeypatch.setattr(worldcup_data, "fetch_fotmob_worldcup_result_rows", lambda working, warnings: [])
    monkeypatch.setattr(worldcup_data, "fetch_sofascore_worldcup_result_rows", lambda working, warnings: [])

    refresh = worldcup_data.refresh_worldcup_2026_results(tournament, refresh=True)
    confirmed = services.confirmed_worldcup_2026_backtest_rows(tournament)

    assert refresh["verified_final_rows"] == 1
    assert refresh["refresh_added"] == 0
    assert refresh["refresh_updated"] == 1
    assert refresh["conflicts"][0]["existing_score"] == "0-1"
    assert refresh["conflicts"][0]["incoming_score"] == "1-2"
    assert confirmed.iloc[0]["G1"] == 1
    assert confirmed.iloc[0]["G2"] == 2
    stored = pd.read_csv(results_path).iloc[0]
    assert int(stored["home_goals"]) == 1
    assert int(stored["away_goals"]) == 2


def test_verified_local_rows_wait_until_result_is_available(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    worldcup_data, results_path = _patch_worldcup_results_file(monkeypatch, tmp_path)
    _freeze_worldcup_now(monkeypatch, services, worldcup_data, datetime(2026, 6, 13, 21, 30, tzinfo=timezone.utc))
    pd.DataFrame(columns=worldcup_data.RESULT_OVERRIDE_COLUMNS).to_csv(results_path, index=False)
    tournament = {
        "matches": [
            {"num": 13, "date": "2026-06-13", "time": "18:00 UTC-4", "team1": "Brazil", "team2": "Morocco", "group": "Group C"},
            {"num": 14, "date": "2026-06-13", "time": "21:00 UTC-4", "team1": "Haiti", "team2": "Scotland", "group": "Group C"},
        ],
    }

    monkeypatch.setattr(worldcup_data, "fetch_fotmob_worldcup_result_rows", lambda working, warnings: [])
    monkeypatch.setattr(worldcup_data, "fetch_sofascore_worldcup_result_rows", lambda working, warnings: [])

    refresh = worldcup_data.refresh_worldcup_2026_results(tournament, refresh=True)

    assert refresh["verified_final_rows"] == 0
    assert refresh["confirmed_results"] == 0


def test_alternatives_backtest_auto_unavailable_without_confirmed_2026_results(monkeypatch):
    from src.web import mundial_services as services

    history = pd.DataFrame([
        {"Date": "2022-11-20", "Year": 2022, "Team 1": "Argentina", "Team 2": "France", "G1": 2, "G2": 1, "Round": "Group", "Group": "A"},
        {"Date": "2022-11-21", "Year": 2022, "Team 1": "Brazil", "Team 2": "England", "G1": 1, "G2": 1, "Round": "Group", "Group": "A"},
        {"Date": "2022-11-22", "Year": 2022, "Team 1": "France", "Team 2": "Brazil", "G1": 0, "G2": 1, "Round": "Group", "Group": "A"},
        {"Date": "2022-11-23", "Year": 2022, "Team 1": "England", "Team 2": "Argentina", "G1": 0, "G2": 2, "Round": "Group", "Group": "A"},
    ])
    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-20", "time": "18:00 UTC+0", "team1": "Argentina", "team2": "England", "group": "Group A"},
        ],
    }
    monkeypatch.setattr(services, "fixture_results_status", lambda fixture_df=None: {"source": "test-results"})

    result = services.alternatives_backtest_report(
        history_df=history,
        tournament=tournament,
        config=services.report_pipeline_config({}, services.ALTERNATIVES_BENCHMARK_PIPELINE_MODE),
        model_sequence=["independent_poisson"],
        start_time=0.0,
        hardware={},
    )

    assert result["summary"]["available"] is False
    assert result["summary"]["scope"] == "worldcup_2026_confirmed_auto"
    assert result["summary"]["evaluated_matches"] == 0
    assert result["summary"]["confirmed_matches"] == 0
    assert "no hay partidos confirmados" in result["warnings"][0].lower()


def test_alternatives_backtest_auto_uses_confirmed_2026_walk_forward_without_leakage(monkeypatch):
    from src.web import mundial_services as services

    train_snapshots = []

    class FakeModel:
        max_goals = 10

        def __init__(self, train_df):
            self.train_df = train_df

        def match_probabilities(self, home, away, max_goals=None):
            return {
                "home": 0.6,
                "draw": 0.2,
                "away": 0.2,
                "over05": 0.9,
                "under05": 0.1,
                "over15": 0.7,
                "under15": 0.3,
                "over25": 0.45,
                "under25": 0.55,
                "over35": 0.2,
                "under35": 0.8,
                "lambda1": 1.4,
                "lambda2": 0.9,
            }

        def score_model_metadata(self):
            return {"key": "independent_poisson", "label": "Poisson", "available": True, "params": {}, "warnings": []}

    class FakeWorldCupModel:
        @classmethod
        def from_history(cls, historical_df, teams, **kwargs):
            train_snapshots.append(historical_df.copy())
            return FakeModel(historical_df.copy())

    history = pd.DataFrame([
        {"Date": "2022-11-20", "Year": 2022, "Team 1": "Argentina", "Team 2": "France", "G1": 2, "G2": 1, "Round": "Group", "Group": "A"},
        {"Date": "2022-11-21", "Year": 2022, "Team 1": "Brazil", "Team 2": "England", "G1": 1, "G2": 1, "Round": "Group", "Group": "A"},
    ])
    tournament = {
        "matches": [
            {"num": 1, "date": "2026-06-11", "time": "12:00 UTC+0", "team1": "Argentina", "team2": "France", "group": "Group A", "score": {"ft": [2, 1]}},
            {"num": 2, "date": "2026-06-12", "time": "12:00 UTC+0", "team1": "Brazil", "team2": "England", "group": "Group A", "score": {"ft": [1, 1]}},
            {"num": 3, "date": "2026-06-13", "time": "12:00 UTC+0", "team1": "France", "team2": "Brazil", "group": "Group A", "score": {"ft": [0, 1]}},
        ],
    }
    monkeypatch.setattr(services, "WorldCupModel", FakeWorldCupModel)
    monkeypatch.setattr(services, "fixture_results_status", lambda fixture_df=None: {"source": "test-results"})

    result = services.alternatives_backtest_report(
        history_df=history,
        tournament=tournament,
        config=services.report_pipeline_config({}, services.ALTERNATIVES_BENCHMARK_PIPELINE_MODE),
        model_sequence=["independent_poisson"],
        start_time=0.0,
        hardware={},
    )

    assert result["summary"]["available"] is True
    assert result["summary"]["evaluated_matches"] == 3
    assert result["summary"]["confirmed_matches"] == 3
    assert [len(frame) for frame in train_snapshots] == [2, 3, 4]
    assert train_snapshots[0][train_snapshots[0]["Year"].eq(2026)].empty
    assert train_snapshots[1][train_snapshots[1]["Year"].eq(2026)]["Team 1"].tolist() == ["Argentina"]
    assert train_snapshots[2][train_snapshots[2]["Year"].eq(2026)]["Team 1"].tolist() == ["Argentina", "Brazil"]
    assert "France" not in train_snapshots[2][train_snapshots[2]["Year"].eq(2026)].tail(1)["Team 1"].tolist()
    assert result["models"][0]["evaluated_matches"] == 3
    assert result["models"][0]["over_under_accuracy"] == 0.75
    assert result["models"][0]["over_under_accuracy_by_line"]["0.5"] == 1.0
    assert len(result["models"][0]["matches"]) == 3
    assert result["models"][0]["matches"][0]["pick_hit"] is True
    assert result["models"][0]["matches"][0]["over_under"][0]["hit"] is True
    assert result["models"][0]["rank"] == 1


def test_recent_matches_for_fixture_uses_last_15_before_fixture_date():
    from src.web import mundial_services as services

    history = pd.DataFrame([
        {
            "Date": f"2026-05-{day:02d}",
            "Year": 2026,
            "Team 1": "Mexico" if day % 2 else "Canada",
            "Team 2": "Canada" if day % 2 else "Mexico",
            "G1": day % 4,
            "G2": (day + 1) % 3,
            "Round": "Friendly",
            "Group": "",
        }
        for day in range(1, 21)
    ] + [
        {"Date": "2026-06-12", "Year": 2026, "Team 1": "Mexico", "Team 2": "Canada", "G1": 9, "G2": 0, "Round": "Future", "Group": ""},
    ])
    fixture = pd.Series({"Fecha": "2026-06-11", "Equipo 1": "Mexico", "Equipo 2": "Canada"})

    recent = services.recent_matches_for_fixture(history, fixture, limit=15)

    assert recent["limit"] == 15
    assert len(recent["home"]) == 15
    assert len(recent["away"]) == 15
    assert recent["home"][0]["date"] == "2026-05-20"
    assert all(item["date"] < "2026-06-11" for item in recent["home"])
    assert all(item["score"] != "9-0" for item in recent["home"])


def test_recent_matches_for_fixture_uses_international_results_when_provided(monkeypatch):
    from src.web import mundial_services as services

    history = pd.DataFrame([
        {"Date": "2022-11-22", "Year": 2022, "Team 1": "Mexico", "Team 2": "Poland", "G1": 0, "G2": 0, "Round": "FIFA World Cup", "Group": "C"},
        {"Date": "2022-12-01", "Year": 2022, "Team 1": "Canada", "Team 2": "Morocco", "G1": 1, "G2": 2, "Round": "FIFA World Cup", "Group": "F"},
    ])
    international = pd.DataFrame([
        {"date": "2026-06-08", "home_team": "Mexico", "away_team": "Switzerland", "home_score": 1, "away_score": 0, "tournament": "Friendly", "neutral": False},
        {"date": "2026-06-09", "home_team": "Canada", "away_team": "Japan", "home_score": 2, "away_score": 2, "tournament": "Friendly", "neutral": True},
        {"date": "2026-06-10", "home_team": "Mexico", "away_team": "Canada", "home_score": 3, "away_score": 1, "tournament": "CONCACAF Gold Cup", "neutral": True},
        {"date": "2026-06-11", "home_team": "Mexico", "away_team": "Canada", "home_score": 9, "away_score": 0, "tournament": "Future leak", "neutral": True},
        {"date": "2026-06-12", "home_team": "Canada", "away_team": "Mexico", "home_score": None, "away_score": None, "tournament": "Future unscored", "neutral": True},
    ])
    international.attrs["source_path"] = "storage/worldcup/international/all_matches.csv"
    monkeypatch.setattr(services, "international_results_status", lambda: {
        "source_path": "storage/worldcup/international/all_matches.csv",
        "max_scored_date": "2026-06-10",
        "warnings": ["Dataset internacional posiblemente viejo"],
    })
    fixture = pd.Series({"Fecha": "2026-06-11", "Equipo 1": "Mexico", "Equipo 2": "Canada"})

    recent = services.recent_matches_for_fixture(history, fixture, limit=2, international_matches=international)
    html = services.recent_matches_report_html(recent, {"home": "Mexico", "away": "Canada"})

    assert recent["source"] == "all_matches.csv"
    assert [row["date"] for row in recent["home"]] == ["2026-06-10", "2026-06-08"]
    assert [row["date"] for row in recent["away"]] == ["2026-06-10", "2026-06-09"]
    assert all(row["date"] != "2022-11-22" for row in recent["home"])
    assert all(row["date"] < "2026-06-11" for row in recent["home"] + recent["away"])
    assert recent["home"][0]["tournament"] == "CONCACAF Gold Cup"
    assert recent["home"][1]["match_type"] == "Friendly"
    assert recent["home"][0]["weight"] > recent["home"][1]["weight"]
    assert recent["home"][0]["importance_label"] == "Muy alta"
    assert "Fuente all_matches.csv" in html
    assert "Dataset internacional posiblemente viejo" in html
    assert "Torneo" in html
    assert "Peso" in html
    assert "Oficial" in html


def test_benchmark_feature_context_uses_only_pre_match_history(monkeypatch):
    from src.web import mundial_services as services
    from src.worldcup.model import WorldCupModel

    history = pd.DataFrame([
        {"Date": "2026-06-09", "Year": 2026, "Team 1": "Mexico", "Team 2": "Canada", "G1": 1, "G2": 0, "Round": "Group", "Group": "A"},
        {"Date": "2026-06-10", "Year": 2026, "Team 1": "Mexico", "Team 2": "Canada", "G1": 8, "G2": 0, "Round": "Group", "Group": "A"},
    ])
    tournament = {"matches": [{"num": 1, "date": "2026-06-10", "team1": "Mexico", "team2": "Canada", "group": "Group A"}]}
    model = WorldCupModel.from_history(history.iloc[:1], teams=["Mexico", "Canada"])

    source = services.BenchmarkFeatureSource(tournament=tournament, history_df=history, config={})
    context = source.context_for_match(
        model,
        {"No.": 1, "Fecha": "2026-06-10", "Equipo 1": "Mexico", "Equipo 2": "Canada", "Grupo": "Group A"},
        model_key="independent_poisson",
    )

    assert context["history_rows"] == 1
    assert context["_feature_row"]["history_last_3_goals_for_avg_home"] == pytest.approx(1.0)
    assert context["_feature_row"]["history_last_3_goal_diff_avg_diff"] == pytest.approx(2.0)


def test_benchmark_feature_enhanced_model_works_without_optional_caches(monkeypatch):
    from src.web import mundial_services as services
    from src.worldcup.model import WorldCupModel

    monkeypatch.setattr(services, "load_market_data", lambda **kwargs: {"matches": pd.DataFrame(), "qualifiers": pd.DataFrame(), "warnings": []})
    monkeypatch.setattr(services, "load_api_football_data", lambda **kwargs: {"team_stats": pd.DataFrame(), "lineups": pd.DataFrame(), "injuries": pd.DataFrame(), "market_rows": pd.DataFrame(), "warnings": []})
    monkeypatch.setattr(services, "load_international_matches", lambda required=False: pd.DataFrame())
    monkeypatch.setattr(services, "player_features_dataframe", lambda tournament: pd.DataFrame())

    history = pd.DataFrame([
        {"Date": "2022-11-20", "Year": 2022, "Team 1": "Mexico", "Team 2": "Canada", "G1": 1, "G2": 0, "Round": "Group", "Group": "A"},
    ])
    model = WorldCupModel.from_history(history, teams=["Mexico", "Canada"])
    source = services.BenchmarkFeatureSource(tournament={"matches": []}, history_df=history, config={})
    enhanced = services.apply_benchmark_feature_model(model, "independent_poisson", source, history_df=history)

    probabilities = enhanced.match_probabilities_for_match("Mexico", "Canada", match={"Fecha": "2026-06-11", "Equipo 1": "Mexico", "Equipo 2": "Canada"})

    assert set(probabilities) >= {"home", "draw", "away", "over25", "under25"}
    assert "feature_context" in probabilities
    assert probabilities["feature_context"]["cutoff"] == "strictly_before_match"


def test_worldcup_ui_keeps_fixture_prediction_probability_labels():
    source = Path("src/web/static/mundial.js").read_text(encoding="utf-8")
    assert "Marcador #1" in source
    assert "modelOutcomeProbabilitiesHtml" in source
    assert "Probabilidades 1X2 por modelo" in source
    assert "modelOverUnderProbabilitiesHtml" in source
    assert "future-total-cards" in source
    assert "O ${escapeHtml(formatProbability(over))}%" in source
    assert "U ${escapeHtml(formatProbability(under))}%" in source


def test_consensus_rounding_signature_and_strength_levels():
    from src.web import mundial_services as services

    assert services.round_half_up_int(1.49) == 1
    assert services.round_half_up_int(1.5) == 2
    assert services.consensus_signature("home", {"0.5": "over", "1.5": "over", "2.5": "under", "3.5": "under"}) == "home|over|over|under|under"

    def report(outcome, totals, signature_suffix="", eligible=True):
        signature = services.consensus_signature(outcome, totals)
        if signature_suffix:
            signature = f"{signature}{signature_suffix}"
        return {
            "consensus_eligible": eligible,
            "signature": signature,
            "decision": {"outcome": outcome},
            "totals": totals,
        }

    same_totals = {"0.5": "over", "1.5": "over", "2.5": "under", "3.5": "under"}
    assert services.fixture_consensus([report("home", same_totals) for _ in range(3)])["strength"] == "Muy fuerte"

    strong = [report("home", same_totals) for _ in range(7)]
    strong += [report("home", {"0.5": "over", "1.5": "under", "2.5": "under", "3.5": "under"}, str(index)) for index in range(3)]
    assert services.fixture_consensus(strong)["strength"] == "Fuerte"

    medium = [report("home", same_totals, str(index)) for index in range(6)]
    medium += [report("away", same_totals, str(index)) for index in range(4)]
    assert services.fixture_consensus(medium)["strength"] == "Media"

    low = [report("home", same_totals, str(index)) for index in range(5)]
    low += [report("away", same_totals, str(index)) for index in range(5)]
    assert services.fixture_consensus(low)["strength"] == "Baja"


def test_unavailable_models_are_excluded_from_consensus_except_independent():
    from src.web import mundial_services as services

    fixture = pd.Series({"Equipo 1": "Mexico", "Equipo 2": "Canada"})
    probabilities = {
        "home": 0.55,
        "draw": 0.25,
        "away": 0.20,
        "over05": 0.9,
        "under05": 0.1,
        "over15": 0.7,
        "under15": 0.3,
        "over25": 0.4,
        "under25": 0.6,
        "over35": 0.2,
        "under35": 0.8,
        "lambda1": 1.4,
        "lambda2": 0.9,
    }

    unavailable_xg = services.score_prediction_model_report(
        "xg_poisson_local",
        {"key": "xg_poisson_local", "label": "xG", "available": False, "warnings": ["sin xG"]},
        probabilities,
        fixture,
        {},
        already_percent=False,
    )
    unavailable_independent = services.score_prediction_model_report(
        "independent_poisson",
        {"key": "independent_poisson", "label": "Poisson", "available": False, "warnings": ["fallback"]},
        probabilities,
        fixture,
        {},
        already_percent=False,
    )

    assert unavailable_xg["consensus_eligible"] is False
    assert unavailable_independent["consensus_eligible"] is True
    consensus = services.fixture_consensus([unavailable_xg, unavailable_independent])
    assert consensus["eligible_models"] == 1
    assert consensus["excluded_models"] == 1


def test_poisson_sota_report_runs_models_sequentially_and_saves_latest(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    prediction_order = []
    fit_order = []

    class FakeModel:
        max_goals = 10

        def __init__(self, key="independent_poisson", available=True):
            self.key = key
            self.available = available

        def match_probabilities(self, home, away, max_goals=None):
            prediction_order.append(self.key)
            return {
                "home": 0.55,
                "draw": 0.25,
                "away": 0.20,
                "over05": 0.9,
                "under05": 0.1,
                "over15": 0.7,
                "under15": 0.3,
                "over25": 0.45,
                "under25": 0.55,
                "over35": 0.2,
                "under35": 0.8,
                "lambda1": 1.4,
                "lambda2": 0.9,
                "modal_g1": 1,
                "modal_g2": 0,
            }

        def score_model_metadata(self):
            return {"key": self.key, "label": self.key, "available": self.available, "params": {}, "warnings": []}

    fixtures = pd.DataFrame([
        {"No.": 1, "Fecha": "2026-06-11", "Hora": "18:00 UTC+0", "Grupo": "Group A", "Equipo 1": "Mexico", "Equipo 2": "Canada", "Sede": "A"},
    ])

    class FakeFeatureSource:
        warnings = []

        def context_for_match(self, model, fixture, model_key, history_df=None):
            feature_row = {
                "rating_diff": 25.0,
                "recent15_goal_diff_avg_diff": 0.4,
                "stage_group": 1.0,
                "market_home_prob": 0.0,
            }
            return {
                "available": True,
                "model_key": model_key,
                "reference_date": "2026-06-11",
                "cutoff": "strictly_before_match",
                "history_rows": 12,
                "usage_counts": {
                    "rating": 1,
                    "form": 1,
                    "odds": 0,
                    "xg_shots": 0,
                    "api_football": 0,
                    "xi_players": 0,
                    "h2h": 0,
                    "fixture": 1,
                    "score_grid": 0,
                },
                "available_families": ["rating", "form", "fixture"],
                "feature_count": 4,
                "feature_list": [
                    {"name": "rating_diff", "family": "rating", "value": 25.0, "present": True},
                    {"name": "recent15_goal_diff_avg_diff", "family": "form", "value": 0.4, "present": True},
                    {"name": "stage_group", "family": "fixture", "value": 1.0, "present": True},
                    {"name": "market_home_prob", "family": "odds", "value": 0.0, "present": False},
                ],
                "sample": {"rating_diff": 25.0, "recent15_goal_diff_avg_diff": 0.4},
                "warnings": [],
                "_feature_row": feature_row,
            }

    monkeypatch.setattr(services, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: ({}, "fixture-test"))
    monkeypatch.setattr(services, "build_model", lambda tournament, config: (FakeModel(), "history-test"))
    monkeypatch.setattr(services, "upcoming_fixture_rows", lambda tournament, group_filter="": fixtures)
    monkeypatch.setattr(services, "groups_from_tournament", lambda tournament: {"Group A": ["Mexico", "Canada"]})
    monkeypatch.setattr(services, "load_historical_matches", lambda refresh=False: (pd.DataFrame(), "history-test"))
    monkeypatch.setattr(services, "benchmark_feature_source", lambda tournament, history_df, config: FakeFeatureSource())
    monkeypatch.setattr(services, "contextual_poisson_for_match", lambda *args, **kwargs: {})
    captured_backtest_config = {}

    def fake_backtest_report(**kwargs):
        captured_backtest_config.update(kwargs["config"])
        return {
            "summary": {
                "available": True,
                "evaluated_matches": 1,
                "confirmed_matches": 1,
                "requested_matches": kwargs["config"]["backtest_last_n"],
                "generated_at": "2026-06-18T00:00:00+00:00",
                "backtest_range": {"evaluated_matches": 1},
            },
            "models": [{
                "model_key": "independent_poisson",
                "model_label": "Poisson",
                "available": True,
                "evaluated_matches": 1,
                "rank": 1,
                "score_resultados": 75.0,
                "reliability_score": 75.0,
                "log_loss": 0.3,
                "brier": 0.1,
                "rps": 0.2,
                "expected_calibration_error": 0.04,
                "pick_accuracy": 1.0,
                "score_accuracy": 0.0,
                "top3_score_accuracy": 1.0,
                "over_under_accuracy": 1.0,
                "ou25_log_loss": 0.2,
                "matches": [],
                "vs_poisson": {"summary": "baseline"},
            }],
            "warnings": [],
        }

    monkeypatch.setattr(services, "alternatives_backtest_report", fake_backtest_report)

    def fake_build_score_model(base_model, history_df, teams, config):
        key = config["score_model"]
        fit_order.append(key)
        return FakeModel(key=key, available=True)

    monkeypatch.setattr(services, "build_score_model", fake_build_score_model)
    progress = []

    result = services.predict_upcoming_report(
        {"pipeline_mode": "poisson_sota", "limit": 1, "backtest_last_n": 9},
        progress_callback=progress.append,
    )

    compact_prediction_order = []
    for key in prediction_order:
        if not compact_prediction_order or compact_prediction_order[-1] != key:
            compact_prediction_order.append(key)
    assert compact_prediction_order == services.SOTA_SCORE_MODEL_SEQUENCE
    assert fit_order == services.SOTA_SCORE_MODEL_SEQUENCE[1:]
    assert result["summary"]["sota_calculation_mode"] == "exact"
    assert captured_backtest_config["backtest_last_n"] == 0
    assert captured_backtest_config["backtest_mode"] == "auto_since_opening"
    assert result["summary"]["backtest_last_n"] == 1
    assert result["summary"]["backtest_auto_n"] == 1
    assert result["summary"]["backtest_mode"] == "auto_since_opening"
    assert result["model_backtests"][0]["model_key"] == "independent_poisson"
    assert "use_ml_model" not in result["summary"]
    assert result["fixture_reports"][0]["models"][0]["model_key"] == "independent_poisson"
    assert "monte_carlo_consensus" not in result["fixture_reports"][0]
    assert result["fixture_reports"][0]["consensus"]["eligible_models"] == len(services.SOTA_SCORE_MODEL_SEQUENCE)
    assert len(result["fixture_reports"][0]["top_models_1x2"]) == min(4, len(services.SOTA_SCORE_MODEL_SEQUENCE))
    assert result["fixture_reports"][0]["consensus_score_distribution"]["available"] is True
    assert result["fixture_reports"][0]["consensus_score_distribution"]["score_matrix"]
    assert result["fixture_reports"][0]["consensus_score_distribution"]["score_matrix_home_goals"][0] == 0
    assert result["fixture_reports"][0]["consensus_score_distribution"]["score_matrix_away_goals"][0] == 0
    assert result["fixture_reports"][0]["model_statistics"]["model_count"] == len(services.SOTA_SCORE_MODEL_SEQUENCE)
    assert result["fixture_reports"][0]["models"][0]["score_distribution"]["top_scores"]
    assert result["fixture_reports"][0]["models"][0]["score_distribution"]["score_matrix"]
    assert result["fixture_reports"][0]["models"][0]["feature_context"]["feature_list"]
    assert result["fixture_reports"][0]["models"][0]["feature_context"]["feature_list"][0]["name"]
    assert result["summary"]["pipeline_steps"]
    assert (tmp_path / "latest.json").exists()
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["report_id"] == result["report_id"]
    assert not any(item.get("model_key") == "xg_poisson_local" for item in progress)
    assert {item.get("model_total") for item in progress if item.get("model_key")} == {len(services.SOTA_SCORE_MODEL_SEQUENCE)}


def test_poisson_sota_report_monte_carlo_consensus_uses_form_iterations(tmp_path, monkeypatch):
    from src.web import mundial_services as services

    class FakeModel:
        max_goals = 10

        def match_probabilities(self, home, away, max_goals=None):
            return {
                "home": 0.55,
                "draw": 0.25,
                "away": 0.20,
                "over05": 0.9,
                "under05": 0.1,
                "over15": 0.7,
                "under15": 0.3,
                "over25": 0.45,
                "under25": 0.55,
                "over35": 0.2,
                "under35": 0.8,
                "lambda1": 1.4,
                "lambda2": 0.9,
                "modal_g1": 1,
                "modal_g2": 0,
            }

        def score_model_metadata(self):
            return {"key": "independent_poisson", "label": "Poisson", "available": True, "params": {}, "warnings": []}

    fixtures = pd.DataFrame([
        {"No.": 1, "Fecha": "2026-06-11", "Hora": "18:00 UTC+0", "Grupo": "Group A", "Equipo 1": "Mexico", "Equipo 2": "Canada", "Sede": "A"},
    ])
    captured = {}

    def fake_count_matrix(grid, iterations, seed, backend="numpy"):
        captured["iterations"] = iterations
        captured["backend"] = backend
        captured["grid_shape"] = tuple(np.asarray(grid).shape)
        return np.array([[35000, 15000], [40000, 10000]]), "numpy"

    monkeypatch.setattr(services, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(services, "SOTA_SCORE_MODEL_SEQUENCE", ["independent_poisson"])
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: ({}, "fixture-test"))
    monkeypatch.setattr(services, "build_model", lambda tournament, config: (FakeModel(), "history-test"))
    monkeypatch.setattr(services, "upcoming_fixture_rows", lambda tournament, group_filter="": fixtures)
    monkeypatch.setattr(services, "groups_from_tournament", lambda tournament: {"Group A": ["Mexico", "Canada"]})
    monkeypatch.setattr(services, "load_historical_matches", lambda refresh=False: (pd.DataFrame(), "history-test"))
    monkeypatch.setattr(services, "contextual_poisson_for_match", lambda *args, **kwargs: {})
    monkeypatch.setattr(services, "monte_carlo_count_matrix_from_grid", fake_count_matrix)
    monkeypatch.setattr(services, "detect_hardware", lambda: {
        "cpu_count": 8,
        "default_n_jobs": -1,
        "cuda_available": False,
        "cuda_devices": [],
        "cuda_device_names": [],
        "cuda_detection_source": "none",
        "cuda_detection_sources": [],
        "cuda_error": "sin dispositivos",
        "cuda_warning": "sin dispositivos",
        "device_default": "cpu",
    })

    result = services.predict_upcoming_report({
        "pipeline_mode": "poisson_sota",
        "sota_calculation_mode": "monte_carlo",
        "iterations": 100000,
        "limit": 1,
    })

    monte_carlo = result["fixture_reports"][0]["monte_carlo_consensus"]
    assert result["summary"]["sota_calculation_mode"] == "monte_carlo"
    assert result["summary"]["monte_carlo_iterations"] == 100000
    assert captured["iterations"] == 100000
    assert captured["grid_shape"] == (11, 11)
    assert monte_carlo["available"] is True
    assert monte_carlo["iterations"] == 100000
    assert monte_carlo["source"] == "SOTA Monte Carlo sobre matriz consenso"
    assert "model_weights" not in monte_carlo
    assert "model_sample_counts" not in monte_carlo
    assert set(monte_carlo["probabilities"]) >= {"home", "draw", "away", "over05", "under05", "over25", "under25"}
    assert monte_carlo["top_scores"]


def test_monte_carlo_single_grid_matches_distribution():
    from src.web import mundial_services as services

    grid = np.array([[0.20, 0.30], [0.10, 0.40]])
    counts, backend = services.monte_carlo_count_matrix_from_grid(
        grid=grid,
        iterations=50000,
        seed=17,
        backend="numpy",
    )

    exact = services.score_grid_probabilities(grid)
    simulated = services.score_grid_probabilities(counts)

    assert backend == "numpy"
    for key in ("home", "draw", "away", "over05", "under05"):
        assert abs(simulated[key] - exact[key]) < 1.5


def test_consensus_agreement_and_model_statistics_include_ranges():
    from src.web import mundial_services as services

    reports = [
        {
            "consensus_eligible": True,
            "signature": "home|over|over|under|under",
            "decision": {"outcome": "home"},
            "totals": {"0.5": "over", "1.5": "over", "2.5": "under", "3.5": "under"},
            "probabilities": {"home": 60, "draw": 20, "away": 20, "over05": 90, "under05": 10, "over15": 70, "under15": 30, "over25": 40, "under25": 60, "over35": 20, "under35": 80},
            "expected_goals": {"home": 1.5, "away": 0.8},
            "top_score": "1-0",
        },
        {
            "consensus_eligible": True,
            "signature": "home|over|over|under|under",
            "decision": {"outcome": "home"},
            "totals": {"0.5": "over", "1.5": "over", "2.5": "under", "3.5": "under"},
            "probabilities": {"home": 50, "draw": 25, "away": 25, "over05": 85, "under05": 15, "over15": 65, "under15": 35, "over25": 35, "under25": 65, "over35": 18, "under35": 82},
            "expected_goals": {"home": 1.3, "away": 1.0},
            "top_score": "1-1",
        },
    ]

    consensus = services.fixture_consensus(reports)
    statistics = services.model_statistics_payload(reports)

    assert consensus["agreement"] == {"market": "1X2", "pick": "home", "pick_label": "1", "count": 2, "total": 2, "share": 1.0}
    assert statistics["probability_ranges"]["home"] == {"min": 50.0, "max": 60.0, "spread": 10.0}


def test_sota_report_honors_explicit_cuda_for_exact_cupy_backend(monkeypatch):
    from src.web import mundial_services as services

    monkeypatch.setattr(services, "detect_hardware", lambda: {
        "cpu_count": 16,
        "default_n_jobs": -1,
        "cuda_available": True,
        "cuda_devices": ["GPU 0: NVIDIA GeForce RTX 5070"],
        "cuda_device_names": ["NVIDIA GeForce RTX 5070"],
        "cuda_detection_source": "nvidia-smi:/usr/bin/nvidia-smi",
        "cuda_detection_sources": ["nvidia-smi:/usr/bin/nvidia-smi"],
        "cuda_error": "",
        "cuda_warning": "",
        "device_default": "cuda",
    })
    monkeypatch.setattr(services, "score_backend_status", lambda requested_device="auto": {
        "score_backend": "cupy",
        "actual_device": "cuda",
        "backend_supports_cuda": True,
        "cuda_available": True,
        "cuda_device_names": ["NVIDIA GeForce RTX 5070"],
        "warning": "",
    })

    hardware = services.stat_report_hardware("cuda", "poisson_sota")
    assert hardware["actual_device"] == "cuda"
    assert hardware["backend_supports_cuda"] is True
    assert hardware["score_backend"] == "cupy"
    assert sum("CPU-bound" in warning for warning in hardware["warnings"]) == 0


def test_sota_monte_carlo_hardware_uses_cuda_backend_when_available(monkeypatch):
    from src.web import mundial_services as services

    monkeypatch.setattr(services, "detect_hardware", lambda: {
        "cpu_count": 16,
        "default_n_jobs": -1,
        "cuda_available": True,
        "cuda_devices": ["GPU 0: NVIDIA GeForce RTX 5070"],
        "cuda_device_names": ["NVIDIA GeForce RTX 5070"],
        "cuda_detection_source": "nvidia-smi:/usr/bin/nvidia-smi",
        "cuda_detection_sources": ["nvidia-smi:/usr/bin/nvidia-smi"],
        "cuda_error": "",
        "cuda_warning": "",
        "device_default": "cuda",
    })
    monkeypatch.setattr(services, "monte_carlo_cuda_backend", lambda: ("cupy", ""))

    hardware = services.stat_report_hardware("cuda", "poisson_sota", "monte_carlo")

    assert hardware["actual_device"] == "cuda"
    assert hardware["backend_supports_cuda"] is True
    assert hardware["monte_carlo_backend"] == "cupy"


def test_fixture_report_warnings_groups_duplicate_model_warnings():
    from src.web import mundial_services as services

    report = {
        "models": [
            {"model_label": "A", "warnings": ["warning compartido"]},
            {"model_label": "B", "warnings": ["warning compartido"]},
            {"model_label": "C", "warnings": ["warning unico"]},
        ]
    }

    warnings = services.fixture_report_warnings(report)

    assert "warning compartido (2 modelos)" in warnings
    assert "C: warning unico" in warnings


def test_sota_report_explicit_cuda_without_gpu_returns_clear_device_error(monkeypatch):
    from src.web import mundial_services as services

    monkeypatch.setattr(services, "detect_hardware", lambda: {
        "cpu_count": 8,
        "default_n_jobs": -1,
        "cuda_available": False,
        "cuda_devices": [],
        "cuda_device_names": [],
        "cuda_detection_source": "none",
        "cuda_detection_sources": [],
        "cuda_error": "nvidia-smi no disponible",
        "cuda_warning": "nvidia-smi no disponible",
        "device_default": "cpu",
    })

    hardware = services.stat_report_hardware("cuda", "poisson_sota")

    assert hardware["actual_device"] == "cpu"
    assert "CUDA fue solicitada explicitamente" in hardware["device_error"]
    assert hardware["device_error"] in hardware["warnings"]
