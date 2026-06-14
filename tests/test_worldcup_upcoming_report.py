import json

import numpy as np
import pandas as pd
import pytest


def test_sota_and_alternative_sequences_are_statistical_score_models():
    from src.web import mundial_services as services
    from src.worldcup.score_models import score_model_options

    disabled = {"bayesian_hierarchical_poisson", "bayesian_dynamic_poisson"}
    catalog_keys = {option["key"] for option in score_model_options()}

    assert disabled.isdisjoint(services.SOTA_SCORE_MODEL_SEQUENCE)
    assert disabled <= catalog_keys
    assert "xg_poisson_local" not in services.SOTA_SCORE_MODEL_SEQUENCE
    assert len(services.SOTA_SCORE_MODEL_SEQUENCE) == 7
    assert services.ALTERNATIVE_SCORE_MODEL_SEQUENCE == [
        "dixon_coles_mle",
        "bivariate_poisson_mle",
        "diagonal_inflated_bivariate_poisson",
        "zero_inflated_generalized_poisson",
        "skellam_margin",
        "copula_weibull_count",
    ]
    assert services.BENCHMARK_SCORE_MODEL_SEQUENCE == services.SOTA_SCORE_MODEL_SEQUENCE
    assert "independent_poisson" not in services.ALTERNATIVE_SCORE_MODEL_SEQUENCE


def test_alternatives_benchmark_aliases_and_statistical_registry():
    from src.web import mundial_services as services
    from src.worldcup.sota_alternatives import sota_alternatives_catalog

    assert services.normalize_report_pipeline_mode("benchmark_alternativas") == "alternatives_benchmark"
    assert services.normalize_report_pipeline_mode("sota_alternatives") == "alternatives_benchmark"
    assert services.normalize_report_pipeline_mode("modelos mejores") == "alternatives_benchmark"
    assert services.normalize_report_pipeline_mode("poisson_sota") == "poisson_sota"
    assert services.normalize_report_pipeline_mode("modo_desconocido") == "poisson_sota"
    assert services.SOTA_SCORE_MODEL_SEQUENCE == [
        "independent_poisson",
        "dixon_coles_mle",
        "bivariate_poisson_mle",
        "diagonal_inflated_bivariate_poisson",
        "zero_inflated_generalized_poisson",
        "skellam_margin",
        "copula_weibull_count",
    ]
    alternatives = sota_alternatives_catalog()
    assert [item["key"] for item in alternatives] == services.ALTERNATIVE_SCORE_MODEL_SEQUENCE
    forbidden = {"catboost", "xgboost", "lightgbm", "random forest", "mlp", "machine learning"}
    registry_text = json.dumps(alternatives).lower()
    assert not any(term in registry_text for term in forbidden)
    assert all(item["model_name"] and item["description"] for item in alternatives)


def test_alternatives_benchmark_default_backtest_and_ranking_policy():
    from src.web import mundial_services as services

    config = services.report_pipeline_config({}, services.ALTERNATIVES_BENCHMARK_PIPELINE_MODE)
    assert config["backtest_last_n"] == 7
    assert config["backtest_scope"] == "worldcup_2026_confirmed_auto"

    ranked = services.rank_backtest_models([
        {
            "model_key": "a",
            "model_label": "A",
            "available": True,
            "evaluated_matches": 7,
            "log_loss": 1.0,
            "brier": 0.22,
            "pick_accuracy": 0.4,
            "score_accuracy": 0.1,
            "score_log_loss": 3.0,
        },
        {
            "model_key": "b",
            "model_label": "B",
            "available": True,
            "evaluated_matches": 7,
            "log_loss": 1.0,
            "brier": 0.30,
            "pick_accuracy": 0.3,
            "score_accuracy": 0.2,
            "score_log_loss": 4.0,
        },
        {
            "model_key": "c",
            "model_label": "C",
            "available": True,
            "evaluated_matches": 7,
            "log_loss": 0.9,
            "brier": 0.40,
            "pick_accuracy": 0.2,
            "score_accuracy": 0.0,
            "score_log_loss": 5.0,
        },
    ], {"holdout_start": "2022-12-09", "holdout_end": "2022-12-18"})

    assert [item["model_key"] for item in ranked] == ["c", "b", "a"]
    assert [item["rank"] for item in ranked] == [1, 2, 3]
    assert ranked[0]["reliability_score"] > ranked[1]["reliability_score"] > ranked[2]["reliability_score"]
    assert all(item["ranking_metric"] == "log_loss" for item in ranked)
    assert all(item["holdout_start"] == "2022-12-09" for item in ranked)


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
    monkeypatch.setattr(services, "refresh_worldcup_2026_results", lambda tournament, refresh=False: {"source": "test-results", "warnings": []})
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
    assert result["summary"]["backtest_scope"] == "worldcup_2026_confirmed_auto"
    assert result["summary"]["backtest_source"] == "test-results"
    assert len(result["summary"]["backtest_confirmed_matches"]) == 3
    assert "posteriores" in result["summary"]["anti_leakage"]
    assert len(result["fixture_reports"]) == 1
    fixture_report = result["fixture_reports"][0]
    assert fixture_report["fixture"]["kickoff_iso"] == "2026-06-20T18:00:00+00:00"
    assert fixture_report["fixture"]["countdown_state"] == "ready"
    assert [model["model_key"] for model in fixture_report["models"]] == result["summary"]["score_models"]
    assert fixture_report["baseline_poisson"]["model_key"] == "independent_poisson"
    assert "consensus" not in fixture_report
    assert "consensus_score_distribution" not in fixture_report
    assert "consensus_eligible" not in fixture_report["baseline_poisson"]
    assert all("consensus_eligible" not in model and "signature" not in model for model in fixture_report["models"])
    assert len(result["model_backtests"]) == len(services.BENCHMARK_SCORE_MODEL_SEQUENCE)
    assert [item["rank"] for item in result["model_backtests"]] == list(range(1, len(services.BENCHMARK_SCORE_MODEL_SEQUENCE) + 1))
    assert result["best_model"]["model_key"] == result["model_backtests"][0]["model_key"]
    assert result["best_model"]["model_key"] in services.BENCHMARK_SCORE_MODEL_SEQUENCE
    assert result["table"]["total"] == len(services.BENCHMARK_SCORE_MODEL_SEQUENCE)
    assert (tmp_path / "latest.json").exists()
    assert progress[-1]["stage"] == "complete"


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
    assert result["models"][0]["rank"] == 1


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

    monkeypatch.setattr(services, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: ({}, "fixture-test"))
    monkeypatch.setattr(services, "build_model", lambda tournament, config: (FakeModel(), "history-test"))
    monkeypatch.setattr(services, "upcoming_fixture_rows", lambda tournament, group_filter="": fixtures)
    monkeypatch.setattr(services, "groups_from_tournament", lambda tournament: {"Group A": ["Mexico", "Canada"]})
    monkeypatch.setattr(services, "load_historical_matches", lambda refresh=False: (pd.DataFrame(), "history-test"))
    monkeypatch.setattr(services, "contextual_poisson_for_match", lambda *args, **kwargs: {})

    def fake_build_score_model(base_model, history_df, teams, config):
        key = config["score_model"]
        fit_order.append(key)
        return FakeModel(key=key, available=True)

    monkeypatch.setattr(services, "build_score_model", fake_build_score_model)
    progress = []

    result = services.predict_upcoming_report(
        {"pipeline_mode": "poisson_sota", "limit": 1},
        progress_callback=progress.append,
    )

    assert prediction_order == services.SOTA_SCORE_MODEL_SEQUENCE
    assert fit_order == services.SOTA_SCORE_MODEL_SEQUENCE[1:]
    assert result["summary"]["sota_calculation_mode"] == "exact"
    assert "use_ml_model" not in result["summary"]
    assert result["fixture_reports"][0]["models"][0]["model_key"] == "independent_poisson"
    assert "monte_carlo_consensus" not in result["fixture_reports"][0]
    assert result["fixture_reports"][0]["consensus"]["eligible_models"] == len(services.SOTA_SCORE_MODEL_SEQUENCE)
    assert len(result["fixture_reports"][0]["top_models_1x2"]) == 4
    assert result["fixture_reports"][0]["consensus_score_distribution"]["available"] is True
    assert result["fixture_reports"][0]["model_statistics"]["model_count"] == len(services.SOTA_SCORE_MODEL_SEQUENCE)
    assert result["fixture_reports"][0]["models"][0]["score_distribution"]["top_scores"]
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

    def fake_count_matrix(grids, weights, iterations, seed, backend="numpy"):
        captured["iterations"] = iterations
        captured["backend"] = backend
        captured["grid_count"] = len(grids)
        captured["weights"] = list(weights)
        return np.array([[35000, 15000], [40000, 10000]]), "numpy", [iterations]

    monkeypatch.setattr(services, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(services, "SOTA_SCORE_MODEL_SEQUENCE", ["independent_poisson"])
    monkeypatch.setattr(services, "load_tournament_2026", lambda refresh=False: ({}, "fixture-test"))
    monkeypatch.setattr(services, "build_model", lambda tournament, config: (FakeModel(), "history-test"))
    monkeypatch.setattr(services, "upcoming_fixture_rows", lambda tournament, group_filter="": fixtures)
    monkeypatch.setattr(services, "groups_from_tournament", lambda tournament: {"Group A": ["Mexico", "Canada"]})
    monkeypatch.setattr(services, "load_historical_matches", lambda refresh=False: (pd.DataFrame(), "history-test"))
    monkeypatch.setattr(services, "contextual_poisson_for_match", lambda *args, **kwargs: {})
    monkeypatch.setattr(services, "monte_carlo_count_matrix_from_model_grids", fake_count_matrix)
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
    assert captured["grid_count"] == 1
    assert monte_carlo["available"] is True
    assert monte_carlo["iterations"] == 100000
    assert monte_carlo["source"] == "SOTA Monte Carlo por mezcla de modelos"
    assert monte_carlo["exact_consensus"]["available"] is True
    assert monte_carlo["model_count"] == 1
    assert monte_carlo["model_sample_counts"][0]["sample_count"] == 100000
    assert set(monte_carlo["probability_deltas"]) >= {"home", "draw", "away", "over25", "under25"}
    assert set(monte_carlo["probabilities"]) >= {"home", "draw", "away", "over05", "under05", "over25", "under25"}
    assert monte_carlo["top_scores"]


def test_monte_carlo_model_entries_exclude_unavailable_and_fallback_models():
    from src.web import mundial_services as services

    def report(key, eligible=True, fallback=False):
        return {
            "model_key": key,
            "model_label": key,
            "consensus_eligible": eligible,
            "fallback": fallback,
            "score_distribution": {
                "score_matrix": [[25.0, 25.0], [25.0, 25.0]],
                "lambdas": {"home": 1.0, "away": 1.0},
            },
        }

    entries = services.monte_carlo_model_grid_entries([
        report("independent_poisson"),
        report("xg_poisson_local", eligible=False),
        report("dixon_coles_mle", fallback=True),
    ])

    assert [entry["model_key"] for entry in entries] == ["independent_poisson"]


def test_monte_carlo_single_model_matches_that_model_distribution():
    from src.web import mundial_services as services

    grid = np.array([[0.20, 0.30], [0.10, 0.40]])
    counts, backend, sample_counts = services.monte_carlo_count_matrix_from_model_grids(
        grids=[grid],
        weights=[1.0],
        iterations=50000,
        seed=17,
        backend="numpy",
    )

    exact = services.score_grid_probabilities(grid)
    simulated = services.score_grid_probabilities(counts)

    assert backend == "numpy"
    assert sample_counts == [50000]
    for key in ("home", "draw", "away", "over05", "under05"):
        assert abs(simulated[key] - exact[key]) < 1.5


def test_monte_carlo_two_distinct_models_approximates_uniform_mixture():
    from src.web import mundial_services as services

    home_win_grid = np.array([[0.0, 0.0], [1.0, 0.0]])
    away_win_grid = np.array([[0.0, 1.0], [0.0, 0.0]])

    counts, backend, sample_counts = services.monte_carlo_count_matrix_from_model_grids(
        grids=[home_win_grid, away_win_grid],
        weights=[0.5, 0.5],
        iterations=40000,
        seed=2026,
        backend="numpy",
    )
    simulated = services.score_grid_probabilities(counts)

    assert backend == "numpy"
    assert abs(sample_counts[0] - sample_counts[1]) < 900
    assert 48.0 <= simulated["home"] <= 52.0
    assert 48.0 <= simulated["away"] <= 52.0
    assert simulated["draw"] == 0.0


def test_sota_model_weights_use_walk_forward_metrics_and_penalize_experimental(monkeypatch):
    from src.web import mundial_services as services

    def entry(key):
        return {
            "model_key": key,
            "model_label": key,
            "grid": np.ones((2, 2), dtype=float) / 4.0,
            "model_report": {},
        }

    monkeypatch.setattr(services, "load_sota_performance_metrics", lambda: ({
        "independent_poisson": {"sample_size": 120, "brier": 0.40},
        "dixon_coles_mle": {"sample_size": 120, "brier": 0.80},
    }, "storage/worldcup/walk_forward/sota_model_metrics.json"))

    payload = services.sota_model_weighting_payload([
        entry("independent_poisson"),
        entry("dixon_coles_mle"),
    ])
    weights = {item["model_key"]: item["weight"] for item in payload["items"]}

    assert payload["source"] == "walk_forward_metrics"
    assert weights["independent_poisson"] > weights["dixon_coles_mle"]
    assert payload["metrics_path"].endswith("sota_model_metrics.json")

    monkeypatch.setattr(services, "load_sota_performance_metrics", lambda: ({}, ""))
    payload = services.sota_model_weighting_payload([
        entry("independent_poisson"),
        entry("copula_weibull_count"),
    ])
    weights = {item["model_key"]: item["weight"] for item in payload["items"]}
    reasons = {item["model_key"]: item["reasons"] for item in payload["items"]}

    assert payload["source"] == "uniform_with_experimental_penalty"
    assert weights["copula_weibull_count"] < weights["independent_poisson"]
    assert any("penalizacion experimental" in reason for reason in reasons["copula_weibull_count"])


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


def test_sota_report_honors_explicit_cuda_when_detected_and_warns_cpu_bound(monkeypatch):
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

    hardware = services.stat_report_hardware("cuda", "poisson_sota")
    assert hardware["actual_device"] == "cuda"
    assert hardware["backend_supports_cuda"] is False
    assert sum("CPU-bound" in warning for warning in hardware["warnings"]) == 1


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
