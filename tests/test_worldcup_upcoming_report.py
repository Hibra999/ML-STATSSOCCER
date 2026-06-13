import json

import numpy as np
import pandas as pd


def test_sota_sequence_temporarily_excludes_bayes_models_but_catalog_keeps_them():
    from src.web import mundial_services as services
    from src.worldcup.score_models import score_model_options

    disabled = {"bayesian_hierarchical_poisson", "bayesian_dynamic_poisson"}
    catalog_keys = {option["key"] for option in score_model_options()}

    assert disabled.isdisjoint(services.SOTA_SCORE_MODEL_SEQUENCE)
    assert disabled <= catalog_keys
    assert len(services.SOTA_SCORE_MODEL_SEQUENCE) == 8


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
    assert any(item.get("model_key") == "xg_poisson_local" for item in progress)
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
    assert monte_carlo["available"] is True
    assert monte_carlo["iterations"] == 100000
    assert set(monte_carlo["probabilities"]) >= {"home", "draw", "away", "over05", "under05", "over25", "under25"}
    assert monte_carlo["top_scores"]


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
