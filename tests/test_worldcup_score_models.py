import numpy as np
import pandas as pd

from src.worldcup.model import WorldCupModel
from src.worldcup.score_models import (
    DYNAMIC_STRENGTH_KALMAN_MODEL,
    NEGATIVE_BINOMIAL_DIXON_COLES_MODEL,
    ScoreModelState,
    STACKED_META_MNLOGIT_MODEL,
    XG_DIXON_COLES_MODEL,
    build_score_model,
    fit_score_model_state,
    normalize_score_model_key,
    score_grid_from_lambdas,
    score_grids_from_lambdas,
    score_grids_from_lambdas_with_backend,
)


def test_score_model_grids_are_normalized_and_non_negative():
    states = [
        ScoreModelState("dixon_coles_mle", "Dixon-Coles", True, {"rho": -0.08}),
        ScoreModelState("statsmodels_poisson_glm", "Poisson GLM", True, {}),
        ScoreModelState("negative_binomial_glm", "Negative Binomial GLM", True, {"alpha": 0.2}),
        ScoreModelState(NEGATIVE_BINOMIAL_DIXON_COLES_MODEL, "NB DC", True, {"alpha": 0.2, "rho": -0.05}),
        ScoreModelState(XG_DIXON_COLES_MODEL, "xG DC", True, {"rho": -0.05}),
        ScoreModelState(DYNAMIC_STRENGTH_KALMAN_MODEL, "Dynamic", True, {}),
        ScoreModelState(STACKED_META_MNLOGIT_MODEL, "Stacking", True, {}),
        ScoreModelState("bivariate_poisson_mle", "Bivariado", True, {"corr_share": 0.12}),
    ]

    for state in states:
        grid = score_grid_from_lambdas(state, lambda1=1.42, lambda2=1.08, max_goals=8)

        assert grid.shape == (9, 9)
        assert np.isclose(grid.sum(), 1.0)
        assert np.all(grid >= 0.0)


def test_worldcup_model_from_history_handles_mixed_date_types_safely():
    history = pd.DataFrame([
        {"Date": pd.Timestamp("2026-06-10"), "Team 1": "Mexico", "Team 2": "Canada", "G1": 2, "G2": 1},
        {"Date": "2026-06-08", "Team 1": "Canada", "Team 2": "USA", "G1": 0, "G2": 1},
        {"Date": pd.to_datetime("2026-06-12"), "Team 1": "USA", "Team 2": "Mexico", "G1": 1, "G2": 3},
    ])
    model = WorldCupModel.from_history(history, teams=["Mexico", "Canada", "USA"])

    lambda1, lambda2 = model.expected_goals("Mexico", "Canada")
    assert lambda1 > 0.0
    assert lambda2 > 0.0


def test_batched_score_model_grids_match_scalar_numpy_backend():
    state = ScoreModelState("dixon_coles_mle", "Dixon-Coles", True, {"rho": -0.08})
    lambdas_home = np.asarray([1.42, 0.95, 2.1], dtype=float)
    lambdas_away = np.asarray([1.08, 1.2, 0.7], dtype=float)

    batched = score_grids_from_lambdas(
        state,
        lambdas_home,
        lambdas_away,
        max_goals=8,
        backend="numpy",
    )

    assert batched.shape == (3, 9, 9)
    for index in range(3):
        scalar = score_grid_from_lambdas(state, lambdas_home[index], lambdas_away[index], max_goals=8)
        assert np.allclose(batched[index], scalar)


def test_batched_score_model_default_requests_cuda(monkeypatch):
    from src.worldcup import score_models

    captured = []

    def fake_score_backend_status(requested_device=score_models.GPU_FIRST_BACKEND):
        captured.append(requested_device)
        return {"score_backend": "numpy", "warning": "CuPy/CUDA no disponible"}

    monkeypatch.setattr(score_models, "score_backend_status", fake_score_backend_status)
    state = ScoreModelState("dixon_coles_mle", "Dixon-Coles", True, {"rho": 0.0})

    grids, backend, warnings = score_grids_from_lambdas_with_backend(
        state,
        np.asarray([1.2, 1.4], dtype=float),
        np.asarray([0.9, 1.1], dtype=float),
        max_goals=6,
    )

    assert captured == ["cuda"]
    assert backend == "numpy"
    assert warnings == ["CuPy/CUDA no disponible"]
    assert grids.shape == (2, 7, 7)


def test_removed_score_model_keys_fall_back_to_independent_poisson():
    removed = [
        "diagonal-inflated-bivariate-poisson",
        "zero_inflated_generalized_poisson",
        "negative_binomial_mle",
        "conway_maxwell_poisson",
        "skellam_margin",
        "copula_weibull_count",
    ]

    assert all(normalize_score_model_key(key) == "independent_poisson" for key in removed)


def test_statsmodels_poisson_glm_fits_regularized_lambda_model():
    history = pd.DataFrame([
        {"Date": f"2020-{(index % 9) + 1:02d}-01", "Team 1": ["Mexico", "Canada", "USA"][index % 3], "Team 2": ["Canada", "USA", "Mexico"][index % 3], "G1": (index * 2) % 4, "G2": (index + 1) % 3}
        for index in range(24)
    ])
    teams = ["Mexico", "Canada", "USA"]
    base = WorldCupModel.from_history(history, teams=teams)

    model = build_score_model(
        base,
        history_df=history,
        teams=teams,
        config={"score_model": "statsmodels_poisson_glm", "stat_model_cache": False, "stat_glm_min_matches": 4},
    )
    metadata = model.score_model_metadata()
    lambda_model = metadata["params"]["lambda_model"]
    probabilities = model.match_probabilities("Mexico", "Canada")

    assert metadata["available"] is True
    assert lambda_model["type"] == "statsmodels_poisson_glm"
    assert lambda_model["validation"]["source"] == "temporal_holdout_pretest"
    assert lambda_model["diagnostics"]["pearson_dispersion"] >= 0.0
    assert probabilities["score_model"] == "statsmodels_poisson_glm"
    assert probabilities["lambda1"] > 0.0
    assert probabilities["lambda2"] > 0.0


def test_dixon_coles_uses_statsmodels_lambda_model_before_rho_mle():
    history = pd.DataFrame([
        {"Date": f"2021-{(index % 9) + 1:02d}-01", "Team 1": ["Mexico", "Canada", "USA"][index % 3], "Team 2": ["Canada", "USA", "Mexico"][index % 3], "G1": (index + 2) % 4, "G2": index % 3}
        for index in range(24)
    ])
    teams = ["Mexico", "Canada", "USA"]
    base = WorldCupModel.from_history(history, teams=teams)

    state = fit_score_model_state(
        "dixon_coles_mle",
        base_model=base,
        history_df=history,
        teams=teams,
        config={"stat_model_cache": False, "stat_glm_min_matches": 4},
    )

    assert state.available is True
    assert "rho" in state.params
    assert state.params["lambda_model"]["type"] == "statsmodels_poisson_glm"


def test_negative_binomial_glm_uses_overdispersion_alpha():
    history = pd.DataFrame([
        {"Date": f"2022-{(index % 9) + 1:02d}-01", "Team 1": ["Mexico", "Canada", "USA"][index % 3], "Team 2": ["Canada", "USA", "Mexico"][index % 3], "G1": [0, 1, 5, 0][index % 4], "G2": [0, 0, 4][index % 3]}
        for index in range(30)
    ])
    teams = ["Mexico", "Canada", "USA"]
    base = WorldCupModel.from_history(history, teams=teams)

    model = build_score_model(
        base,
        history_df=history,
        teams=teams,
        config={"score_model": "negative_binomial_glm", "stat_model_cache": False, "stat_glm_min_matches": 4},
    )
    metadata = model.score_model_metadata()
    grid = model.score_grid("Mexico", "Canada", max_goals=8)

    assert metadata["available"] is True
    assert metadata["params"]["lambda_model"]["type"] == "statsmodels_poisson_glm"
    assert metadata["params"]["alpha"] >= 0.0
    assert np.isclose(grid.sum(), 1.0)
    assert np.all(grid >= 0.0)


def test_negative_binomial_dixon_coles_fits_alpha_and_rho():
    history = pd.DataFrame([
        {"Date": f"2023-{(index % 9) + 1:02d}-01", "Team 1": ["Mexico", "Canada", "USA"][index % 3], "Team 2": ["Canada", "USA", "Mexico"][index % 3], "G1": [0, 1, 5, 0][index % 4], "G2": [0, 0, 4][index % 3]}
        for index in range(30)
    ])
    teams = ["Mexico", "Canada", "USA"]
    base = WorldCupModel.from_history(history, teams=teams)

    state = fit_score_model_state(
        NEGATIVE_BINOMIAL_DIXON_COLES_MODEL,
        base_model=base,
        history_df=history,
        teams=teams,
        config={"stat_model_cache": False, "stat_glm_min_matches": 4},
    )
    grid = score_grid_from_lambdas(state, 1.25, 1.05, max_goals=8)

    assert state.available is True
    assert state.params["alpha"] >= 0.0
    assert -0.24 <= state.params["rho"] <= 0.24
    assert np.isclose(grid.sum(), 1.0)


def test_dynamic_strength_kalman_returns_finite_lambdas():
    history = pd.DataFrame([
        {"Date": f"2024-{(index % 9) + 1:02d}-01", "Team 1": ["Mexico", "Canada", "USA"][index % 3], "Team 2": ["Canada", "USA", "Mexico"][index % 3], "G1": (index + 1) % 4, "G2": index % 3}
        for index in range(18)
    ])
    teams = ["Mexico", "Canada", "USA"]
    base = WorldCupModel.from_history(history, teams=teams)

    model = build_score_model(
        base,
        history_df=history,
        teams=teams,
        config={"score_model": DYNAMIC_STRENGTH_KALMAN_MODEL, "stat_model_cache": False},
    )
    probabilities = model.match_probabilities("Mexico", "Canada")

    assert model.score_model_metadata()["available"] is True
    assert probabilities["lambda1"] > 0.0
    assert probabilities["lambda2"] > 0.0
    assert probabilities["score_model"] == DYNAMIC_STRENGTH_KALMAN_MODEL


def test_xg_dixon_coles_uses_local_xg_file(tmp_path, monkeypatch):
    from src.worldcup import score_models

    xg_file = tmp_path / "manual_xg.csv"
    xg_file.write_text("date,home,away,home_xg,away_xg\n2026-06-11,Mexico,Canada,1.8,0.7\n", encoding="utf-8")
    monkeypatch.setattr(score_models, "LOCAL_XG_FILE", xg_file)
    history = pd.DataFrame([
        {"Date": f"2021-{(index % 9) + 1:02d}-01", "Team 1": ["Mexico", "Canada", "USA"][index % 3], "Team 2": ["Canada", "USA", "Mexico"][index % 3], "G1": (index + 1) % 3, "G2": index % 2}
        for index in range(12)
    ])
    base = WorldCupModel.from_history(history, teams=["Mexico", "Canada", "USA"])

    model = build_score_model(
        base,
        history_df=history,
        teams=["Mexico", "Canada", "USA"],
        config={"score_model": XG_DIXON_COLES_MODEL, "stat_model_cache": False},
    )
    probabilities = model.match_probabilities_for_match(
        "Mexico",
        "Canada",
        match={"date": "2026-06-11"},
    )

    assert model.score_model_metadata()["available"] is True
    assert probabilities["score_model"] == XG_DIXON_COLES_MODEL
    assert probabilities["lambda1"] == 1.8
    assert probabilities["lambda2"] == 0.7


def test_stacked_meta_mnlogit_changes_outcome_probabilities_when_fit():
    outcomes = [(2, 0), (1, 1), (0, 2), (3, 1), (0, 0), (1, 2)]
    history = pd.DataFrame([
        {"Date": f"2020-{(index % 9) + 1:02d}-01", "Team 1": ["Mexico", "Canada", "USA"][index % 3], "Team 2": ["Canada", "USA", "Mexico"][index % 3], "G1": outcomes[index % len(outcomes)][0], "G2": outcomes[index % len(outcomes)][1]}
        for index in range(36)
    ])
    teams = ["Mexico", "Canada", "USA"]
    base = WorldCupModel.from_history(history, teams=teams)

    model = build_score_model(
        base,
        history_df=history,
        teams=teams,
        config={"score_model": STACKED_META_MNLOGIT_MODEL, "stat_model_cache": False, "stat_glm_min_matches": 4},
    )
    probabilities = model.match_probabilities("Mexico", "Canada")

    assert model.score_model_metadata()["key"] == STACKED_META_MNLOGIT_MODEL
    assert probabilities["score_model"] == STACKED_META_MNLOGIT_MODEL
    assert abs(probabilities["home"] + probabilities["draw"] + probabilities["away"] - 1.0) < 1e-9


def test_build_score_model_wraps_worldcup_model_with_metadata():
    history = pd.DataFrame([
        {"Date": "2018-06-01", "Team 1": "Mexico", "Team 2": "Canada", "G1": 1, "G2": 1},
        {"Date": "2019-06-01", "Team 1": "Mexico", "Team 2": "USA", "G1": 2, "G2": 0},
        {"Date": "2020-06-01", "Team 1": "Canada", "Team 2": "USA", "G1": 0, "G2": 0},
        {"Date": "2021-06-01", "Team 1": "USA", "Team 2": "Mexico", "G1": 1, "G2": 1},
    ])
    base = WorldCupModel.from_history(history, teams=["Mexico", "Canada", "USA"])

    model = build_score_model(
        base,
        history_df=history,
        teams=["Mexico", "Canada", "USA"],
        config={"score_model": "dixon_coles_mle", "stat_model_cache": False},
    )
    probabilities = model.match_probabilities("Mexico", "Canada")

    assert model.score_model_metadata()["key"] == "dixon_coles_mle"
    assert probabilities["score_model"] == "dixon_coles_mle"
    assert set(probabilities) >= {"home", "draw", "away", "over25", "under25"}


def test_simulation_config_accepts_score_model_key():
    from src.web import mundial_services

    config = mundial_services.simulation_config({
        "score_model": "dixon-coles-mle",
        "bayes_draws": 50,
    })

    assert config["score_model"] == "dixon_coles_mle"
    assert config["bayes_draws"] == 100


def test_pytensor_cxx_guard_appends_empty_cxx_only_without_compiler(monkeypatch):
    from src.worldcup import score_models

    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("PYTENSOR_FLAGS", raising=False)
    monkeypatch.setattr(score_models.shutil, "which", lambda command: None)

    assert score_models._ensure_pytensor_cxx_flag() is True
    assert score_models.os.environ["PYTENSOR_FLAGS"] == "cxx="

    monkeypatch.setenv("PYTENSOR_FLAGS", "floatX=float32")
    assert score_models._ensure_pytensor_cxx_flag() is True
    assert score_models.os.environ["PYTENSOR_FLAGS"] == "floatX=float32,cxx="

    monkeypatch.setenv("PYTENSOR_FLAGS", "floatX=float32,cxx=/custom/compiler")
    assert score_models._ensure_pytensor_cxx_flag() is False
    assert score_models.os.environ["PYTENSOR_FLAGS"] == "floatX=float32,cxx=/custom/compiler"

    monkeypatch.setenv("PYTENSOR_FLAGS", "floatX=float64")
    monkeypatch.setattr(score_models.shutil, "which", lambda command: "/usr/bin/g++" if command == "g++" else None)
    assert score_models._ensure_pytensor_cxx_flag() is False
    assert score_models.os.environ["PYTENSOR_FLAGS"] == "floatX=float64"

    monkeypatch.setattr(score_models.shutil, "which", lambda command: None)
    monkeypatch.setenv("CXX", "custom-cxx")
    assert score_models._ensure_pytensor_cxx_flag() is False
    assert score_models.os.environ["PYTENSOR_FLAGS"] == "floatX=float64"
