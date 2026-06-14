import numpy as np
import pandas as pd

from src.worldcup.model import WorldCupModel
from src.worldcup.score_models import (
    ScoreModelState,
    build_score_model,
    normalize_score_model_key,
    score_grid_from_lambdas,
)


def test_score_model_grids_are_normalized_and_non_negative():
    states = [
        ScoreModelState("dixon_coles_mle", "Dixon-Coles", True, {"rho": -0.08}),
        ScoreModelState("bivariate_poisson_mle", "Bivariado", True, {"corr_share": 0.12}),
    ]

    for state in states:
        grid = score_grid_from_lambdas(state, lambda1=1.42, lambda2=1.08, max_goals=8)

        assert grid.shape == (9, 9)
        assert np.isclose(grid.sum(), 1.0)
        assert np.all(grid >= 0.0)


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
