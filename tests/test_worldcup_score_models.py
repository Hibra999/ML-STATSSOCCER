import numpy as np
import pandas as pd

from src.worldcup.model import WorldCupModel
from src.worldcup.score_models import (
    ScoreModelState,
    build_score_model,
    score_grid_from_lambdas,
)


def test_score_model_grids_are_normalized_and_non_negative():
    states = [
        ScoreModelState("dixon_coles_mle", "Dixon-Coles", True, {"rho": -0.08}),
        ScoreModelState("bivariate_poisson_mle", "Bivariado", True, {"corr_share": 0.12}),
        ScoreModelState("diagonal_inflated_bivariate_poisson", "Diagonal", True, {"corr_share": 0.08, "diagonal_boost": 1.8}),
        ScoreModelState("zero_inflated_generalized_poisson", "ZIGP", True, {"alpha_home": 0.08, "alpha_away": 0.05, "zero_home": 0.06, "zero_away": 0.04}),
        ScoreModelState("skellam_margin", "Skellam", True, {"margin_scale": 1.15}),
        ScoreModelState("copula_weibull_count", "Copula", True, {"beta": 1.12, "theta": 0.6}),
    ]

    for state in states:
        grid = score_grid_from_lambdas(state, lambda1=1.42, lambda2=1.08, max_goals=8)

        assert grid.shape == (9, 9)
        assert np.isclose(grid.sum(), 1.0)
        assert np.all(grid >= 0.0)


def test_diagonal_inflation_increases_draw_mass():
    base = ScoreModelState("bivariate_poisson_mle", "Bivariado", True, {"corr_share": 0.05})
    inflated = ScoreModelState("diagonal_inflated_bivariate_poisson", "Diagonal", True, {"corr_share": 0.05, "diagonal_boost": 2.0})

    base_draw = np.trace(score_grid_from_lambdas(base, 1.2, 1.2, max_goals=8))
    inflated_draw = np.trace(score_grid_from_lambdas(inflated, 1.2, 1.2, max_goals=8))

    assert inflated_draw > base_draw


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
        "score_model": "diagonal-inflated-bivariate-poisson",
        "bayes_draws": 50,
    })

    assert config["score_model"] == "diagonal_inflated_bivariate_poisson"
    assert config["bayes_draws"] == 100
