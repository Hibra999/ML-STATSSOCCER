from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


TEAM_RATING_PRIORS = {
    "Argentina": 1850,
    "France": 1830,
    "Brazil": 1800,
    "England": 1800,
    "Spain": 1800,
    "Portugal": 1780,
    "Netherlands": 1760,
    "Germany": 1750,
    "Croatia": 1700,
    "Belgium": 1700,
    "Uruguay": 1680,
    "Morocco": 1670,
    "Colombia": 1670,
    "Japan": 1640,
    "Austria": 1640,
    "USA": 1630,
    "Mexico": 1620,
    "Switzerland": 1620,
    "Senegal": 1600,
    "Turkey": 1580,
    "Sweden": 1580,
    "Norway": 1580,
    "Ecuador": 1580,
    "Czech Republic": 1580,
    "Ivory Coast": 1560,
    "Algeria": 1560,
    "South Korea": 1560,
    "Iran": 1550,
    "Egypt": 1540,
    "Australia": 1530,
    "Paraguay": 1530,
    "Scotland": 1520,
    "Tunisia": 1510,
    "Ghana": 1510,
    "Canada": 1510,
    "Bosnia & Herzegovina": 1500,
    "DR Congo": 1500,
    "Uzbekistan": 1500,
    "Cape Verde": 1500,
    "South Africa": 1490,
    "Saudi Arabia": 1460,
    "Panama": 1450,
    "Qatar": 1450,
    "Iraq": 1450,
    "New Zealand": 1430,
    "Jordan": 1420,
    "Curacao": 1420,
    "Curaçao": 1420,
    "Haiti": 1400,
}

HOST_TEAMS = {"Mexico", "USA", "Canada"}
TOTAL_GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)


@dataclass(frozen=True)
class TeamProfile:
    team: str
    rating: float
    matches: int
    gf_per_match: float
    ga_per_match: float
    attack: float
    defense: float


class WorldCupModel:
    def __init__(
            self,
            profiles: Dict[str, TeamProfile],
            global_g1: float = 1.25,
            global_g2: float = 1.05,
            host_advantage: float = 45.0,
            max_goals: int = 10,
    ):
        self._profiles = profiles
        self.global_g1 = float(global_g1 or 1.25)
        self.global_g2 = float(global_g2 or 1.05)
        self.host_advantage = _clamp(host_advantage, 0.0, 120.0)
        self.max_goals = int(_clamp(max_goals, 6, 14))

    @classmethod
    def from_history(
            cls,
            historical_df: pd.DataFrame,
            teams: Iterable[str],
            history_weight: float = 1.0,
            recency_weight: float = 0.0,
            host_advantage: float = 45.0,
            max_goals: int = 10,
    ) -> "WorldCupModel":
        team_list = sorted(set(teams))
        ratings = {team: float(TEAM_RATING_PRIORS.get(team, 1500)) for team in team_list}
        stats = {team: {"matches": 0, "weight": 0.0, "gf": 0.0, "ga": 0.0} for team in team_list}
        global_g1 = 1.25
        global_g2 = 1.05
        history_weight = _clamp(history_weight, 0.2, 2.0)
        recency_weight = _clamp(recency_weight, 0.0, 1.0)

        if historical_df is not None and not historical_df.empty:
            ordered_history = _ordered_history_for_model(historical_df)
            ordered_history["_weight"] = _match_weights(ordered_history, recency_weight)
            global_g1 = max(float(np.average(ordered_history["G1"], weights=ordered_history["_weight"])), 0.5)
            global_g2 = max(float(np.average(ordered_history["G2"], weights=ordered_history["_weight"])), 0.5)
            for _, row in ordered_history.iterrows():
                team1 = str(row["Team 1"])
                team2 = str(row["Team 2"])
                g1 = int(row["G1"])
                g2 = int(row["G2"])
                match_weight = float(row.get("_weight", 1.0)) * history_weight
                ratings.setdefault(team1, float(TEAM_RATING_PRIORS.get(team1, 1500)))
                ratings.setdefault(team2, float(TEAM_RATING_PRIORS.get(team2, 1500)))
                stats.setdefault(team1, {"matches": 0, "weight": 0.0, "gf": 0.0, "ga": 0.0})
                stats.setdefault(team2, {"matches": 0, "weight": 0.0, "gf": 0.0, "ga": 0.0})
                _update_elo(ratings, team1, team2, g1, g2, k_factor=28.0 * match_weight)
                stats[team1]["matches"] += 1
                stats[team1]["weight"] += match_weight
                stats[team1]["gf"] += g1 * match_weight
                stats[team1]["ga"] += g2 * match_weight
                stats[team2]["matches"] += 1
                stats[team2]["weight"] += match_weight
                stats[team2]["gf"] += g2 * match_weight
                stats[team2]["ga"] += g1 * match_weight

        global_team_goals = max((global_g1 + global_g2) / 2.0, 0.5)
        profiles: Dict[str, TeamProfile] = {}
        for team in team_list:
            team_stats = stats.get(team, {"matches": 0, "weight": 0.0, "gf": 0.0, "ga": 0.0})
            matches = int(team_stats["matches"])
            normalizer = max(float(team_stats.get("weight", 0.0)), 1e-9)
            gf_per_match = float(team_stats["gf"] / normalizer) if matches else global_team_goals
            ga_per_match = float(team_stats["ga"] / normalizer) if matches else global_team_goals
            attack = _blend_toward_neutral(_clamp(gf_per_match / global_team_goals, 0.55, 1.75), history_weight)
            defense = _blend_toward_neutral(_clamp(ga_per_match / global_team_goals, 0.55, 1.75), history_weight)
            profiles[team] = TeamProfile(
                team=team,
                rating=float(ratings.get(team, TEAM_RATING_PRIORS.get(team, 1500))),
                matches=matches,
                gf_per_match=gf_per_match,
                ga_per_match=ga_per_match,
                attack=attack,
                defense=defense,
            )
        return cls(
            profiles=profiles,
            global_g1=global_g1,
            global_g2=global_g2,
            host_advantage=host_advantage,
            max_goals=max_goals,
        )
    def profile(self, team: str) -> TeamProfile:
        if team in self._profiles:
            return self._profiles[team]
        rating = float(TEAM_RATING_PRIORS.get(team, 1500))
        return TeamProfile(team=team, rating=rating, matches=0, gf_per_match=1.15, ga_per_match=1.15, attack=1.0, defense=1.0)

    def adjusted(self, rating_adjustments: Dict[str, float]) -> "WorldCupModel":
        profiles = {
            team: replace(profile, rating=profile.rating + float(rating_adjustments.get(team, 0.0)))
            for team, profile in self._profiles.items()
        }
        return WorldCupModel(
            profiles=profiles,
            global_g1=self.global_g1,
            global_g2=self.global_g2,
            host_advantage=self.host_advantage,
            max_goals=self.max_goals,
        )

    def expected_goals(self, team1: str, team2: str) -> Tuple[float, float]:
        p1 = self.profile(team1)
        p2 = self.profile(team2)
        rating1 = p1.rating + (self.host_advantage if team1 in HOST_TEAMS else 0)
        rating2 = p2.rating + (self.host_advantage if team2 in HOST_TEAMS else 0)
        diff = (rating1 - rating2) / 650.0
        lambda1 = self.global_g1 * math.sqrt(p1.attack * p2.defense) * math.exp(diff)
        lambda2 = self.global_g2 * math.sqrt(p2.attack * p1.defense) * math.exp(-diff)
        return _clamp(lambda1, 0.2, 4.5), _clamp(lambda2, 0.2, 4.5)

    def match_probabilities(self, team1: str, team2: str, max_goals: int | None = None) -> Dict[str, float]:
        max_goals = int(max_goals if max_goals is not None else self.max_goals)
        lambda1, lambda2 = self.expected_goals(team1, team2)
        max_goals = _normalized_max_goals(max_goals)
        grid = poisson_score_grid(lambda1=lambda1, lambda2=lambda2, max_goals=max_goals)
        masks = _score_grid_masks(max_goals)
        modal_index = int(np.argmax(grid))
        modal_score = np.unravel_index(modal_index, grid.shape)
        output = {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "home": float(grid[masks["home"]].sum()),
            "draw": float(grid[masks["draw"]].sum()),
            "away": float(grid[masks["away"]].sum()),
            "modal_g1": int(modal_score[0]),
            "modal_g2": int(modal_score[1]),
        }
        for line in TOTAL_GOAL_LINES:
            over_prob = float(grid[masks["over"][line]].sum())
            suffix = total_line_suffix(line)
            output[f"over{suffix}"] = over_prob
            output[f"under{suffix}"] = 1.0 - output[f"over{suffix}"]
        return output

    def sample_score(self, team1: str, team2: str, rng: np.random.Generator) -> Tuple[int, int]:
        lambda1, lambda2 = self.expected_goals(team1, team2)
        return int(rng.poisson(lambda1)), int(rng.poisson(lambda2))

    def sample_knockout_winner(self, team1: str, team2: str, rng: np.random.Generator) -> Tuple[str, str, int, int]:
        goals1, goals2 = self.sample_score(team1, team2, rng)
        if goals1 > goals2:
            return team1, team2, goals1, goals2
        if goals2 > goals1:
            return team2, team1, goals1, goals2
        probabilities = self.match_probabilities(team1, team2)
        win_share = probabilities["home"] / max(probabilities["home"] + probabilities["away"], 1e-9)
        if rng.random() <= win_share:
            return team1, team2, goals1, goals2
        return team2, team1, goals1, goals2


def _ordered_history_for_model(historical_df: pd.DataFrame) -> pd.DataFrame:
    ordered_history = historical_df.copy()
    if "Date" in ordered_history:
        ordered_history["Date"] = pd.to_datetime(ordered_history["Date"], errors="coerce")
        ordered_history = ordered_history.sort_values("Date", kind="stable").copy()
    return ordered_history


def score_grid_features(lambda1: float, lambda2: float, max_goals: int = 10, score_cap: int = 4) -> Dict[str, float]:
    return dict(_score_grid_features_cached(
        round(float(lambda1), 4),
        round(float(lambda2), 4),
        int(max_goals),
        int(score_cap),
    ))


@lru_cache(maxsize=8192)
def _score_grid_features_cached(lambda1: float, lambda2: float, max_goals: int = 10, score_cap: int = 4) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted(_score_grid_features_uncached(lambda1, lambda2, max_goals=max_goals, score_cap=score_cap).items()))


def _score_grid_features_uncached(lambda1: float, lambda2: float, max_goals: int = 10, score_cap: int = 4) -> Dict[str, float]:
    grid = poisson_score_grid(lambda1=lambda1, lambda2=lambda2, max_goals=max_goals)
    max_goals = grid.shape[0] - 1
    masks = _score_grid_masks(max_goals)
    features: Dict[str, float] = {}
    for home_goals in range(score_cap + 1):
        for away_goals in range(score_cap + 1):
            value = grid[home_goals, away_goals] if home_goals <= max_goals and away_goals <= max_goals else 0.0
            features[f"prob_score_{home_goals}_{away_goals}"] = float(value)

    goals = masks["goals"]
    total_goals = masks["total_goals_int"].ravel()
    flat_grid = grid.ravel()
    total_distribution = np.bincount(total_goals, weights=flat_grid, minlength=(max_goals * 2) + 1)
    home_distribution = grid.sum(axis=1)
    away_distribution = grid.sum(axis=0)
    total_values = np.arange(total_distribution.shape[0], dtype=float)
    goal_values = goals.astype(float)
    total_mean, total_var, total_skew, total_kurt = _distribution_moments_from_arrays(total_values, total_distribution)
    home_mean, home_var, home_skew, home_kurt = _distribution_moments_from_arrays(goal_values, home_distribution)
    away_mean, away_var, away_skew, away_kurt = _distribution_moments_from_arrays(goal_values, away_distribution)
    margins = masks["margin"]
    btts = float(grid[masks["btts"]].sum())
    draw = float(grid[masks["draw"]].sum())
    home_win_by_1 = float(grid[margins == 1].sum())
    home_win_by_2plus = float(grid[margins >= 2].sum())
    away_win_by_1 = float(grid[margins == -1].sum())
    away_win_by_2plus = float(grid[margins <= -2].sum())
    features.update({
        "prob_home_clean_sheet": float(grid[:, 0].sum()),
        "prob_away_clean_sheet": float(grid[0, :].sum()),
        "prob_home_2plus_goals": float(grid[2:, :].sum()),
        "prob_away_2plus_goals": float(grid[:, 2:].sum()),
        "prob_total_0_1": float(total_distribution[:2].sum()),
        "prob_total_2_3": float(total_distribution[2:4].sum()),
        "prob_total_4_5": float(total_distribution[4:6].sum()),
        "prob_total_6plus": float(total_distribution[6:].sum()),
        "prob_margin_home_1": float(home_win_by_1),
        "prob_margin_home_2plus": float(home_win_by_2plus),
        "prob_margin_draw": float(draw),
        "prob_margin_away_1": float(away_win_by_1),
        "prob_margin_away_2plus": float(away_win_by_2plus),
        "prob_btts": float(btts),
        "prob_no_btts": float(1.0 - btts),
        "total_goals_mean": float(total_mean),
        "total_goals_variance": float(total_var),
        "total_goals_skew": float(total_skew),
        "total_goals_kurtosis": float(total_kurt),
        "total_goals_tail_5plus": float(total_distribution[5:].sum()),
        "home_goals_variance": float(home_var),
        "away_goals_variance": float(away_var),
        "home_goals_skew": float(home_skew),
        "away_goals_skew": float(away_skew),
        "home_goals_kurtosis": float(home_kurt),
        "away_goals_kurtosis": float(away_kurt),
        "goal_mean_balance": float(home_mean - away_mean),
        "goal_variance_balance": float(home_var - away_var),
    })
    for line in TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        over_prob = float(total_distribution[total_values > line].sum())
        features[f"prob_over{suffix}"] = over_prob
        features[f"prob_under{suffix}"] = float(1.0 - over_prob)
    return features


def poisson_score_grid(lambda1: float, lambda2: float, max_goals: int = 10) -> np.ndarray:
    max_goals = _normalized_max_goals(max_goals)
    probs1 = _poisson_pmf_vector(lambda1, max_goals)
    probs2 = _poisson_pmf_vector(lambda2, max_goals)
    grid = np.outer(probs1, probs2)
    total = float(grid.sum())
    if total <= 0.0:
        return np.full((max_goals + 1, max_goals + 1), 1.0 / ((max_goals + 1) ** 2), dtype=float)
    return grid / total


def dixon_coles_probabilities(lambda1: float, lambda2: float, rho: float = 0.0, max_goals: int = 10) -> Dict[str, float]:
    return dict(_dixon_coles_probabilities_cached(
        round(float(lambda1), 4),
        round(float(lambda2), 4),
        round(float(rho), 5),
        int(max_goals),
    ))


@lru_cache(maxsize=8192)
def _dixon_coles_probabilities_cached(lambda1: float, lambda2: float, rho: float = 0.0, max_goals: int = 10) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted(_dixon_coles_probabilities_uncached(lambda1, lambda2, rho=rho, max_goals=max_goals).items()))


def _dixon_coles_probabilities_uncached(lambda1: float, lambda2: float, rho: float = 0.0, max_goals: int = 10) -> Dict[str, float]:
    adjusted = dixon_coles_score_grid(lambda1=lambda1, lambda2=lambda2, rho=rho, max_goals=max_goals)
    max_goals = adjusted.shape[0] - 1
    masks = _score_grid_masks(max_goals)
    rho = float(_clamp(rho, -0.25, 0.25))
    home = float(adjusted[masks["home"]].sum())
    draw = float(adjusted[masks["draw"]].sum())
    away = float(adjusted[masks["away"]].sum())
    over25 = float(adjusted[masks["total_goals"] >= 3.0].sum())
    return {
        "dc_rho": rho,
        "dc_prob_home_win": float(home),
        "dc_prob_draw": float(draw),
        "dc_prob_away_win": float(away),
        "dc_prob_over25": float(over25),
        "dc_prob_under25": float(1.0 - over25),
        "dc_low_score_mass": float(adjusted[:2, :2].sum()),
    }


def estimate_dixon_coles_rho(history_df: pd.DataFrame, max_goals: int = 10) -> float:
    if history_df is None or history_df.empty or not {"G1", "G2"}.issubset(history_df.columns):
        return 0.0
    working = history_df.copy()
    working["G1"] = pd.to_numeric(working["G1"], errors="coerce")
    working["G2"] = pd.to_numeric(working["G2"], errors="coerce")
    working = working[working["G1"].notna() & working["G2"].notna()].copy()
    if working.empty:
        return 0.0
    lambda1 = float(max(working["G1"].mean(), 0.2))
    lambda2 = float(max(working["G2"].mean(), 0.2))
    max_goals = _normalized_max_goals(max_goals)
    observed_home = working["G1"].to_numpy(dtype=float)
    observed_away = working["G2"].to_numpy(dtype=float)
    observed_home = np.clip(np.nan_to_num(observed_home, nan=0.0), 0, max_goals).astype(int)
    observed_away = np.clip(np.nan_to_num(observed_away, nan=0.0), 0, max_goals).astype(int)
    observed_counts = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    np.add.at(observed_counts, (observed_home, observed_away), 1.0)
    candidates = np.linspace(-0.2, 0.2, 41)
    best_rho = 0.0
    best_ll = -float("inf")
    for rho in candidates:
        grid = dixon_coles_score_grid(lambda1, lambda2, float(rho), max_goals=max_goals)
        log_likelihood = float(np.sum(observed_counts * np.log(np.maximum(grid, 1e-12))))
        if log_likelihood > best_ll:
            best_ll = log_likelihood
            best_rho = float(rho)
    return float(best_rho)


def dixon_coles_score_grid(lambda1: float, lambda2: float, rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    grid = poisson_score_grid(lambda1=lambda1, lambda2=lambda2, max_goals=max_goals)
    adjusted = grid.copy()
    rho = float(_clamp(rho, -0.25, 0.25))
    adjusted[:2, :2] *= np.maximum(np.asarray([
        [1.0 - lambda1 * lambda2 * rho, 1.0 + lambda1 * rho],
        [1.0 + lambda2 * rho, 1.0 - rho],
    ], dtype=float), 1e-6)
    return adjusted / max(float(adjusted.sum()), 1e-9)


def _normalized_max_goals(max_goals: int) -> int:
    return int(_clamp(max_goals, 4, 14))


@lru_cache(maxsize=32)
def _poisson_goal_cache(max_goals: int) -> Tuple[np.ndarray, np.ndarray]:
    max_goals = _normalized_max_goals(max_goals)
    goals = np.arange(max_goals + 1, dtype=float)
    factorials = np.asarray([math.factorial(int(goal)) for goal in goals], dtype=float)
    return goals, factorials


def _poisson_pmf_vector(rate: float, max_goals: int) -> np.ndarray:
    goals, factorials = _poisson_goal_cache(max_goals)
    rate = float(rate)
    if not np.isfinite(rate) or rate < 0.0:
        rate = 0.0
    if rate <= 0.0:
        output = np.zeros_like(goals, dtype=float)
        output[0] = 1.0
        return output
    return np.exp(-rate) * np.power(rate, goals) / factorials


@lru_cache(maxsize=32)
def _score_grid_masks(max_goals: int) -> Dict[str, Any]:
    max_goals = _normalized_max_goals(max_goals)
    goals = np.arange(max_goals + 1, dtype=int)
    home_goals, away_goals = np.meshgrid(goals, goals, indexing="ij")
    total_goals = home_goals + away_goals
    margin = home_goals - away_goals
    return {
        "goals": goals,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "total_goals": total_goals.astype(float),
        "total_goals_int": total_goals,
        "margin": margin,
        "home": margin > 0,
        "draw": margin == 0,
        "away": margin < 0,
        "btts": (home_goals > 0) & (away_goals > 0),
        "over": {line: total_goals > line for line in TOTAL_GOAL_LINES},
    }


def _distribution_moments_from_arrays(values: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float, float, float]:
    probs = np.asarray(probabilities, dtype=float)
    values = np.asarray(values, dtype=float)
    total = max(float(probs.sum()), 1e-12)
    mean = float(np.sum(values * probs) / total)
    centered = values - mean
    variance = float(np.sum((centered ** 2) * probs) / total)
    if variance <= 1e-12:
        return mean, 0.0, 0.0, 0.0
    std = math.sqrt(variance)
    skew = float(np.sum((centered ** 3) * probs) / total / (std ** 3))
    kurtosis = float(np.sum((centered ** 4) * probs) / total / (variance ** 2))
    return mean, variance, skew, kurtosis


def distribution_moments(distribution: Dict[int, float]) -> Tuple[float, float, float, float]:
    total = max(float(sum(distribution.values())), 1e-12)
    mean = sum(float(value) * float(prob) for value, prob in distribution.items()) / total
    variance = sum(((float(value) - mean) ** 2) * float(prob) for value, prob in distribution.items()) / total
    if variance <= 1e-12:
        return mean, 0.0, 0.0, 0.0
    std = math.sqrt(variance)
    skew = sum(((float(value) - mean) ** 3) * float(prob) for value, prob in distribution.items()) / total / (std ** 3)
    kurtosis = sum(((float(value) - mean) ** 4) * float(prob) for value, prob in distribution.items()) / total / (variance ** 2)
    return float(mean), float(variance), float(skew), float(kurtosis)


def total_line_suffix(line: float) -> str:
    return str(line).replace(".", "")


def _update_elo(ratings: Dict[str, float], team1: str, team2: str, g1: int, g2: int, k_factor: float = 28.0) -> None:
    rating1 = ratings[team1]
    rating2 = ratings[team2]
    expected1 = 1.0 / (1.0 + 10 ** ((rating2 - rating1) / 400.0))
    result1 = 1.0 if g1 > g2 else 0.5 if g1 == g2 else 0.0
    margin = abs(g1 - g2)
    multiplier = 1.0 if margin <= 1 else math.log(margin + 1.0)
    change = float(k_factor) * multiplier * (result1 - expected1)
    ratings[team1] = rating1 + change
    ratings[team2] = rating2 - change


def _match_weights(history: pd.DataFrame, recency_weight: float) -> pd.Series:
    if recency_weight <= 0 or "Date" not in history:
        return pd.Series([1.0] * len(history), index=history.index)
    dates = pd.to_datetime(history["Date"], errors="coerce")
    if dates.notna().sum() < 2:
        return pd.Series([1.0] * len(history), index=history.index)
    latest = dates.max()
    years_ago = (latest - dates).dt.days.fillna(0).clip(lower=0) / 365.25
    recent_curve = np.exp(-years_ago / 12.0)
    weights = (1.0 - recency_weight) + recency_weight * recent_curve
    return pd.Series(weights, index=history.index).clip(lower=0.1)


def _blend_toward_neutral(value: float, history_weight: float) -> float:
    if history_weight >= 1.0:
        return value
    return 1.0 + (value - 1.0) * history_weight


def _poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * (rate ** k) / math.factorial(k)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
