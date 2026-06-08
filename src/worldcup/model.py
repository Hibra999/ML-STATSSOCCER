from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple

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
    def __init__(self, profiles: Dict[str, TeamProfile], global_g1: float = 1.25, global_g2: float = 1.05):
        self._profiles = profiles
        self.global_g1 = float(global_g1 or 1.25)
        self.global_g2 = float(global_g2 or 1.05)

    @classmethod
    def from_history(cls, historical_df: pd.DataFrame, teams: Iterable[str]) -> "WorldCupModel":
        team_list = sorted(set(teams))
        ratings = {team: float(TEAM_RATING_PRIORS.get(team, 1500)) for team in team_list}
        stats = {team: {"matches": 0, "gf": 0.0, "ga": 0.0} for team in team_list}
        global_g1 = 1.25
        global_g2 = 1.05

        if historical_df is not None and not historical_df.empty:
            global_g1 = max(float(historical_df["G1"].mean()), 0.5)
            global_g2 = max(float(historical_df["G2"].mean()), 0.5)
            for _, row in historical_df.sort_values("Date", kind="stable").iterrows():
                team1 = str(row["Team 1"])
                team2 = str(row["Team 2"])
                g1 = int(row["G1"])
                g2 = int(row["G2"])
                ratings.setdefault(team1, float(TEAM_RATING_PRIORS.get(team1, 1500)))
                ratings.setdefault(team2, float(TEAM_RATING_PRIORS.get(team2, 1500)))
                stats.setdefault(team1, {"matches": 0, "gf": 0.0, "ga": 0.0})
                stats.setdefault(team2, {"matches": 0, "gf": 0.0, "ga": 0.0})
                _update_elo(ratings, team1, team2, g1, g2)
                stats[team1]["matches"] += 1
                stats[team1]["gf"] += g1
                stats[team1]["ga"] += g2
                stats[team2]["matches"] += 1
                stats[team2]["gf"] += g2
                stats[team2]["ga"] += g1

        global_team_goals = max((global_g1 + global_g2) / 2.0, 0.5)
        profiles: Dict[str, TeamProfile] = {}
        for team in team_list:
            team_stats = stats.get(team, {"matches": 0, "gf": 0.0, "ga": 0.0})
            matches = int(team_stats["matches"])
            gf_per_match = float(team_stats["gf"] / matches) if matches else global_team_goals
            ga_per_match = float(team_stats["ga"] / matches) if matches else global_team_goals
            attack = _clamp(gf_per_match / global_team_goals, 0.55, 1.75)
            defense = _clamp(ga_per_match / global_team_goals, 0.55, 1.75)
            profiles[team] = TeamProfile(
                team=team,
                rating=float(ratings.get(team, TEAM_RATING_PRIORS.get(team, 1500))),
                matches=matches,
                gf_per_match=gf_per_match,
                ga_per_match=ga_per_match,
                attack=attack,
                defense=defense,
            )
        return cls(profiles=profiles, global_g1=global_g1, global_g2=global_g2)

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
        return WorldCupModel(profiles=profiles, global_g1=self.global_g1, global_g2=self.global_g2)

    def expected_goals(self, team1: str, team2: str) -> Tuple[float, float]:
        p1 = self.profile(team1)
        p2 = self.profile(team2)
        rating1 = p1.rating + (45 if team1 in HOST_TEAMS else 0)
        rating2 = p2.rating + (45 if team2 in HOST_TEAMS else 0)
        diff = (rating1 - rating2) / 650.0
        lambda1 = self.global_g1 * math.sqrt(p1.attack * p2.defense) * math.exp(diff)
        lambda2 = self.global_g2 * math.sqrt(p2.attack * p1.defense) * math.exp(-diff)
        return _clamp(lambda1, 0.2, 4.5), _clamp(lambda2, 0.2, 4.5)

    def match_probabilities(self, team1: str, team2: str, max_goals: int = 10) -> Dict[str, float]:
        lambda1, lambda2 = self.expected_goals(team1, team2)
        probs1 = [_poisson_pmf(goals, lambda1) for goals in range(max_goals + 1)]
        probs2 = [_poisson_pmf(goals, lambda2) for goals in range(max_goals + 1)]
        total = 0.0
        home = 0.0
        draw = 0.0
        away = 0.0
        over25 = 0.0
        modal_score = (0, 0)
        modal_prob = -1.0
        for goals1, prob1 in enumerate(probs1):
            for goals2, prob2 in enumerate(probs2):
                prob = prob1 * prob2
                total += prob
                if goals1 > goals2:
                    home += prob
                elif goals1 == goals2:
                    draw += prob
                else:
                    away += prob
                if goals1 + goals2 > 2.5:
                    over25 += prob
                if prob > modal_prob:
                    modal_prob = prob
                    modal_score = (goals1, goals2)
        total = max(total, 1e-9)
        return {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "home": home / total,
            "draw": draw / total,
            "away": away / total,
            "over25": over25 / total,
            "under25": 1.0 - over25 / total,
            "modal_g1": modal_score[0],
            "modal_g2": modal_score[1],
        }

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


def _update_elo(ratings: Dict[str, float], team1: str, team2: str, g1: int, g2: int) -> None:
    rating1 = ratings[team1]
    rating2 = ratings[team2]
    expected1 = 1.0 / (1.0 + 10 ** ((rating2 - rating1) / 400.0))
    result1 = 1.0 if g1 > g2 else 0.5 if g1 == g2 else 0.0
    margin = abs(g1 - g2)
    multiplier = 1.0 if margin <= 1 else math.log(margin + 1.0)
    change = 28.0 * multiplier * (result1 - expected1)
    ratings[team1] = rating1 + change
    ratings[team2] = rating2 - change


def _poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * (rate ** k) / math.factorial(k)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
