from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.worldcup.data import group_letter, group_sort_key, group_stage_matches, groups_from_tournament, knockout_matches
from src.worldcup.model import WorldCupModel


ADVANCEMENT_COLUMNS = [
    "Grupo",
    "Equipo",
    "Rating",
    "Top 2 %",
    "Mejor tercero %",
    "Pasa grupo %",
    "R32 %",
    "Octavos %",
    "Cuartos %",
    "Semis %",
    "Final %",
    "Campeon %",
]
COUNT_TOP2 = 0
COUNT_BEST_THIRD = 1
COUNT_GROUP_ADVANCE = 2
COUNT_ROUND32 = 3
COUNT_ROUND16 = 4
COUNT_QUARTER = 5
COUNT_SEMI = 6
COUNT_FINAL = 7
COUNT_CHAMPION = 8
COUNT_COLUMNS = 9


def simulate_worldcup(
        tournament: Dict[str, Any],
        model: WorldCupModel,
        iterations: int = 5000,
        seed: int = 2026,
        include_confirmed_results: bool = False,
        confirmed_results: List[Dict[str, Any]] | None = None,
        progress_callback=None,
) -> Dict[str, pd.DataFrame]:
    iterations = min(max(int(iterations or 5000), 100), 20000)
    seed = int(seed if seed is not None else 2026)
    rng = np.random.default_rng(seed)
    groups = groups_from_tournament(tournament)
    group_matches = group_stage_matches(tournament)
    knockouts = sorted(knockout_matches(tournament), key=lambda match: _knockout_sort_key(match))
    teams = [team for group in groups.values() for team in group]
    team_index = {team: index for index, team in enumerate(teams)}
    counts = np.zeros((len(teams), COUNT_COLUMNS), dtype=np.int32)
    ratings = np.asarray([model.profile(team).rating for team in teams], dtype=float)
    group_infos = _group_simulation_infos(groups, team_index)
    group_lookup = {info["group"]: index for index, info in enumerate(group_infos)}
    confirmed_specs = _confirmed_group_result_specs(
        confirmed_results or [],
        group_lookup,
        group_infos,
        team_index,
    ) if include_confirmed_results else []
    confirmed_keys = {spec[5] for spec in confirmed_specs}
    group_specs = _group_match_specs(group_matches, group_lookup, group_infos, team_index, model, skip_keys=confirmed_keys)
    group_lambda_home = np.asarray([spec[3] for spec in group_specs], dtype=float)
    group_lambda_away = np.asarray([spec[4] for spec in group_specs], dtype=float)
    pair_cache: Dict[Tuple[Any, ...], Tuple[float, float, float]] = {}
    report_every = max(1, iterations // 100)
    _emit_progress(progress_callback, "simulation", 0, iterations, "Monte Carlo en ejecucion")

    for iteration in range(iterations):
        points = [np.zeros(len(info["teams"]), dtype=np.int16) for info in group_infos]
        goals_for = [np.zeros(len(info["teams"]), dtype=np.int16) for info in group_infos]
        goals_against = [np.zeros(len(info["teams"]), dtype=np.int16) for info in group_infos]
        for group_idx, pos1, pos2, goals1, goals2, _ in confirmed_specs:
            goals_for[group_idx][pos1] += goals1
            goals_against[group_idx][pos1] += goals2
            goals_for[group_idx][pos2] += goals2
            goals_against[group_idx][pos2] += goals1
            if goals1 > goals2:
                points[group_idx][pos1] += 3
            elif goals2 > goals1:
                points[group_idx][pos2] += 3
            else:
                points[group_idx][pos1] += 1
                points[group_idx][pos2] += 1
        if group_specs:
            sampled_home = rng.poisson(group_lambda_home)
            sampled_away = rng.poisson(group_lambda_away)
            for spec_index, (group_idx, pos1, pos2, _, _, _) in enumerate(group_specs):
                goals1 = int(sampled_home[spec_index])
                goals2 = int(sampled_away[spec_index])
                goals_for[group_idx][pos1] += goals1
                goals_against[group_idx][pos1] += goals2
                goals_for[group_idx][pos2] += goals2
                goals_against[group_idx][pos2] += goals1
                if goals1 > goals2:
                    points[group_idx][pos1] += 3
                elif goals2 > goals1:
                    points[group_idx][pos2] += 3
                else:
                    points[group_idx][pos1] += 1
                    points[group_idx][pos2] += 1

        slots: Dict[str, int] = {}
        third_candidates: List[Tuple[int, int]] = []
        for group_idx, info in enumerate(group_infos):
            ranking = _rank_group_arrays(info, points[group_idx], goals_for[group_idx], goals_against[group_idx])
            if len(ranking) < 3:
                continue
            first_pos = ranking[0]
            second_pos = ranking[1]
            third_pos = ranking[2]
            first_idx = int(info["team_indices"][first_pos])
            second_idx = int(info["team_indices"][second_pos])
            slots[f"1{info['letter']}"] = first_idx
            slots[f"2{info['letter']}"] = second_idx
            counts[first_idx, COUNT_TOP2] += 1
            counts[second_idx, COUNT_TOP2] += 1
            third_candidates.append((group_idx, third_pos))

        best_thirds = _best_third_team_indices(third_candidates, group_infos, points, goals_for, goals_against)
        third_slots = {group_infos[group_idx]["letter"]: int(group_infos[group_idx]["team_indices"][team_pos]) for group_idx, team_pos in best_thirds}
        available_thirds = third_slots.copy()
        best_third_indices = set(third_slots.values())
        qualifiers = set(slots.values()) | best_third_indices
        for team_idx in qualifiers:
            counts[team_idx, COUNT_BEST_THIRD] += int(team_idx in best_third_indices)
            counts[team_idx, COUNT_GROUP_ADVANCE] += 1
            counts[team_idx, COUNT_ROUND32] += 1

        winners: Dict[int, int] = {}
        losers: Dict[int, int] = {}
        for match in knockouts:
            round_name = str(match.get("round", ""))
            team1_idx = _resolve_slot_index(str(match.get("team1", "")), slots, winners, losers, available_thirds, team_index, ratings)
            team2_idx = _resolve_slot_index(str(match.get("team2", "")), slots, winners, losers, available_thirds, team_index, ratings)
            if team1_idx < 0 or team2_idx < 0 or team1_idx == team2_idx:
                continue
            winner_idx, loser_idx = _sample_knockout_winner_index(team1_idx, team2_idx, teams, model, rng, pair_cache, match)
            number = int(match.get("num", len(winners) + 73))
            winners[number] = winner_idx
            losers[number] = loser_idx
            if round_name == "Round of 32":
                counts[winner_idx, COUNT_ROUND16] += 1
            elif round_name == "Round of 16":
                counts[winner_idx, COUNT_QUARTER] += 1
            elif round_name == "Quarter-final":
                counts[winner_idx, COUNT_SEMI] += 1
            elif round_name == "Semi-final":
                counts[winner_idx, COUNT_FINAL] += 1
            elif round_name == "Final":
                counts[winner_idx, COUNT_CHAMPION] += 1
        current = iteration + 1
        if current == iterations or current % report_every == 0:
            _emit_progress(progress_callback, "simulation", current, iterations, "Monte Carlo en ejecucion")

    advancement = _advancement_dataframe_from_counts(groups, model, counts, team_index, iterations)
    match_probs = match_probabilities_dataframe(group_matches, model)
    _emit_progress(progress_callback, "simulation", iterations, iterations, "Monte Carlo completado")
    return {"advancement": advancement, "matches": match_probs, "confirmed_results": pd.DataFrame(confirmed_results or [])}


def _emit_progress(callback, stage: str, current: int, total: int, message: str) -> None:
    if callback is None:
        return
    total = max(int(total or 1), 1)
    current = min(max(int(current or 0), 0), total)
    callback({
        "stage": stage,
        "current": current,
        "total": total,
        "current_trial": "",
        "total_trials": "",
        "percent": int(round(current * 100 / total)),
        "message": message,
    })


def _group_simulation_infos(groups: Dict[str, List[str]], team_index: Dict[str, int]) -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []
    for group, group_teams in groups.items():
        indices = np.asarray([team_index[team] for team in group_teams if team in team_index], dtype=int)
        teams = [team for team in group_teams if team in team_index]
        infos.append({
            "group": group,
            "letter": group_letter(group),
            "sort_key": group_sort_key(group),
            "teams": teams,
            "team_indices": indices,
            "positions": {int(team_idx): pos for pos, team_idx in enumerate(indices)},
        })
    return infos


def _group_match_specs(
        group_matches: List[Dict[str, Any]],
        group_lookup: Dict[str, int],
        group_infos: List[Dict[str, Any]],
        team_index: Dict[str, int],
        model: WorldCupModel,
        skip_keys: set[Tuple[str, str, str]] | None = None,
) -> List[Tuple[int, int, int, float, float, Dict[str, Any]]]:
    specs: List[Tuple[int, int, int, float, float, Dict[str, Any]]] = []
    skip_keys = skip_keys or set()
    for match in group_matches:
        group = str(match.get("group") or "")
        group_idx = group_lookup.get(group)
        team1 = str(match.get("team1") or "")
        team2 = str(match.get("team2") or "")
        if _result_match_key(group, team1, team2) in skip_keys:
            continue
        team1_idx = team_index.get(team1)
        team2_idx = team_index.get(team2)
        if group_idx is None or team1_idx is None or team2_idx is None:
            continue
        positions = group_infos[group_idx]["positions"]
        if team1_idx not in positions or team2_idx not in positions:
            continue
        lambda1, lambda2 = _model_expected_goals(model, team1, team2, match)
        specs.append((int(group_idx), int(positions[team1_idx]), int(positions[team2_idx]), float(lambda1), float(lambda2), match))
    return specs


def _model_expected_goals(model: WorldCupModel, team1: str, team2: str, match: Dict[str, Any] | None = None) -> Tuple[float, float]:
    method = getattr(model, "expected_goals_for_match", None)
    if callable(method):
        return method(team1, team2, match=match)
    return model.expected_goals(team1, team2)


def _model_match_probabilities(model: WorldCupModel, team1: str, team2: str, match: Dict[str, Any] | None = None) -> Dict[str, float]:
    method = getattr(model, "match_probabilities_for_match", None)
    if callable(method):
        return method(team1, team2, match=match)
    return model.match_probabilities(team1, team2)


def _confirmed_group_result_specs(
        confirmed_results: List[Dict[str, Any]],
        group_lookup: Dict[str, int],
        group_infos: List[Dict[str, Any]],
        team_index: Dict[str, int],
) -> List[Tuple[int, int, int, int, int, Tuple[str, str, str]]]:
    specs: List[Tuple[int, int, int, int, int, Tuple[str, str, str]]] = []
    for result in confirmed_results:
        group = str(result.get("group") or result.get("Grupo") or "")
        team1 = str(result.get("team1") or result.get("home") or result.get("Equipo 1") or "")
        team2 = str(result.get("team2") or result.get("away") or result.get("Equipo 2") or "")
        group_idx = group_lookup.get(group)
        team1_idx = team_index.get(team1)
        team2_idx = team_index.get(team2)
        if group_idx is None or team1_idx is None or team2_idx is None:
            continue
        try:
            goals1 = int(float(result.get("goals1", result.get("home_goals", result.get("Goles 1")))))
            goals2 = int(float(result.get("goals2", result.get("away_goals", result.get("Goles 2")))))
        except (TypeError, ValueError):
            continue
        positions = group_infos[group_idx]["positions"]
        if team1_idx not in positions or team2_idx not in positions:
            continue
        specs.append((
            int(group_idx),
            int(positions[team1_idx]),
            int(positions[team2_idx]),
            goals1,
            goals2,
            _result_match_key(group, team1, team2),
        ))
    return specs


def _result_match_key(group: str, team1: str, team2: str) -> Tuple[str, str, str]:
    return (str(group or "").strip().lower(), str(team1 or "").strip().lower(), str(team2 or "").strip().lower())


def _rank_group_arrays(info: Dict[str, Any], points: np.ndarray, goals_for: np.ndarray, goals_against: np.ndarray) -> List[int]:
    goal_diff = goals_for - goals_against
    teams = info["teams"]
    return sorted(
        range(len(teams)),
        key=lambda pos: (-int(points[pos]), -int(goal_diff[pos]), -int(goals_for[pos]), int(pos), teams[pos]),
    )


def _best_third_team_indices(
        third_candidates: List[Tuple[int, int]],
        group_infos: List[Dict[str, Any]],
        points: List[np.ndarray],
        goals_for: List[np.ndarray],
        goals_against: List[np.ndarray],
) -> List[Tuple[int, int]]:
    return sorted(
        third_candidates,
        key=lambda item: (
            -int(points[item[0]][item[1]]),
            -int((goals_for[item[0]] - goals_against[item[0]])[item[1]]),
            -int(goals_for[item[0]][item[1]]),
            group_infos[item[0]]["sort_key"],
        ),
    )[:8]


def _resolve_slot_index(
        token: str,
        slots: Dict[str, int],
        winners: Dict[int, int],
        losers: Dict[int, int],
        third_slots: Dict[str, int],
        team_index: Dict[str, int],
        ratings: np.ndarray,
) -> int:
    token = token.strip()
    if token in slots:
        return int(slots[token])
    if token.startswith("W") and token[1:].isdigit():
        return int(winners.get(int(token[1:]), -1))
    if token.startswith("L") and token[1:].isdigit():
        return int(losers.get(int(token[1:]), -1))
    if token.startswith("3"):
        allowed = [part for part in token[1:].split("/") if part]
        for letter in allowed:
            if letter in third_slots:
                return int(third_slots.pop(letter))
        if third_slots:
            best_letter = max(third_slots, key=lambda letter: ratings[int(third_slots[letter])])
            return int(third_slots.pop(best_letter))
    if token and not any(char.isdigit() for char in token):
        return int(team_index.get(token, -1))
    return -1


def _sample_knockout_winner_index(
        team1_idx: int,
        team2_idx: int,
        teams: List[str],
        model: WorldCupModel,
        rng: np.random.Generator,
        pair_cache: Dict[Tuple[Any, ...], Tuple[float, float, float]],
        match: Dict[str, Any] | None = None,
) -> Tuple[int, int]:
    cache_key = (int(team1_idx), int(team2_idx), str((match or {}).get("date", "")))
    if cache_key not in pair_cache:
        team1 = teams[team1_idx]
        team2 = teams[team2_idx]
        probabilities = _model_match_probabilities(model, team1, team2, match)
        lambda1 = float(probabilities.get("lambda1", 1.0))
        lambda2 = float(probabilities.get("lambda2", 1.0))
        win_share = float(probabilities.get("home", 0.0)) / max(
            float(probabilities.get("home", 0.0)) + float(probabilities.get("away", 0.0)),
            1e-9,
        )
        pair_cache[cache_key] = (lambda1, lambda2, win_share)
    lambda1, lambda2, win_share = pair_cache[cache_key]
    goals1 = int(rng.poisson(lambda1))
    goals2 = int(rng.poisson(lambda2))
    if goals1 > goals2:
        return team1_idx, team2_idx
    if goals2 > goals1:
        return team2_idx, team1_idx
    if rng.random() <= win_share:
        return team1_idx, team2_idx
    return team2_idx, team1_idx


def match_probabilities_dataframe(matches: List[Dict[str, Any]], model: WorldCupModel) -> pd.DataFrame:
    rows = []
    for match in matches:
        team1 = str(match.get("team1", ""))
        team2 = str(match.get("team2", ""))
        probabilities = _model_match_probabilities(model, team1, team2, match)
        rows.append({
            "Fecha": match.get("date", ""),
            "Hora": match.get("time", ""),
            "Grupo": match.get("group", ""),
            "Equipo 1": team1,
            "Equipo 2": team2,
            "Goles E1": round(probabilities["lambda1"], 2),
            "Goles E2": round(probabilities["lambda2"], 2),
            "P E1 %": _pct(probabilities["home"]),
            "P Empate %": _pct(probabilities["draw"]),
            "P E2 %": _pct(probabilities["away"]),
            "Over 0.5 %": _pct(probabilities.get("over05", 0.0)),
            "Under 0.5 %": _pct(probabilities.get("under05", 0.0)),
            "Over 1.5 %": _pct(probabilities.get("over15", 0.0)),
            "Under 1.5 %": _pct(probabilities.get("under15", 0.0)),
            "Over 2.5 %": _pct(probabilities["over25"]),
            "Under 2.5 %": _pct(probabilities["under25"]),
            "Over 3.5 %": _pct(probabilities.get("over35", 0.0)),
            "Under 3.5 %": _pct(probabilities.get("under35", 0.0)),
            "Sede": match.get("ground", ""),
        })
    return pd.DataFrame(rows)


def _initial_standings(groups: Dict[str, List[str]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    return {
        group: {
            team: {"team": team, "played": 0, "points": 0, "gf": 0, "ga": 0, "gd": 0}
            for team in teams
        }
        for group, teams in groups.items()
    }


def _apply_group_result(table: Dict[str, Dict[str, int]], team1: str, team2: str, goals1: int, goals2: int) -> None:
    if team1 not in table or team2 not in table:
        return
    table[team1]["played"] += 1
    table[team2]["played"] += 1
    table[team1]["gf"] += goals1
    table[team1]["ga"] += goals2
    table[team2]["gf"] += goals2
    table[team2]["ga"] += goals1
    table[team1]["gd"] = table[team1]["gf"] - table[team1]["ga"]
    table[team2]["gd"] = table[team2]["gf"] - table[team2]["ga"]
    if goals1 > goals2:
        table[team1]["points"] += 3
    elif goals2 > goals1:
        table[team2]["points"] += 3
    else:
        table[team1]["points"] += 1
        table[team2]["points"] += 1


def _rank_group(group: str, table: Dict[str, Dict[str, int]], seed_order: List[str]) -> List[Dict[str, int]]:
    seed_rank = {team: index for index, team in enumerate(seed_order)}
    return sorted(
        table.values(),
        key=lambda row: (-row["points"], -row["gd"], -row["gf"], seed_rank.get(row["team"], 99), row["team"]),
    )


def _best_third_teams(third_candidates: List[Tuple[str, Dict[str, int]]]) -> List[Tuple[str, Dict[str, int]]]:
    return sorted(
        third_candidates,
        key=lambda item: (-item[1]["points"], -item[1]["gd"], -item[1]["gf"], group_sort_key(item[0])),
    )[:8]


def _resolve_slot(
        token: str,
        slots: Dict[str, str],
        winners: Dict[int, str],
        losers: Dict[int, str],
        third_slots: Dict[str, str],
        model: WorldCupModel,
) -> str:
    token = token.strip()
    if token in slots:
        return slots[token]
    if token.startswith("W") and token[1:].isdigit():
        return winners.get(int(token[1:]), "")
    if token.startswith("L") and token[1:].isdigit():
        return losers.get(int(token[1:]), "")
    if token.startswith("3"):
        allowed = [part for part in token[1:].split("/") if part]
        for letter in allowed:
            if letter in third_slots:
                return third_slots.pop(letter)
        if third_slots:
            best_letter = max(third_slots, key=lambda letter: model.profile(third_slots[letter]).rating)
            return third_slots.pop(best_letter)
    return token if token and not any(char.isdigit() for char in token) else ""


def _advancement_dataframe_from_counts(
        groups: Dict[str, List[str]],
        model: WorldCupModel,
        counts: np.ndarray,
        team_index: Dict[str, int],
        iterations: int,
) -> pd.DataFrame:
    rows = []
    for group, teams in groups.items():
        for team in teams:
            idx = team_index[team]
            profile = model.profile(team)
            team_counts = counts[idx]
            rows.append({
                "Grupo": group,
                "Equipo": team,
                "Rating": round(profile.rating, 1),
                "Top 2 %": _pct(team_counts[COUNT_TOP2] / iterations),
                "Mejor tercero %": _pct(team_counts[COUNT_BEST_THIRD] / iterations),
                "Pasa grupo %": _pct(team_counts[COUNT_GROUP_ADVANCE] / iterations),
                "R32 %": _pct(team_counts[COUNT_ROUND32] / iterations),
                "Octavos %": _pct(team_counts[COUNT_ROUND16] / iterations),
                "Cuartos %": _pct(team_counts[COUNT_QUARTER] / iterations),
                "Semis %": _pct(team_counts[COUNT_SEMI] / iterations),
                "Final %": _pct(team_counts[COUNT_FINAL] / iterations),
                "Campeon %": _pct(team_counts[COUNT_CHAMPION] / iterations),
            })
    return pd.DataFrame(rows, columns=ADVANCEMENT_COLUMNS)


def _advancement_dataframe(groups: Dict[str, List[str]], model: WorldCupModel, counters: Dict[str, Counter], iterations: int) -> pd.DataFrame:
    rows = []
    for group, teams in groups.items():
        for team in teams:
            counter = counters[team]
            profile = model.profile(team)
            rows.append({
                "Grupo": group,
                "Equipo": team,
                "Rating": round(profile.rating, 1),
                "Top 2 %": _pct(counter["top2"] / iterations),
                "Mejor tercero %": _pct(counter["best_third"] / iterations),
                "Pasa grupo %": _pct(counter["group_advance"] / iterations),
                "R32 %": _pct(counter["round32"] / iterations),
                "Octavos %": _pct(counter["round16"] / iterations),
                "Cuartos %": _pct(counter["quarter"] / iterations),
                "Semis %": _pct(counter["semi"] / iterations),
                "Final %": _pct(counter["final"] / iterations),
                "Campeon %": _pct(counter["champion"] / iterations),
            })
    return pd.DataFrame(rows, columns=ADVANCEMENT_COLUMNS)


def _knockout_sort_key(match: Dict[str, Any]) -> Tuple[int, str]:
    round_order = {
        "Round of 32": 1,
        "Round of 16": 2,
        "Quarter-final": 3,
        "Semi-final": 4,
        "Final": 5,
    }
    return (round_order.get(str(match.get("round", "")), 99), str(match.get("date", "")))


def _pct(value: float) -> float:
    return round(float(value) * 100.0, 1)
