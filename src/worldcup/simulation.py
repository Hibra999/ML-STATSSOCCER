from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.worldcup.data import group_letter, group_sort_key, group_stage_matches, groups_from_tournament, knockout_matches
from src.worldcup.model import WorldCupModel
from src.worldcup.simulation_accelerated import sample_group_scores


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


def simulate_worldcup(
        tournament: Dict[str, Any],
        model: WorldCupModel,
        iterations: int = 5000,
        seed: int = 2026,
        progress_callback=None,
) -> Dict[str, pd.DataFrame]:
    iterations = min(max(int(iterations or 5000), 100), 20000)
    seed = int(seed if seed is not None else 2026)
    rng = np.random.default_rng(seed)
    groups = groups_from_tournament(tournament)
    group_matches = group_stage_matches(tournament)
    knockouts = sorted(knockout_matches(tournament), key=lambda match: _knockout_sort_key(match))
    teams = [team for group in groups.values() for team in group]
    counters = {team: Counter() for team in teams}
    group_score_samples, backend = sample_group_scores(group_matches, model, iterations, seed, prefer_cuda=True)
    report_every = max(1, iterations // 100)
    _emit_progress(progress_callback, "simulation", 0, iterations, backend.get("label", "Monte Carlo en ejecucion"))

    for iteration in range(iterations):
        standings = _initial_standings(groups)
        for match_index, match in enumerate(group_matches):
            team1 = str(match["team1"])
            team2 = str(match["team2"])
            if team1 not in counters or team2 not in counters:
                continue
            if group_score_samples is not None:
                goals1 = int(group_score_samples[iteration, match_index, 0])
                goals2 = int(group_score_samples[iteration, match_index, 1])
            else:
                goals1, goals2 = model.sample_score(team1, team2, rng)
            _apply_group_result(standings[str(match["group"])], team1, team2, goals1, goals2)

        ranked_groups = {group: _rank_group(group, table, groups[group]) for group, table in standings.items()}
        slots: Dict[str, str] = {}
        third_candidates = []
        for group, ranking in ranked_groups.items():
            letter = group_letter(group)
            first = ranking[0]["team"]
            second = ranking[1]["team"]
            third = ranking[2]
            slots[f"1{letter}"] = first
            slots[f"2{letter}"] = second
            counters[first]["top2"] += 1
            counters[second]["top2"] += 1
            third_candidates.append((group, third))

        best_thirds = _best_third_teams(third_candidates)
        third_slots = {group_letter(group): row["team"] for group, row in best_thirds}
        available_thirds = third_slots.copy()
        qualifiers = set(slots.values()) | set(third_slots.values())
        for team in qualifiers:
            counters[team]["best_third"] += int(team in third_slots.values())
            counters[team]["group_advance"] += 1
            counters[team]["round32"] += 1

        winners: Dict[int, str] = {}
        losers: Dict[int, str] = {}
        for match in knockouts:
            round_name = str(match.get("round", ""))
            team1 = _resolve_slot(str(match.get("team1", "")), slots, winners, losers, available_thirds, model)
            team2 = _resolve_slot(str(match.get("team2", "")), slots, winners, losers, available_thirds, model)
            if not team1 or not team2 or team1 == team2:
                continue
            winner, loser, _, _ = model.sample_knockout_winner(team1, team2, rng)
            number = int(match.get("num", len(winners) + 73))
            winners[number] = winner
            losers[number] = loser
            if round_name == "Round of 32":
                counters[winner]["round16"] += 1
            elif round_name == "Round of 16":
                counters[winner]["quarter"] += 1
            elif round_name == "Quarter-final":
                counters[winner]["semi"] += 1
            elif round_name == "Semi-final":
                counters[winner]["final"] += 1
            elif round_name == "Final":
                counters[winner]["champion"] += 1
        current = iteration + 1
        if current == iterations or current % report_every == 0:
            _emit_progress(progress_callback, "simulation", current, iterations, backend.get("label", "Monte Carlo en ejecucion"))

    advancement = _advancement_dataframe(groups, model, counters, iterations)
    match_probs = match_probabilities_dataframe(group_matches, model)
    _emit_progress(progress_callback, "simulation", iterations, iterations, "Monte Carlo completado")
    return {"advancement": advancement, "matches": match_probs, "backend": backend}


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


def match_probabilities_dataframe(matches: List[Dict[str, Any]], model: WorldCupModel) -> pd.DataFrame:
    rows = []
    for match in matches:
        team1 = str(match.get("team1", ""))
        team2 = str(match.get("team2", ""))
        probabilities = model.match_probabilities(team1, team2)
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
            "Over 2.5 %": _pct(probabilities["over25"]),
            "Under 2.5 %": _pct(probabilities["under25"]),
            "Marcador modal": f"{probabilities['modal_g1']}-{probabilities['modal_g2']}",
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
