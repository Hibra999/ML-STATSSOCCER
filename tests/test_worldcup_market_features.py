from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from src.worldcup import training
from src.worldcup import api_football_provider
from src.worldcup.api_football_provider import api_football_feature_table, normalize_api_football_payloads
from src.worldcup.market_provider import (
    load_football_data_workbook,
    market_feature_row,
    no_vig_probabilities,
    normalize_market_frame,
    qualifier_feature_table,
)
from src.worldcup.model import WorldCupModel


def test_football_data_parser_normalizes_worldcup_workbook(tmp_path):
    workbook = tmp_path / "WorldCup2026.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame([
            {
                "Date": "2014-06-13",
                "Home": "Mexico",
                "Away": "Cameroon",
                "HG": 1,
                "AG": 0,
                "H-Avg": 2.10,
                "D-Avg": 3.20,
                "A-Avg": 4.10,
            },
        ]).to_excel(writer, sheet_name="WorldCup2014", index=False)
        pd.DataFrame([
            {
                "Date": "2025-06-07",
                "Home": "Mexico",
                "Away": "Canada",
                "HG": 2,
                "AG": 1,
                "H_Avg": 1.80,
                "D_Avg": 3.40,
                "A_Avg": 4.50,
                "HXG": 1.7,
                "AXG": 0.9,
                "HS": 13,
                "AS": 8,
                "HST": 5,
                "AST": 3,
            },
        ]).to_excel(writer, sheet_name="WorldCup2026Qualifiers", index=False)

    rows = load_football_data_workbook(workbook)

    assert set(rows["market_sheet"]) == {"WorldCup2014", "WorldCup2026Qualifiers"}
    mexico = rows[rows["market_sheet"] == "WorldCup2014"].iloc[0]
    qualifier = rows[rows["market_sheet"] == "WorldCup2026Qualifiers"].iloc[0]
    assert mexico["market_odds_home"] == pytest.approx(2.10)
    assert mexico["market_odds_draw"] == pytest.approx(3.20)
    assert mexico["market_odds_away"] == pytest.approx(4.10)
    assert bool(qualifier["is_qualifier"]) is True
    assert qualifier["home_xg"] == pytest.approx(1.7)
    assert qualifier["away_shots_on_target"] == pytest.approx(3)


def test_market_probability_features_remove_vig_and_compare_model():
    implied, no_vig, vig = no_vig_probabilities({"home": 2.0, "draw": 3.5, "away": 4.0})
    features = market_feature_row(
        {
            "market_odds_home": 2.0,
            "market_odds_draw": 3.5,
            "market_odds_away": 4.0,
            "market_odds_over25": 1.9,
            "market_odds_under25": 1.95,
        },
        model_probs={"H": 0.55, "D": 0.25, "A": 0.20},
        model_totals={"over25": 0.52, "under25": 0.48},
    )

    assert implied["home"] == pytest.approx(0.5)
    assert vig == pytest.approx((1 / 2.0) + (1 / 3.5) + (1 / 4.0) - 1.0)
    assert sum(no_vig.values()) == pytest.approx(1.0)
    assert features["market_has_1x2"] == 1.0
    assert features["market_has_ou25"] == 1.0
    assert features["market_prob_home"] > features["market_prob_away"]
    assert features["market_logit_home"] != 0.0
    assert features["model_vs_market_kl_1x2"] >= 0.0
    assert features["model_vs_market_over25_abs"] == pytest.approx(abs(0.52 - features["market_prob_over25"]))


def test_qualifier_features_exclude_future_matches_from_match_row():
    model = WorldCupModel.from_history(pd.DataFrame(), teams=["Mexico", "Canada"])
    rows = training.sanitize_match_rows(pd.DataFrame([
        {
            "Date": "2026-06-10",
            "Home": "Mexico",
            "Away": "Canada",
            "Label": "H",
            "HG": 1,
            "AG": 0,
            "OverUnder25": 0,
            "Source": "fixture",
        },
    ]))
    future_qualifiers = normalize_market_frame(pd.DataFrame([
        {
            "Date": "2026-06-11",
            "Year": 2026,
            "Home": "Mexico",
            "Away": "Canada",
            "HG": 4,
            "AG": 0,
            "is_qualifier": True,
            "market_source": "football-data:qualifier",
        },
    ]))

    x, _, _ = training.build_training_matrix(
        rows,
        base_model=model,
        qualifier_rows=future_qualifiers,
        team_features=pd.DataFrame(),
        target="result",
    )

    assert x.iloc[0]["qualifier_context_available"] == 0.0
    assert x.iloc[0].get("qualifier_matches_home", 0.0) == 0.0


def test_api_football_provider_without_key_does_not_call_network(tmp_path, monkeypatch):
    monkeypatch.setattr(api_football_provider, "API_FOOTBALL_RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(api_football_provider, "DOTENV_FILE", tmp_path / ".env")
    for env_name in api_football_provider.API_FOOTBALL_ENV_KEYS:
        monkeypatch.delenv(env_name, raising=False)

    class NoNetwork:
        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called without API_FOOTBALL_KEY")

    result = api_football_provider.load_api_football_data(allow_download=True, session=NoNetwork())

    assert result["status"] == "missing"
    assert any("api-football" in warning.lower() for warning in result["warnings"])


def test_api_football_key_reads_dotenv_when_environment_is_empty(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("export API_FOOTBALL_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(api_football_provider, "DOTENV_FILE", env_file)
    for env_name in api_football_provider.API_FOOTBALL_ENV_KEYS:
        monkeypatch.delenv(env_name, raising=False)

    assert api_football_provider.api_football_key() == "from-dotenv"


def test_api_football_payloads_normalize_stats_and_odds():
    payloads = [
        {
            "endpoint": "/fixtures",
            "params": {"league": 1, "season": 2022},
            "fetched_at": "2022-11-20T10:00:00+00:00",
            "payload": {
                "response": [{
                    "fixture": {"id": 10, "date": "2022-11-21T16:00:00+00:00", "venue": {"name": "Test Stadium"}, "status": {"short": "FT"}},
                    "league": {"id": 1, "name": "World Cup", "season": 2022, "round": "Group A - 1"},
                    "teams": {"home": {"id": 1, "name": "Mexico"}, "away": {"id": 2, "name": "Canada"}},
                    "goals": {"home": 2, "away": 1},
                }],
            },
        },
        {
            "endpoint": "/fixtures/statistics",
            "params": {"fixture": 10},
            "fetched_at": "2022-11-21T18:10:00+00:00",
            "payload": {
                "response": [
                    {"team": {"id": 1, "name": "Mexico"}, "statistics": [{"type": "Total Shots", "value": 13}, {"type": "Shots on Goal", "value": 5}]},
                    {"team": {"id": 2, "name": "Canada"}, "statistics": [{"type": "Total Shots", "value": 8}, {"type": "Shots on Goal", "value": 2}]},
                ],
            },
        },
        {
            "endpoint": "/odds",
            "params": {"fixture": 10},
            "fetched_at": "2022-11-20T11:00:00+00:00",
            "payload": {
                "response": [{
                    "fixture": {"id": 10},
                    "bookmakers": [{
                        "name": "book",
                        "bets": [
                            {"name": "Match Winner", "values": [{"value": "Home", "odd": "1.90"}, {"value": "Draw", "odd": "3.40"}, {"value": "Away", "odd": "4.60"}]},
                            {"name": "Goals Over/Under", "values": [{"value": "Over 2.5", "odd": "1.95"}, {"value": "Under 2.5", "odd": "1.85"}]},
                        ],
                    }],
                }],
            },
        },
    ]

    result = normalize_api_football_payloads(payloads)

    assert result["fixtures"].iloc[0]["Home"] == "Mexico"
    assert result["team_stats"].shape[0] == 2
    assert result["team_stats"].iloc[0]["total_shots_for"] == pytest.approx(13)
    assert result["team_stats"].iloc[0]["total_shots_against"] == pytest.approx(8)
    assert result["market_rows"].iloc[0]["market_odds_home"] == pytest.approx(1.90)
    assert result["market_rows"].iloc[0]["market_odds_over25"] == pytest.approx(1.95)


def test_api_football_features_exclude_future_matches_from_match_row():
    model = WorldCupModel.from_history(pd.DataFrame(), teams=["Mexico", "Canada"])
    rows = training.sanitize_match_rows(pd.DataFrame([
        {
            "Date": "2026-06-10",
            "Home": "Mexico",
            "Away": "Canada",
            "Label": "H",
            "HG": 1,
            "AG": 0,
            "OverUnder25": 0,
            "Source": "fixture",
        },
    ]))
    api_team_stats = pd.DataFrame([
        {"Date": "2026-06-09", "Team": "Mexico", "Opponent": "Canada", "GF": 2, "GA": 0, "GoalDiff": 2, "Points": 3, "Win": 1, "Draw": 0, "Loss": 0, "Over25": 0, "Under25": 1, "BTTS": 0, "CleanSheet": 1, "total_shots_for": 12, "total_shots_against": 5},
        {"Date": "2026-06-10", "Team": "Mexico", "Opponent": "Canada", "GF": 8, "GA": 0, "GoalDiff": 8, "Points": 3, "Win": 1, "Draw": 0, "Loss": 0, "Over25": 1, "Under25": 0, "BTTS": 0, "CleanSheet": 1, "total_shots_for": 99, "total_shots_against": 1},
        {"Date": "2026-06-09", "Team": "Canada", "Opponent": "Mexico", "GF": 0, "GA": 2, "GoalDiff": -2, "Points": 0, "Win": 0, "Draw": 0, "Loss": 1, "Over25": 0, "Under25": 1, "BTTS": 0, "CleanSheet": 0, "total_shots_for": 5, "total_shots_against": 12},
    ])

    x, _, _ = training.build_training_matrix(
        rows,
        base_model=model,
        team_features=pd.DataFrame(),
        api_football={"team_stats": api_team_stats},
        target="result",
    )

    assert x.iloc[0]["api_football_matches_home"] == pytest.approx(1.0)
    assert x.iloc[0]["api_football_total_shots_for_avg_home"] == pytest.approx(12.0)
    assert x.iloc[0]["api_football_total_shots_for_avg_diff"] == pytest.approx(7.0)


def test_api_football_features_accept_timezone_aware_dates():
    team_stats = pd.DataFrame([
        {"Date": "2026-06-09T22:00:00+00:00", "Team": "Mexico", "Opponent": "Canada", "GF": 2, "GA": 0, "GoalDiff": 2, "Points": 3, "Win": 1, "Draw": 0, "Loss": 0, "Over25": 0, "Under25": 1, "BTTS": 0, "CleanSheet": 1},
        {"Date": "2026-06-11T00:00:00+00:00", "Team": "Mexico", "Opponent": "Canada", "GF": 8, "GA": 0, "GoalDiff": 8, "Points": 3, "Win": 1, "Draw": 0, "Loss": 0, "Over25": 1, "Under25": 0, "BTTS": 0, "CleanSheet": 1},
    ])

    features = api_football_feature_table(team_stats, reference_date="2026-06-10", teams=["Mexico"])

    assert features.iloc[0]["matches"] == pytest.approx(1.0)
    assert features.iloc[0]["all_goals_for_avg"] == pytest.approx(2.0)


def test_api_football_context_requires_prefetched_rows():
    team_stats = pd.DataFrame([
        {"Date": "2026-06-09", "Team": "Mexico", "Opponent": "Canada", "GF": 2, "GA": 0, "GoalDiff": 2, "Points": 3, "Win": 1, "Draw": 0, "Loss": 0, "Over25": 0, "Under25": 1, "BTTS": 0, "CleanSheet": 1},
    ])
    lineups = pd.DataFrame([
        {"Date": "2026-06-11", "Team": "Mexico", "FixtureId": "1", "StartXI": 11, "fetched_at": "2026-06-12T00:00:00+00:00"},
    ])

    features = api_football_feature_table(team_stats, reference_date="2026-06-11", teams=["Mexico"], lineups=lineups)

    assert features.iloc[0]["lineup_context_available"] == 0.0
    assert features.iloc[0]["lineup_rows"] == 0.0


def test_international_recent_provider_aliases_features_and_contextual_poisson(tmp_path, monkeypatch):
    from src.worldcup import international_provider

    monkeypatch.setattr(international_provider, "INTERNATIONAL_ROOT", tmp_path)
    monkeypatch.setattr(international_provider, "INTERNATIONAL_MATCHES_FILE", tmp_path / "all_matches.csv")
    pd.DataFrame([
        {"date": "2025-01-10", "home_team": "USA", "away_team": "Czech Republic", "home_score": 2, "away_score": 1, "tournament": "Friendly", "country": "USA", "neutral": False},
        {"date": "2025-03-01", "home_team": "Mexico", "away_team": "South Africa", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup qualification", "country": "Mexico", "neutral": False},
        {"date": "2025-06-01", "home_team": "South Africa", "away_team": "Bosnia & Herzegovina", "home_score": 1, "away_score": 1, "tournament": "Friendly", "country": "South Africa", "neutral": False},
        {"date": "2025-10-01", "home_team": "South Africa", "away_team": "Mexico", "home_score": 1, "away_score": 3, "tournament": "CONCACAF Gold Cup", "country": "USA", "neutral": True},
        {"date": "2026-07-01", "home_team": "Mexico", "away_team": "Canada", "home_score": 0, "away_score": 0, "tournament": "Friendly", "country": "Mexico", "neutral": False},
    ]).to_csv(international_provider.INTERNATIONAL_MATCHES_FILE, index=False)

    matches = international_provider.load_international_matches(required=True)
    model = WorldCupModel.from_history(pd.DataFrame(), teams=["Mexico", "South Africa", "United States", "Czechia", "Bosnia and Herzegovina"])
    features = international_provider.recent15_feature_table(
        matches,
        teams=["Mexico", "South Africa"],
        before_date="2026-06-11",
        base_model=model,
    )
    context = international_provider.contextual_poisson_for_match(
        "Mexico",
        "South Africa",
        base_model=model,
        before_date="2026-06-11",
        max_goals=6,
        matches=matches,
    )
    row = training.match_feature_row(
        model,
        pd.DataFrame(),
        "Mexico",
        "South Africa",
        recent15_features=features,
    )

    assert set(matches["home_team"]) >= {"United States", "Mexico", "South Africa"}
    assert "Czechia" in set(matches["away_team"])
    assert "Bosnia and Herzegovina" in set(matches["away_team"])
    assert features.loc[features["Team"] == "Mexico", "recent15_matches"].iloc[0] == pytest.approx(2.0)
    assert features.loc[features["Team"] == "South Africa", "recent15_friendly_matches"].iloc[0] == pytest.approx(1.0)
    assert row["recent15_context_available"] == 1.0
    assert row["recent15_matches_home"] == pytest.approx(2.0)
    assert "recent15_recent15_matches_home" not in row
    assert context["available"] is True
    status = international_provider.international_results_status()
    assert status["exists"] is True
    assert status["available"] is True
    assert status["rows"] == 5
    assert context["context_lambda_home"] > 0
    assert set(context["probabilities"]) >= {"home", "draw", "away", "over25", "under25"}
    assert len(context["top_scores"]) == 5
    assert len(context["score_matrix"]) == 7
    assert len(context["recent_matches"]["home"]) == 2


def test_international_provider_uses_valid_alternate_csv_name(tmp_path, monkeypatch):
    from src.worldcup import international_provider

    monkeypatch.setattr(international_provider, "INTERNATIONAL_ROOT", tmp_path)
    monkeypatch.setattr(international_provider, "INTERNATIONAL_MATCHES_FILE", tmp_path / "all_matches.csv")
    pd.DataFrame([
        {"match_date": "2025-02-01", "home": "USA", "away": "Canada", "home_goals": 2, "away_goals": 0, "competition": "Friendly"},
        {"match_date": "2025-03-01", "home": "Mexico", "away": "Canada", "home_goals": 1, "away_goals": 1, "competition": "Gold Cup"},
    ]).to_csv(tmp_path / "results.csv", index=False)

    matches = international_provider.load_international_matches(required=True)
    status = international_provider.international_results_status()

    assert international_provider.INTERNATIONAL_MATCHES_FILE.exists() is False
    assert matches.shape[0] == 2
    assert set(matches["home_team"]) == {"United States", "Mexico"}
    assert status["exists"] is False
    assert status["available"] is True
    assert status["rows"] == 2
    assert status["source_path"].endswith("results.csv")
    assert "warning" in status


def test_download_international_results_copies_valid_alternate_to_all_matches(tmp_path, monkeypatch):
    from src.worldcup import international_provider

    local_root = tmp_path / "local"
    source_root = tmp_path / "downloaded"
    source_root.mkdir()
    monkeypatch.setattr(international_provider, "INTERNATIONAL_ROOT", local_root)
    monkeypatch.setattr(international_provider, "INTERNATIONAL_MATCHES_FILE", local_root / "all_matches.csv")
    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(dataset_download=lambda slug: str(source_root)),
    )
    pd.DataFrame([
        {"match_date": "2025-02-01", "home": "USA", "away": "Canada", "home_goals": 2, "away_goals": 0},
    ]).to_csv(source_root / "results.csv", index=False)

    status = international_provider.download_international_results(force=True)
    matches = international_provider.load_international_matches(required=True)

    assert international_provider.INTERNATIONAL_MATCHES_FILE.exists() is True
    assert status["available"] is True
    assert status["exists"] is True
    assert status["rows"] == 1
    assert status["source_file"].endswith("results.csv")
    assert status["copied_files"] == [str(international_provider.INTERNATIONAL_MATCHES_FILE)]
    assert matches.iloc[0]["home_team"] == "United States"


def test_contextual_poisson_missing_all_matches_returns_base_matrix(tmp_path, monkeypatch):
    from src.worldcup import international_provider

    monkeypatch.setattr(international_provider, "INTERNATIONAL_ROOT", tmp_path)
    monkeypatch.setattr(international_provider, "INTERNATIONAL_MATCHES_FILE", tmp_path / "all_matches.csv")
    model = WorldCupModel.from_history(pd.DataFrame(), teams=["Mexico", "Canada"])

    context = international_provider.contextual_poisson_for_match(
        "Mexico",
        "Canada",
        base_model=model,
        before_date="2026-06-11",
        max_goals=4,
    )

    assert context["available"] is False
    assert context["matrix_available"] is True
    assert context["matrix_source"] == "base_model"
    assert context["reason"] == "all_matches.csv no disponible"
    assert set(context["probabilities"]) >= {"home", "draw", "away", "over25", "under25"}
    assert len(context["top_scores"]) == 5
    assert len(context["score_matrix"]) == 5
    assert len(context["heatmap"]["cells"]) == 25
    assert context["recent_matches"] == {"home": [], "away": []}


def test_match_feature_row_includes_market_dc_score_grid_shrinkage_history_h2h_and_context():
    history = pd.DataFrame([
        {"Date": "2010-06-11", "Team 1": "Mexico", "Team 2": "South Africa", "G1": 1, "G2": 1, "Round": "Group", "Group": "A"},
        {"Date": "2014-06-13", "Team 1": "Mexico", "Team 2": "Cameroon", "G1": 1, "G2": 0, "Round": "Group", "Group": "A"},
        {"Date": "2018-06-17", "Team 1": "Mexico", "Team 2": "Germany", "G1": 1, "G2": 0, "Round": "Group", "Group": "F"},
        {"Date": "2022-11-22", "Team 1": "Mexico", "Team 2": "Poland", "G1": 0, "G2": 0, "Round": "Group", "Group": "C"},
        {"Date": "2010-06-22", "Team 1": "South Africa", "Team 2": "France", "G1": 2, "G2": 1, "Round": "Group", "Group": "A"},
        {"Date": "2010-06-11", "Team 1": "Mexico", "Team 2": "South Africa", "G1": 1, "G2": 1, "Round": "Group", "Group": "A"},
    ])
    model = WorldCupModel.from_history(history, teams=["Mexico", "South Africa", "Cameroon", "Germany", "Poland", "France"])
    market_rows = normalize_market_frame(pd.DataFrame([
        {
            "Date": "2026-06-11",
            "Year": 2026,
            "Home": "Mexico",
            "Away": "South Africa",
            "market_odds_home": 1.85,
            "market_odds_draw": 3.40,
            "market_odds_away": 4.60,
            "market_source": "manual",
        },
    ]))
    qualifier_rows = normalize_market_frame(pd.DataFrame([
        {
            "Date": "2026-03-20",
            "Year": 2026,
            "Home": "Mexico",
            "Away": "Canada",
            "HG": 2,
            "AG": 1,
            "home_xg": 1.8,
            "away_xg": 0.7,
            "is_qualifier": True,
            "market_source": "football-data:qualifier",
        },
    ]))
    row = training.match_feature_row(
        model,
        pd.DataFrame(),
        "Mexico",
        "South Africa",
        history_team_features=training.build_history_feature_table(history),
        matchup_features=training.build_matchup_feature_table(history),
        market_rows=market_rows,
        qualifier_features=qualifier_feature_table(qualifier_rows, reference_date="2026-06-11", teams=["Mexico", "South Africa"]),
        match_date="2026-06-11",
        match_year=2026,
        fixture_context={"Round": "Final", "Group": "", "FixtureId": 103, "Date": "2026-06-11", "Venue": "Mexico City"},
        dc_rho=0.05,
    )

    assert row["market_has_1x2"] == 1.0
    assert "prob_score_4_4" in row
    assert "dc_prob_draw" in row
    assert "rating_home_shrunk" in row
    assert "model_entropy_1x2" in row
    assert "history_all_goals_for_skew_home" in row
    assert "history_all_trend_points_slope_home" in row
    assert "h2h_recency_weighted_points" in row
    assert row["stage_final"] == 1.0
    assert row["qualifier_context_available"] == 1.0
    assert row["qualifier_matches_home"] == pytest.approx(1.0)


def test_match_feature_row_fallback_without_market_or_qualifiers_keeps_zero_flags():
    model = WorldCupModel.from_history(pd.DataFrame(), teams=["Mexico", "Canada"])
    row = training.match_feature_row(model, pd.DataFrame(), "Mexico", "Canada")

    assert row["market_has_1x2"] == 0.0
    assert row["market_has_ou25"] == 0.0
    assert row["qualifier_context_available"] == 0.0
    assert "prob_score_0_0" in row
    assert "dc_prob_home_win" in row
