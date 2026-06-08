"""World Cup 2026 data, modeling and simulation helpers."""

from src.worldcup.data import (
    FALLBACK_2026_GROUPS,
    group_stage_matches,
    groups_dataframe,
    load_historical_matches,
    load_players,
    load_tournament_2026,
    teams_dataframe,
    tournament_fixtures_dataframe,
)
from src.worldcup.lanus_provider import (
    link_fixture_lineup,
    lineup_payload_for_fixture,
    lineup_rating_adjustments,
    lineups_summary,
    lineups_table,
)
from src.worldcup.model import WorldCupModel
from src.worldcup.simulation import simulate_worldcup

__all__ = [
    "FALLBACK_2026_GROUPS",
    "WorldCupModel",
    "group_stage_matches",
    "groups_dataframe",
    "load_historical_matches",
    "load_players",
    "load_tournament_2026",
    "link_fixture_lineup",
    "lineup_payload_for_fixture",
    "lineup_rating_adjustments",
    "lineups_summary",
    "lineups_table",
    "simulate_worldcup",
    "teams_dataframe",
    "tournament_fixtures_dataframe",
]
