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
    auto_refresh_lineups,
    autodetect_fixture_event,
    link_fixture_lineup,
    lineup_payload_for_fixture,
    lineup_payload_from_detected_event,
    lineup_rating_adjustments,
    lineups_summary,
    lineups_table,
    player_feature_rating_adjustments,
    player_features_dataframe,
    player_stats_payload_for_fixture,
)
from src.worldcup.model import WorldCupModel
from src.worldcup.simulation import simulate_worldcup

__all__ = [
    "FALLBACK_2026_GROUPS",
    "WorldCupModel",
    "auto_refresh_lineups",
    "autodetect_fixture_event",
    "group_stage_matches",
    "groups_dataframe",
    "load_historical_matches",
    "load_players",
    "load_tournament_2026",
    "link_fixture_lineup",
    "lineup_payload_for_fixture",
    "lineup_payload_from_detected_event",
    "lineup_rating_adjustments",
    "lineups_summary",
    "lineups_table",
    "player_feature_rating_adjustments",
    "player_features_dataframe",
    "player_stats_payload_for_fixture",
    "simulate_worldcup",
    "teams_dataframe",
    "tournament_fixtures_dataframe",
]
