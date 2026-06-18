import pandas as pd


def test_prepare_advanced_data_writes_api_football_xg_cache(tmp_path, monkeypatch):
    from src.worldcup import advanced_data

    advanced_root = tmp_path / "advanced"
    xg_root = tmp_path / "xg"
    monkeypatch.setattr(advanced_data, "ADVANCED_ROOT", advanced_root)
    monkeypatch.setattr(advanced_data, "XG_ROOT", xg_root)
    monkeypatch.setattr(advanced_data, "MATCH_FEATURES_FILE", advanced_root / "match_features.csv")
    monkeypatch.setattr(advanced_data, "STATUS_FILE", advanced_root / "status.json")
    monkeypatch.setattr(advanced_data, "LOCAL_SOURCE_FILES", {
        "manual_xg": xg_root / "manual_xg.csv",
        "api_football_xg": xg_root / "api_football_xg.csv",
        "shots": xg_root / "shots.csv",
        "events": xg_root / "events.csv",
        "xthreat": xg_root / "xthreat.csv",
        "psxg": xg_root / "psxg.csv",
    })

    team_stats = pd.DataFrame([
        {
            "FixtureId": "10",
            "Team": "Mexico",
            "Opponent": "Canada",
            "Date": "2026-06-11T20:00:00+00:00",
            "Side": "home",
            "xg_for": 1.7,
            "xg_against": 0.8,
            "total_shots_for": 13,
            "total_shots_against": 7,
        },
        {
            "FixtureId": "10",
            "Team": "Canada",
            "Opponent": "Mexico",
            "Date": "2026-06-11T20:00:00+00:00",
            "Side": "away",
            "xg_for": 0.8,
            "xg_against": 1.7,
            "total_shots_for": 7,
            "total_shots_against": 13,
        },
    ])

    def fake_load_api_football_data(allow_download=False, force_download=False):
        assert allow_download is True
        assert force_download is False
        return {"team_stats": team_stats, "warnings": [], "sources": ["api-cache.json"]}

    monkeypatch.setattr(advanced_data, "api_football_key", lambda: "test-key")
    monkeypatch.setattr(advanced_data, "load_api_football_data", fake_load_api_football_data)

    status = advanced_data.prepare_advanced_data({"use_api_football": True, "allow_api_download": True})

    assert status["prepared_rows"] == 1
    assert "api_football_xg" in status["active_sources"]
    assert (xg_root / "manual_xg.csv").exists()
    assert (xg_root / "api_football_xg.csv").exists()
    assert (advanced_root / "match_features.csv").exists()
    assert not any("No se encontraron filas locales" in warning for warning in status["warnings"])
    assert not any("Sin cache avanzado local" in warning for warning in status["warnings"])
