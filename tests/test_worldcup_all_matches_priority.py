import pickle

import numpy as np
import pandas as pd
import pytest

from src.worldcup import training
from src.worldcup import international_provider


def minimal_normalized_dataset() -> dict:
    return {
        "train": pd.DataFrame(),
        "test": pd.DataFrame(),
        "team_train": pd.DataFrame(),
        "team_test": pd.DataFrame(),
        "team_prediction": pd.DataFrame([{"Team": "Mexico"}]),
        "team_features": pd.DataFrame(),
        "target_column": "",
        "team_columns": [],
        "training_mode": "",
        "trainable": False,
        "preview": {"columns": [], "rows": [], "total": 0},
    }


def empty_market_bundle() -> dict:
    return {
        "matches": pd.DataFrame(),
        "qualifiers": pd.DataFrame(),
        "qualifier_rows": 0,
        "status": "missing",
        "warnings": [],
        "sources": [],
        "loaded_at": "",
    }


def empty_api_football_bundle() -> dict:
    return {
        "fixtures": pd.DataFrame(),
        "statistics": pd.DataFrame(),
        "team_stats": pd.DataFrame(),
        "lineups": pd.DataFrame(),
        "injuries": pd.DataFrame(),
        "odds": pd.DataFrame(),
        "market_rows": pd.DataFrame(),
        "status": "missing",
        "warnings": [],
        "sources": [],
        "loaded_at": "",
    }


def test_worldcup_tournament_detector_is_mens_senior_only():
    accepted = [
        "FIFA World Cup",
        "World Cup",
        "FIFA World Cup 2022",
        "World Cup 2018",
    ]
    rejected = [
        "FIFA World Cup qualification",
        "FIFA Futsal World Cup",
        "FIFA Futsal World Cup 2024",
        "FIFA Women's World Cup",
        "FIFA U-20 World Cup",
        "FIFA U-17 World Cup",
        "FIFA Club World Cup",
        "FIFA Beach Soccer World Cup",
    ]

    assert all(international_provider.is_worldcup_tournament(name) for name in accepted)
    assert not any(international_provider.is_worldcup_tournament(name) for name in rejected)


def test_all_matches_rows_normalize_targets_metadata_and_weights():
    raw = pd.DataFrame([
        {
            "date": "2022-11-20",
            "home_team": "Qatar",
            "away_team": "Ecuador",
            "home_score": 0,
            "away_score": 2,
            "tournament": "FIFA World Cup",
        },
        {
            "date": "2021-09-02",
            "home_team": "Mexico",
            "away_team": "Canada",
            "home_score": 2,
            "away_score": 1,
            "tournament": "FIFA World Cup qualification",
        },
    ])

    matches = international_provider.normalize_international_matches(raw)
    rows = training.international_match_rows(matches)

    assert rows.shape[0] == 2
    worldcup = rows[rows["Home"] == "Qatar"].iloc[0]
    qualifier = rows[rows["Home"] == "Mexico"].iloc[0]
    assert worldcup["Source"] == "all_matches.csv"
    assert worldcup["Label"] == "A"
    assert worldcup["OverUnder15"] == 1
    assert bool(worldcup["is_worldcup_match"]) is True
    assert worldcup["sample_weight"] == pytest.approx(training.SAMPLE_WEIGHT_POLICY["worldcup"])
    assert bool(qualifier["is_worldcup_match"]) is False
    assert qualifier["sample_weight"] == pytest.approx(training.SAMPLE_WEIGHT_POLICY["qualifier"])


def test_deduplicate_prefers_worldcup_metadata_on_same_match_key():
    rows = training.sanitize_match_rows(pd.DataFrame([
        {
            "Date": "2022-11-20",
            "Year": 2022,
            "Home": "Qatar",
            "Away": "Ecuador",
            "HG": 0,
            "AG": 2,
            "Label": "A",
            "Source": "all_matches.csv",
            "tournament": "FIFA World Cup",
            "is_worldcup_match": False,
            "label_source": "all_matches.csv",
        },
        {
            "Date": "2022-11-20",
            "Year": 2022,
            "Home": "Qatar",
            "Away": "Ecuador",
            "HG": 0,
            "AG": 2,
            "Label": "A",
            "Source": "cache:worldcup_2022.json",
            "tournament": "FIFA World Cup",
            "stage": "Matchday 1",
            "group": "Group A",
            "is_worldcup_match": True,
            "label_source": "historical_worldcup",
        },
    ]))

    deduped = training.deduplicate_labeled_matches(rows)

    assert deduped.shape[0] == 1
    assert deduped.iloc[0]["Source"] == "cache:worldcup_2022.json"
    assert deduped.iloc[0]["stage"] == "Matchday 1"
    assert bool(deduped.iloc[0]["is_worldcup_match"]) is True


def test_latest_worldcup_split_uses_2022_only_for_eval_and_excludes_later_context():
    rows = training.sanitize_match_rows(pd.DataFrame([
        {
            "Date": "2018-06-15",
            "Year": 2018,
            "Home": "Russia",
            "Away": "Saudi Arabia",
            "HG": 5,
            "AG": 0,
            "Label": "H",
            "Source": "worldcup",
            "tournament": "FIFA World Cup",
            "is_worldcup_match": True,
        },
        {
            "Date": "2022-09-01",
            "Year": 2022,
            "Home": "Mexico",
            "Away": "Canada",
            "HG": 2,
            "AG": 1,
            "Label": "H",
            "Source": "all_matches.csv",
            "tournament": "Friendly",
        },
        {
            "Date": "2022-11-20",
            "Year": 2022,
            "Home": "Qatar",
            "Away": "Ecuador",
            "HG": 0,
            "AG": 2,
            "Label": "A",
            "Source": "worldcup",
            "tournament": "FIFA World Cup",
            "is_worldcup_match": True,
        },
        {
            "Date": "2023-03-01",
            "Year": 2023,
            "Home": "Mexico",
            "Away": "Canada",
            "HG": 4,
            "AG": 0,
            "Label": "H",
            "Source": "all_matches.csv",
            "tournament": "Friendly",
        },
    ]))

    train, test, final_year, warning = training.split_latest_worldcup_test(rows)

    assert final_year == "2022"
    assert set(test["Home"]) == {"Qatar"}
    assert set(train["Home"]) == {"Russia", "Mexico"}
    assert "posteriores" in warning
    assert pd.to_datetime(train["Date"]).max() < pd.Timestamp("2022-11-20")


def test_latest_worldcup_split_ignores_futsal_2024_and_2026_target_labels():
    rows = training.sanitize_match_rows(pd.DataFrame([
        {
            "Date": "2018-06-15",
            "Year": 2018,
            "Home": "Russia",
            "Away": "Saudi Arabia",
            "HG": 5,
            "AG": 0,
            "Label": "H",
            "Source": "all_matches.csv",
            "tournament": "FIFA World Cup",
        },
        {
            "Date": "2022-11-20",
            "Year": 2022,
            "Home": "Qatar",
            "Away": "Ecuador",
            "HG": 0,
            "AG": 2,
            "Label": "A",
            "Source": "all_matches.csv",
            "tournament": "FIFA World Cup",
        },
        {
            "Date": "2024-09-14",
            "Year": 2024,
            "Home": "Uzbekistan",
            "Away": "Netherlands",
            "HG": 3,
            "AG": 3,
            "Label": "D",
            "Source": "all_matches.csv",
            "tournament": "FIFA Futsal World Cup",
        },
        {
            "Date": "2026-06-11",
            "Year": 2026,
            "Home": "Mexico",
            "Away": "South Africa",
            "HG": 2,
            "AG": 1,
            "Label": "H",
            "Source": "all_matches.csv",
            "tournament": "FIFA World Cup",
        },
    ]))

    train, test, final_year, warning = training.split_latest_worldcup_test(rows)

    assert final_year == "2022"
    assert set(test["tournament"]) == {"FIFA World Cup"}
    assert set(test["Home"]) == {"Qatar"}
    assert set(train["Home"]) == {"Russia"}
    assert "2022" in warning


def test_prepared_dataset_metadata_uses_2022_benchmark_and_policy_notes(tmp_path, monkeypatch):
    matches = pd.DataFrame([
        {"date": "2018-06-15", "home_team": "Russia", "away_team": "Saudi Arabia", "home_score": 5, "away_score": 0, "tournament": "FIFA World Cup", "country": "Russia", "neutral": False},
        {"date": "2021-09-02", "home_team": "Mexico", "away_team": "Canada", "home_score": 1, "away_score": 0, "tournament": "Friendly", "country": "Mexico", "neutral": False},
        {"date": "2022-11-20", "home_team": "Qatar", "away_team": "Ecuador", "home_score": 0, "away_score": 2, "tournament": "FIFA World Cup", "country": "Qatar", "neutral": False},
        {"date": "2024-09-14", "home_team": "Uzbekistan", "away_team": "Netherlands", "home_score": 3, "away_score": 3, "tournament": "FIFA Futsal World Cup", "country": "Uzbekistan", "neutral": False},
        {"date": "2026-06-11", "home_team": "Mexico", "away_team": "South Africa", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "country": "Mexico", "neutral": False},
    ])

    monkeypatch.setattr(training, "load_historical_matches", lambda refresh=False: (pd.DataFrame(), "none"))
    monkeypatch.setattr(training, "load_market_data", lambda **kwargs: empty_market_bundle())
    monkeypatch.setattr(training, "load_api_football_data", lambda **kwargs: empty_api_football_bundle())
    monkeypatch.setattr(training, "load_international_matches", lambda required=False: matches)
    monkeypatch.setattr(training, "international_results_status", lambda: {
        "available": True,
        "source_path": str(tmp_path / "all_matches.csv"),
        "rows": int(matches.shape[0]),
        "all_matches_rows": int(matches.shape[0]),
        "worldcup_rows": int(matches["tournament"].map(international_provider.is_worldcup_tournament).sum()),
    })

    prepared = training.build_prepared_dataset(
        files=[],
        normalized=minimal_normalized_dataset(),
        refresh_history=False,
    )

    assert prepared["prepared_schema_version"] == training.PREPARED_SCHEMA_VERSION
    assert prepared["target_worldcup_year"] == "2026"
    assert prepared["benchmark_worldcup_year"] == "2022"
    assert prepared["final_test_year"] == "2022"
    assert prepared["benchmark_policy"] == training.BENCHMARK_POLICY
    assert prepared["worldcup_rows"] == 2
    assert int(pd.to_numeric(prepared["train"]["Year"]).max()) == 2021
    assert set(prepared["test"]["Home"]) == {"Qatar"}
    assert any("Year >= 2026" in note for note in prepared["label_policy_notes"])
    assert not any("Test final bloqueado" in warning for warning in prepared["warnings"])
    assert not any("anti-leakage" in warning and "2026" in warning for warning in prepared["warnings"])


def test_prepared_dataset_status_marks_old_schema_stale_without_old_warnings(tmp_path, monkeypatch):
    prepared_path = tmp_path / "prepared.pkl"
    meta_path = tmp_path / "prepared.json"
    normalized = minimal_normalized_dataset()
    old_dataset = {
        **normalized,
        "prepared_at": "2026-06-01T00:00:00+00:00",
        "training_mode": "match_result",
        "source_files": [],
        "label_source": "all_matches.csv",
        "warnings": ["Test final bloqueado al Mundial 2024"],
        "final_test_year": "2024",
        "split_policy": "latest_worldcup_final_test",
        "over_under_ready": True,
        "goals_distribution_ready": True,
    }
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    with prepared_path.open("wb") as handle:
        pickle.dump(old_dataset, handle)

    monkeypatch.setattr(training, "PREPARED_DATASET_FILE", prepared_path)
    monkeypatch.setattr(training, "PREPARED_DATASET_META_FILE", meta_path)
    monkeypatch.setattr(training, "international_results_status", lambda: {"available": False})

    status = training.prepared_dataset_status(files=[], normalized=normalized)

    assert status["ready"] is True
    assert status["stale"] is True
    assert status["prepared_schema_version"] == ""
    assert status["benchmark_worldcup_year"] == ""
    assert status["final_test_year"] == ""
    assert status["warnings"] == []
    assert status["dataset"] is normalized


def test_ensure_prepared_dataset_current_regenerates_old_schema(tmp_path, monkeypatch):
    prepared_path = tmp_path / "prepared.pkl"
    meta_path = tmp_path / "prepared.json"
    kaggle_root = tmp_path / "kaggle"
    kaggle_root.mkdir(parents=True)
    (kaggle_root / "train.csv").write_text("home_team,away_team,home_goals,away_goals\nMexico,Canada,1,0\n", encoding="utf-8")
    old_dataset = {
        **minimal_normalized_dataset(),
        "prepared_at": "2026-06-01T00:00:00+00:00",
        "prepared_schema_version": "",
        "training_mode": "match_result",
        "source_files": [],
        "trainable": True,
    }
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    with prepared_path.open("wb") as handle:
        pickle.dump(old_dataset, handle)
    regenerated = {
        **minimal_normalized_dataset(),
        "prepared_schema_version": training.PREPARED_SCHEMA_VERSION,
        "target_column": "Label + GoalsDistribution + OverUnder05/15/25/35",
        "trainable": True,
        "over_under_ready": True,
        "goals_distribution_ready": True,
        "train": pd.DataFrame([
            {"Home": "Mexico", "Away": "Canada", "Label": "H", "HG": 1, "AG": 0, "OverUnder05": 1, "OverUnder15": 0, "OverUnder25": 0, "OverUnder35": 0},
            {"Home": "Canada", "Away": "Mexico", "Label": "A", "HG": 0, "AG": 2, "OverUnder05": 1, "OverUnder15": 1, "OverUnder25": 0, "OverUnder35": 0},
        ]),
    }
    calls = {"prepare": 0}

    def fake_prepare_training_dataset(force=False, refresh_history=False):
        calls["prepare"] += 1
        training.save_prepared_dataset(regenerated)
        return {"etl_ready": True}

    monkeypatch.setattr(training, "KAGGLE_ROOT", kaggle_root)
    monkeypatch.setattr(training, "PREPARED_DATASET_FILE", prepared_path)
    monkeypatch.setattr(training, "PREPARED_DATASET_META_FILE", meta_path)
    monkeypatch.setattr(training, "international_results_status", lambda: {"available": False})
    monkeypatch.setattr(training, "prepare_training_dataset", fake_prepare_training_dataset)

    current = training.ensure_prepared_dataset_current({}, progress_callback=None)

    assert calls["prepare"] == 1
    assert current["prepared_schema_version"] == training.PREPARED_SCHEMA_VERSION


def test_build_training_matrix_recent15_uses_match_date_not_year_cache():
    rows = training.sanitize_match_rows(pd.DataFrame([
        {"Date": "2022-02-01", "Year": 2022, "Home": "Mexico", "Away": "Canada", "HG": 1, "AG": 0, "Label": "H", "Source": "train"},
        {"Date": "2022-05-01", "Year": 2022, "Home": "Mexico", "Away": "Canada", "HG": 2, "AG": 0, "Label": "H", "Source": "train"},
    ]))
    international_matches = pd.DataFrame([
        {"date": "2022-01-01", "home_team": "Mexico", "away_team": "Canada", "home_score": 1, "away_score": 0, "tournament": "Friendly", "country": "", "neutral": False},
        {"date": "2022-04-01", "home_team": "Mexico", "away_team": "Canada", "home_score": 3, "away_score": 0, "tournament": "Friendly", "country": "", "neutral": False},
    ])
    history_df = pd.DataFrame([
        {"Date": "2021-01-01", "Year": 2021, "Team 1": "Mexico", "Team 2": "Canada", "G1": 1, "G2": 0, "Round": "Group", "Group": "Group A"},
    ])

    x, _, _ = training.build_training_matrix(
        rows,
        history_df=history_df,
        teams=["Mexico", "Canada"],
        team_features=pd.DataFrame(),
        international_matches=international_matches,
    )

    assert x.iloc[0]["recent15_matches_home"] == pytest.approx(1.0)
    assert x.iloc[1]["recent15_matches_home"] == pytest.approx(2.0)


def test_sample_weights_are_passed_to_supported_classifier(monkeypatch):
    captured = {}

    class CapturingClassifier:
        def fit(self, x, y, sample_weight=None):
            captured["sample_weight"] = sample_weight
            self.classes_ = np.asarray([0, 1])
            return self

    monkeypatch.setattr(training, "resolve_device", lambda model_key, requested_device: ("cpu", []))
    monkeypatch.setattr(training, "build_worldcup_classifier", lambda **kwargs: CapturingClassifier())

    training.fit_configured_classifier(
        x_train=pd.DataFrame({"feature": [0.0, 1.0]}),
        y_train=pd.Series([0, 1]),
        model_key="xgboost",
        params={},
        n_jobs=1,
        requested_device="cpu",
        seed=7,
        num_classes=2,
        sample_weight=pd.Series([1.65, 0.6]),
    )

    assert captured["sample_weight"].tolist() == pytest.approx([1.65, 0.6])


def test_preferred_over_under_sources_keep_binary_unless_goals_distribution_is_better():
    markets = {
        "over_under_25": {
            "metrics": {"eval": {"F1": 0.72}},
        },
    }
    weaker_goals = {
        "derived_total_markets": {
            "over_under_25": {"metrics": {"eval": {"F1": 0.71}}},
        },
    }
    stronger_goals = {
        "derived_total_markets": {
            "over_under_25": {"metrics": {"eval": {"F1": 0.8}}},
        },
    }

    assert training.preferred_over_under_sources(markets, weaker_goals)["over_under_25"]["kind"] == "binary_ml"
    assert training.preferred_over_under_sources(markets, stronger_goals)["over_under_25"]["kind"] == "goal_distribution_ml"


def test_build_training_matrix_cache_reuses_features_across_market_targets(monkeypatch):
    rows = training.sanitize_match_rows(pd.DataFrame([
        {"Date": "2022-02-01", "Year": 2022, "Home": "Mexico", "Away": "Canada", "HG": 1, "AG": 0, "Label": "H", "Source": "train"},
        {"Date": "2022-05-01", "Year": 2022, "Home": "Canada", "Away": "Mexico", "HG": 2, "AG": 1, "Label": "H", "Source": "train"},
    ]))
    model = training.WorldCupModel.from_history(pd.DataFrame(), teams=["Mexico", "Canada"])
    cache = training.WorldCupFeatureBuildCache()
    calls = {"count": 0}
    original = training.match_feature_row

    def counted_match_feature_row(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "match_feature_row", counted_match_feature_row)

    x_result, y_result, feature_columns = training.build_training_matrix(
        rows,
        base_model=model,
        team_features=pd.DataFrame(),
        target="result",
        feature_cache=cache,
    )
    x_ou, y_ou, ou_columns = training.build_training_matrix(
        rows,
        base_model=model,
        team_features=pd.DataFrame(),
        target="over_under_25",
        feature_cache=cache,
    )

    assert calls["count"] == rows.shape[0]
    assert cache.summary()["matrix_hits"] == 1
    assert feature_columns == ou_columns
    assert x_result.equals(x_ou)
    assert y_result.tolist() == ["H", "H"]
    assert y_ou.tolist() == [0, 1]
