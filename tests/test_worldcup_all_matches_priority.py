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


def make_international_matches(rows: int = 42, include_worldcup: bool = False) -> pd.DataFrame:
    teams = ["Mexico", "Canada", "South Africa", "Japan", "Brazil", "France"]
    scores = [(0, 0), (1, 0), (0, 2), (2, 1), (3, 1), (1, 3), (4, 2)]
    items = []
    for index in range(rows):
        home = teams[index % len(teams)]
        away = teams[(index + 1) % len(teams)]
        hg, ag = scores[index % len(scores)]
        items.append({
            "date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=index),
            "home_team": home,
            "away_team": away,
            "home_score": hg,
            "away_score": ag,
            "tournament": "Friendly" if index % 3 else "FIFA World Cup qualification",
            "country": home,
            "neutral": False,
        })
    if include_worldcup:
        items.append({
            "date": pd.Timestamp("2018-06-15"),
            "home_team": "Russia",
            "away_team": "Saudi Arabia",
            "home_score": 5,
            "away_score": 0,
            "tournament": "FIFA World Cup",
            "country": "Russia",
            "neutral": False,
        })
        items.append({
            "date": pd.Timestamp("2022-11-20"),
            "home_team": "Qatar",
            "away_team": "Ecuador",
            "home_score": 0,
            "away_score": 2,
            "tournament": "FIFA World Cup",
            "country": "Qatar",
            "neutral": False,
        })
    return pd.DataFrame(items)


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


def test_last_30_international_split_includes_worldcups_temporally():
    matches = make_international_matches(rows=35, include_worldcup=True)
    rows = training.international_match_rows(international_provider.normalize_international_matches(matches))

    train, test, warning = training.split_last_30_international_test(rows)

    assert train.shape[0] == 7
    assert test.shape[0] == 30
    assert pd.concat([train, test])["is_worldcup_match"].map(bool).any()
    assert pd.to_datetime(train["Date"]).max() < pd.to_datetime(test["Date"]).min()
    assert "FIFA World Cup incluidos" in warning


def test_validation_last_30_split_reserves_validation_before_test():
    rows = training.international_match_rows(international_provider.normalize_international_matches(make_international_matches(rows=42, include_worldcup=True)))

    train, validation, test, warning = training.split_validation_last_30_international_test(rows)

    assert train.shape[0] == 12
    assert validation.shape[0] == 2
    assert test.shape[0] == 30
    assert pd.to_datetime(train["Date"]).max() < pd.to_datetime(validation["Date"]).min()
    assert pd.to_datetime(validation["Date"]).max() < pd.to_datetime(test["Date"]).min()
    assert "validacion=2" in warning


def test_international_training_scope_keeps_since_2014_and_drops_future_dates():
    matches = pd.DataFrame([
        {"date": "2013-12-31", "home_team": "Mexico", "away_team": "Canada", "home_score": 1, "away_score": 0, "tournament": "Friendly"},
        {"date": "2014-01-01", "home_team": "Mexico", "away_team": "Canada", "home_score": 2, "away_score": 0, "tournament": "Friendly"},
        {"date": "2026-06-01", "home_team": "Canada", "away_team": "Mexico", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup qualification"},
        {"date": "2026-07-01", "home_team": "Brazil", "away_team": "France", "home_score": 1, "away_score": 1, "tournament": "Friendly"},
    ])
    rows = training.international_match_rows(international_provider.normalize_international_matches(matches))

    filtered, stats = training.filter_international_training_scope(rows, max_date="2026-06-13")

    assert filtered.shape[0] == 2
    assert pd.to_datetime(filtered["Date"]).dt.year.min() == 2014
    assert stats["removed_before_start"] == 1
    assert stats["removed_future"] == 1


def test_last_30_international_split_requires_31_international_rows():
    rows = training.international_match_rows(international_provider.normalize_international_matches(make_international_matches(rows=30)))

    with pytest.raises(training.WorldCupTrainingError, match="al menos 31"):
        training.split_last_30_international_test(rows)


def test_prepared_dataset_metadata_uses_last_30_international_test_and_policy_notes(tmp_path, monkeypatch):
    matches = make_international_matches(rows=42, include_worldcup=True)

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
    assert prepared["benchmark_worldcup_year"] == ""
    assert prepared["final_test_year"] == ""
    assert prepared["benchmark_policy"] == training.BENCHMARK_POLICY
    assert prepared["worldcup_rows"] == 2
    assert prepared["label_source"] == "all_matches.csv"
    assert prepared["split_policy"] == training.SPLIT_POLICY_VALIDATION_LAST_30
    assert prepared["training_start_year"] == 2014
    assert prepared["test"].shape[0] == 30
    assert prepared["validation"].shape[0] == 2
    assert pd.concat([prepared["train"], prepared["validation"], prepared["test"]])["is_worldcup_match"].map(bool).any()
    assert pd.to_datetime(prepared["train"]["Date"]).max() < pd.to_datetime(prepared["test"]["Date"]).min()
    assert any("desde 2014" in note for note in prepared["label_policy_notes"])
    assert not any("Test final bloqueado" in warning for warning in prepared["warnings"])
    assert not any("anti-leakage" in warning and "2026" in warning for warning in prepared["warnings"])


def test_prepared_dataset_requires_all_matches_csv(monkeypatch):
    monkeypatch.setattr(training, "load_historical_matches", lambda refresh=False: (pd.DataFrame(), "none"))
    monkeypatch.setattr(training, "load_market_data", lambda **kwargs: empty_market_bundle())
    monkeypatch.setattr(training, "load_api_football_data", lambda **kwargs: empty_api_football_bundle())
    monkeypatch.setattr(training, "load_international_matches", lambda required=True: (_ for _ in ()).throw(RuntimeError("No existe all_matches.csv")))
    monkeypatch.setattr(training, "international_results_status", lambda: {"available": False, "reason": "No existe all_matches.csv"})

    with pytest.raises(training.WorldCupTrainingError, match="all_matches.csv es obligatorio"):
        training.build_prepared_dataset(
            files=[],
            normalized=minimal_normalized_dataset(),
            refresh_history=False,
        )


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
    rows = training.international_match_rows(international_provider.normalize_international_matches(make_international_matches(rows=36)))
    train_rows, validation_rows, test_rows, _ = training.split_validation_last_30_international_test(rows)
    regenerated = {
        **minimal_normalized_dataset(),
        "prepared_schema_version": training.PREPARED_SCHEMA_VERSION,
        "target_column": "Label + GoalsDistribution + OverUnder05/15/25/35",
        "trainable": True,
        "over_under_ready": True,
        "goals_distribution_ready": True,
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
        "split_policy": training.SPLIT_POLICY_VALIDATION_LAST_30,
        "training_start_year": training.INTERNATIONAL_TRAINING_START_YEAR,
    }
    calls = {"prepare": 0}

    def fake_prepare_training_dataset(force=False, refresh_history=False):
        calls["prepare"] += 1
        training.save_prepared_dataset(regenerated)
        return {"etl_ready": True}

    monkeypatch.setattr(training, "KAGGLE_ROOT", kaggle_root)
    monkeypatch.setattr(training, "PREPARED_DATASET_FILE", prepared_path)
    monkeypatch.setattr(training, "PREPARED_DATASET_META_FILE", meta_path)
    monkeypatch.setattr(training, "international_results_status", lambda: {"available": True, "source_path": str(tmp_path / "all_matches.csv")})
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


def test_build_training_matrix_dynamic_progress_is_batched(monkeypatch):
    total_rows = 5000
    rows = pd.DataFrame([
        {
            "Date": "2022-02-01",
            "Year": 2022,
            "Home": "Mexico" if index % 2 == 0 else "Canada",
            "Away": "Canada" if index % 2 == 0 else "Mexico",
            "HG": 1,
            "AG": 0,
            "Label": "H",
            "Source": "train",
        }
        for index in range(total_rows)
    ])
    model = training.WorldCupModel.from_history(pd.DataFrame(), teams=["Mexico", "Canada"])
    progress = []

    monkeypatch.setattr(training, "match_feature_row", lambda *args, **kwargs: {"bias": 1.0})

    training.build_training_matrix(
        rows,
        base_model=model,
        team_features=pd.DataFrame(),
        international_matches=pd.DataFrame(),
        progress_callback=progress.append,
    )

    assert len(progress) < total_rows // 25
    assert progress[0]["current"] == 1
    assert progress[-1]["current"] == total_rows
    assert {event["progress_every"] for event in progress} == {500}


def test_feature_progress_every_payload_override_is_clamped():
    assert training.feature_progress_every_from_payload({}, 5000) == 500
    assert training.feature_progress_every_from_payload({"feature_progress_every": 1000}, 5000) == 1000
    assert training.feature_progress_every_from_payload({"feature_progress_every": 25}, 5000) == 100
    assert training.feature_progress_every_from_payload({"feature_progress_every": 99999}, 5000) == 5000


def test_build_training_matrix_progress_every_override_controls_events(monkeypatch):
    total_rows = 3000
    rows = pd.DataFrame([
        {
            "Date": "2022-02-01",
            "Year": 2022,
            "Home": "Mexico",
            "Away": "Canada",
            "HG": 1,
            "AG": 0,
            "Label": "H",
            "Source": "train",
        }
        for _ in range(total_rows)
    ])
    model = training.WorldCupModel.from_history(pd.DataFrame(), teams=["Mexico", "Canada"])
    progress = []

    monkeypatch.setattr(training, "match_feature_row", lambda *args, **kwargs: {"bias": 1.0})

    training.build_training_matrix(
        rows,
        base_model=model,
        team_features=pd.DataFrame(),
        international_matches=pd.DataFrame(),
        progress_callback=progress.append,
        progress_every=1000,
    )

    assert [event["current"] for event in progress] == [1, 1000, 2000, 3000]
    assert {event["progress_every"] for event in progress} == {1000}


def test_build_training_matrix_recent15_cache_reuses_date_table(monkeypatch):
    rows = pd.DataFrame([
        {"Date": "2022-02-01", "Year": 2022, "Home": "Mexico", "Away": "Canada", "HG": 1, "AG": 0, "Label": "H", "Source": "train"},
        {"Date": "2022-02-01", "Year": 2022, "Home": "Brazil", "Away": "France", "HG": 2, "AG": 1, "Label": "H", "Source": "train"},
        {"Date": "2022-02-01", "Year": 2022, "Home": "Japan", "Away": "South Africa", "HG": 0, "AG": 0, "Label": "D", "Source": "train"},
        {"Date": "2022-02-01", "Year": 2022, "Home": "Canada", "Away": "Mexico", "HG": 0, "AG": 1, "Label": "A", "Source": "train"},
    ])
    teams = ["Mexico", "Canada", "Brazil", "France", "Japan", "South Africa"]
    model = training.WorldCupModel.from_history(pd.DataFrame(), teams=teams)
    cache = training.WorldCupFeatureBuildCache()

    monkeypatch.setattr(training, "match_feature_row", lambda *args, **kwargs: {"bias": 1.0})

    training.build_training_matrix(
        rows,
        base_model=model,
        teams=teams,
        team_features=pd.DataFrame(),
        international_matches=make_international_matches(30),
        feature_cache=cache,
    )

    assert cache.summary()["recent15_misses"] == 1
    assert cache.summary()["recent15_hits"] == rows.shape[0] - 1


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
