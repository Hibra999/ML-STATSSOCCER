import numpy as np
import pandas as pd
import pytest

from src.worldcup import training
from src.worldcup import international_provider


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
