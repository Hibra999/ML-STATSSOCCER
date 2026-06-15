from argparse import Namespace
import warnings

import numpy as np
import pandas as pd
import pytest

from src.cli import app as cli_app
from src.cli.common import CLIError, parse_eval_odd_range, parse_odd_range, save_figure, validate_identifier
from src.cli.model_specs import MODEL_SPECS, build_model_params, normalize_model_key
from src.models.trainer import Trainer
from src.preprocessing.utils.target import TargetType, construct_targets
from src.models.tuner import Tuner
from src.models.classifiers.boosting import _filter_ngboost_categorical_warning, sanitize_probabilities
from src.network.leagues.downloaders.downloader import FootballDataDownloader
from src.network.leagues.league import League


def test_parse_odd_range_accepts_min_max():
    assert parse_odd_range("1.25:2.00") == (1.25, 2.0)


def test_parse_odd_range_disables_open_max_at_ten():
    assert parse_odd_range("1.0:10.0") is None
    assert parse_odd_range("1.5:10.0") == (1.5, 1000.0)


def test_parse_eval_odd_range():
    assert parse_eval_odd_range("1:1.31:1.60") == ("1", 1.31, 1.60)
    assert parse_eval_odd_range(None) == "None"


def test_validate_identifier_rejects_spaces():
    with pytest.raises(CLIError):
        validate_identifier("bad id", "model id")


def test_save_figure_writes_polished_matplotlib_output(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.barh(["recent15_adjusted_goal_diff_avg", "market_prob_home"], [0.72, 0.48])
    ax.set_title("Feature importance")

    output = tmp_path / "importance.png"
    save_figure(ax, str(output))

    assert output.exists()
    assert output.stat().st_size > 0


def test_run_without_args_shows_web_help(capsys):
    assert cli_app.run([]) == 0
    captured = capsys.readouterr()
    assert "python app.py" in captured.out
    assert "chat" not in captured.out.lower()


def test_predict_cli_no_longer_exposes_manual(capsys):
    parser = cli_app.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "--help"])

    captured = capsys.readouterr()
    assert "fixtures" in captured.out
    assert "manual" not in captured.out.lower()


def test_model_aliases_only_resolve_supported_boosting_models():
    assert normalize_model_key("ngb") == "ngboost"
    assert normalize_model_key("cat") == "catboost"
    assert normalize_model_key("lgbm") == "lightgbm"
    assert normalize_model_key("xgb") == "xgboost"
    assert set(MODEL_SPECS) == {"ngboost", "catboost", "lightgbm", "xgboost"}


@pytest.mark.parametrize("model_key", ["ngboost", "catboost", "lightgbm", "xgboost"])
def test_build_boosting_model_params(model_key):
    args = Namespace(
        target="over-under",
        normalizer="standard",
        sampler="none",
        calibrate=True,
        max_depth=None,
        n_estimators=None,
        min_child_weight=None,
        learning_rate=None,
        lambda_regularization=None,
        alpha_regularization=None,
        num_leaves=None,
        min_child_samples=None,
        minibatch_frac=None,
        natural_gradient=None,
        l2_leaf_reg=None,
        random_strength=None,
    )

    params = build_model_params(args=args, league_id="league", model_id="model", model_key=model_key)

    assert params["target_type"] == TargetType.OVER_UNDER
    assert params["league_id"] == "league"
    assert params["model_id"] == "model"
    if model_key == "ngboost":
        assert "calibrate_probabilities" not in params
    else:
        assert params["calibrate_probabilities"] is True


def test_tuner_builds_configurable_sampler_and_pruner():
    assert Tuner._build_sampler("random").__class__.__name__ == "RandomSampler"
    assert Tuner._build_sampler("tpe").__class__.__name__ == "TPESampler"
    assert Tuner._build_pruner("median").__class__.__name__ == "MedianPruner"
    assert Tuner._build_pruner("successive-halving").__class__.__name__ == "SuccessiveHalvingPruner"


def test_temporal_cross_validation_trains_only_on_past_rows_and_clones_model():
    class RecordingModel:
        target_type = TargetType.RESULT
        records = []

        def __init__(self):
            self.fitted = False

        def fit(self, train_df, eval_df):
            self.fitted = True
            self.__class__.records.append({
                "train_end": train_df["Date"].max(),
                "eval_start": eval_df["Date"].min(),
                "model_id": id(self),
            })
            return pd.DataFrame({
                "Accuracy": [1.0, 1.0],
                "F1": [1.0, 1.0],
                "Precision": [1.0, 1.0],
                "Recall": [1.0, 1.0],
                "data": ["train", "eval"],
            })

    rows = []
    labels = ["H", "D", "A"]
    for index in range(12):
        rows.append({
            "Date": f"2024-01-{index + 1:02d}",
            "Home": f"H{index}",
            "Away": f"A{index}",
            "Result": labels[index % 3],
            "HG": 1,
            "AG": 0,
        })
    df = pd.DataFrame(reversed(rows)).reset_index(drop=True)
    model = RecordingModel()

    metrics = Trainer().cross_validation(model=model, df=df, k_folds=3)

    assert model.fitted is False
    assert metrics["Fold"].nunique() == 3
    assert len(RecordingModel.records) == 3
    assert len({record["model_id"] for record in RecordingModel.records}) == 3
    assert all(record["train_end"] < record["eval_start"] for record in RecordingModel.records)


def test_dataset_masks_use_persisted_split_keys_after_dataset_updates():
    df = pd.DataFrame([
        {"Date": "2024-01-05", "Season": 2024, "Home": "H5", "Away": "A5"},
        {"Date": "2024-01-04", "Season": 2024, "Home": "H4", "Away": "A4"},
        {"Date": "2024-01-03", "Season": 2024, "Home": "H3", "Away": "A3"},
        {"Date": "2024-01-02", "Season": 2024, "Home": "H2", "Away": "A2"},
        {"Date": "2024-01-01", "Season": 2024, "Home": "H1", "Away": "A1"},
    ])
    train_df = df.iloc[2:].reset_index(drop=True)
    eval_df = df.iloc[:2].reset_index(drop=True)
    split = cli_app._build_split_metadata(df=df, train_df=train_df, eval_df=eval_df, eval_size=40.0)
    updated_df = pd.concat([
        pd.DataFrame([{"Date": "2024-01-06", "Season": 2024, "Home": "H6", "Away": "A6"}]),
        df,
    ], ignore_index=True)

    masks = cli_app._dataset_masks(updated_df, {"train": {"eval_samples_size": 40.0, "split": split}})

    assert masks["Eval"].tolist() == [False, True, True, False, False, False]
    assert masks["Train"].tolist() == [False, False, False, True, True, True]


def test_downloader_returns_none_without_touching_empty_download():
    class EmptyDownloader(FootballDataDownloader):
        def _get_additional_columns(self):
            return []

        def _download_dataframe(self, league, start_year):
            return None

        def _preprocess_dataframe(self, df, start_year):
            raise AssertionError("preprocess should not run for a missing download")

    league = League(country="Test", name="League", start_year=2024, category="main", url="", fixture="")

    assert EmptyDownloader().download(league=league, start_year=2024) is None


def test_sanitize_probabilities_clips_and_renormalizes():
    probabilities = np.array([
        [0.0, 0.0, 0.0],
        [np.nan, np.inf, -np.inf],
        [0.2, 0.3, 0.5],
    ])

    sanitized = sanitize_probabilities(probabilities)

    assert np.isfinite(sanitized).all()
    assert (sanitized > 0).all()
    assert np.allclose(sanitized.sum(axis=1), 1.0)


def test_ngboost_warning_filter_is_specific():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _filter_ngboost_categorical_warning()
        warnings.warn_explicit(
            "divide by zero encountered in log",
            RuntimeWarning,
            filename="categorical.py",
            lineno=13,
            module="ngboost.distns.categorical",
        )
        warnings.warn("other warning", RuntimeWarning)

    assert len(caught) == 1
    assert str(caught[0].message) == "other warning"


def test_tuner_emits_progress_payloads():
    progress = []
    tuner = Tuner(
        model_cls=object,
        fixed_params={},
        tunable_params={"x": [1]},
        df=None,
        metric="Accuracy",
        progress_callback=progress.append,
    )
    tuner._total_trials = 2

    class Trial:
        number = 0
        value = 0.75
        state = type("State", (), {"name": "COMPLETE"})()

    class Study:
        best_value = 0.75
        best_trial = type("BestTrial", (), {"number": 0})()

    tuner._emit_progress({
        "stage": "tuning",
        "current": 0,
        "total": 2,
        "percent": 0,
        "message": "Iniciando Optuna",
    })
    tuner._on_trial_complete(Study(), Trial())

    assert progress[0]["percent"] == 0
    assert progress[1]["current"] == 1
    assert progress[1]["total"] == 2
    assert progress[1]["percent"] == 50
    assert progress[1]["best_value"] == 0.75


def test_construct_targets_result_without_warnings():
    df = pd.DataFrame({"Result": ["H", "D", "A"]})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        targets = construct_targets(df=df, target_type=TargetType.RESULT)

    assert targets.tolist() == [0, 1, 2]


def test_construct_targets_over_under_without_warnings():
    df = pd.DataFrame({"HG": [1, 2, 0], "AG": [0, 1, 3]})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        targets = construct_targets(df=df, target_type=TargetType.OVER_UNDER)

    assert targets.tolist() == [0, 1, 1]


def test_construct_targets_rejects_unknown_result():
    df = pd.DataFrame({"Result": ["H", "W"]})

    with pytest.raises(ValueError, match="Expected"):
        construct_targets(df=df, target_type=TargetType.RESULT)


@pytest.mark.parametrize("model_key", ["ngboost", "catboost", "lightgbm", "xgboost"])
def test_three_optuna_trials_per_boosting_model_without_warnings(model_key):
    spec = MODEL_SPECS[model_key]
    params = _small_model_params(model_key)
    progress = []
    tuner = Tuner(
        model_cls=spec.model_cls,
        fixed_params=params,
        tunable_params={"learning_rate": [params["learning_rate"], params["learning_rate"] * 1.5]},
        df=_warning_free_training_df(),
        metric="Accuracy",
        sampler="random",
        pruner="none",
        progress_callback=progress.append,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        study = tuner.tune(trials=3)

    assert len(study.trials) == 3
    assert progress[-1]["current"] == 3
    assert progress[-1]["percent"] == 100


def _warning_free_training_df() -> pd.DataFrame:
    rows = []
    labels = ["H", "D", "A"]
    for index in range(60):
        label = labels[index % len(labels)]
        home_goals = {"H": 2, "D": 1, "A": 0}[label]
        away_goals = {"H": 0, "D": 1, "A": 2}[label]
        rows.append({
            "Date": f"2024-01-{(index % 28) + 1:02d}",
            "Season": 2024,
            "Week": index + 1,
            "Home": f"Home {index % 6}",
            "Away": f"Away {index % 6}",
            "HG": home_goals,
            "AG": away_goals,
            "Result": label,
            "Result-U/O": "O" if home_goals + away_goals >= 3 else "U",
            "HST": 0,
            "AST": 0,
            "HC": 0,
            "AC": 0,
            "1": 1.4 + (index % 5) * 0.08,
            "X": 2.8 + (index % 4) * 0.12,
            "2": 1.7 + (index % 6) * 0.09,
            "home_form": float(index % 3),
            "away_form": float((index + 1) % 3),
        })
    return pd.DataFrame(rows)


def _small_model_params(model_key: str):
    params = {
        "league_id": "warning-test",
        "model_id": f"{model_key}-warning-test",
        "target_type": TargetType.RESULT,
        "normalizer": None,
        "sampler": None,
        **MODEL_SPECS[model_key].defaults,
    }
    if MODEL_SPECS[model_key].supports_calibration:
        params["calibrate_probabilities"] = False
    params["n_estimators"] = 5
    params["max_depth"] = 2
    params["learning_rate"] = 0.02 if model_key == "ngboost" else 0.05
    if model_key == "lightgbm":
        params["num_leaves"] = 7
        params["min_child_samples"] = 1
    return params
