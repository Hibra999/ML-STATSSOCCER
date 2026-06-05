from argparse import Namespace
import warnings

import numpy as np
import pytest

from src.cli import app as cli_app
from src.cli.common import CLIError, parse_eval_odd_range, parse_odd_range, validate_identifier
from src.cli.model_specs import MODEL_SPECS, build_model_params, normalize_model_key
from src.preprocessing.utils.target import TargetType
from src.models.tuner import Tuner
from src.models.classifiers.boosting import _filter_ngboost_categorical_warning, sanitize_probabilities


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


def test_run_without_args_shows_web_help(capsys):
    assert cli_app.run([]) == 0
    captured = capsys.readouterr()
    assert "python app.py" in captured.out
    assert "chat" not in captured.out.lower()


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
