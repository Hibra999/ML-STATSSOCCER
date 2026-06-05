from argparse import Namespace

import pytest

from src.cli import app as cli_app
from src.cli.common import CLIError, parse_eval_odd_range, parse_odd_range, validate_identifier
from src.cli.model_specs import MODEL_SPECS, build_model_params, normalize_model_key
from src.preprocessing.utils.target import TargetType
from src.models.tuner import Tuner


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
