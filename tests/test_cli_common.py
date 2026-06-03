from argparse import Namespace

import pytest

from src.cli.common import CLIError, parse_eval_odd_range, parse_odd_range, validate_identifier
from src.cli.model_specs import build_model_params
from src.preprocessing.utils.target import TargetType


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


def test_build_dnn_params_omits_calibration():
    args = Namespace(
        target="over-under",
        normalizer="standard",
        sampler="none",
        calibrate=True,
        penalty=None,
        oas=None,
        decision_boundary=None,
        criterion=None,
        min_samples_leaf=None,
        min_samples_split=None,
        max_features=None,
        max_depth=None,
        class_weight=None,
        n_estimators=None,
        min_child_weight=None,
        learning_rate=None,
        lambda_regularization=None,
        alpha_regularization=None,
        n_neighbors=None,
        weights=None,
        p=None,
        algorithm=None,
        kernel=None,
        degree=None,
        gamma=None,
        hidden_layers=2,
        hidden_units=None,
        hidden_activation=None,
        vsn=False,
        layer_normalization=None,
        batch_normalization=None,
        dropout_rate=None,
        odd_noise_std=None,
        optimizer=None,
        lookahead=None,
        label_smoothing=None,
        batch_size=None,
        epochs=None,
        early_stopping_patience=None,
        lr_decay_patience=None,
        lr_decay_factor=None,
        verbose=None,
    )

    params = build_model_params(args=args, league_id="league", model_id="model", model_key="dnn")

    assert params["target_type"] == TargetType.OVER_UNDER
    assert "calibrate_probabilities" not in params
    assert params["hidden_layers"] == 2
    assert params["vsn"] is False

