import numpy as np
import pandas as pd
import pytest


def test_worldcup_metric_names_include_probabilistic_objectives():
    from src.worldcup import training

    assert training.normalize_metric_name("BalancedAccuracy") == "BalancedAccuracy"
    assert training.normalize_metric_name("balanced_accuracy") == "BalancedAccuracy"
    assert training.normalize_metric_name("LogLoss") == "LogLoss"
    assert training.normalize_metric_name("Brier") == "Brier"
    assert training.normalize_metric_name("PredictiveScore") == "PredictiveScore"


def test_worldcup_metric_score_handles_hard_and_probability_metrics():
    from src.worldcup import training

    y_true = pd.Series([0, 1, 1, 0])
    y_pred = np.asarray([0, 1, 0, 0])
    proba = np.asarray([
        [0.90, 0.10],
        [0.20, 0.80],
        [0.55, 0.45],
        [0.70, 0.30],
    ])

    assert training.metric_score(y_true, y_pred, "BalancedAccuracy") == pytest.approx(0.75)
    assert training.metric_score(y_true, y_pred, "LogLoss", y_proba=proba, classes=[0, 1]) < 0.0
    assert training.metric_score(y_true, y_pred, "Brier", y_proba=proba, classes=[0, 1]) < 0.0
    assert 0.0 <= training.metric_score(y_true, y_pred, "PredictiveScore", y_proba=proba, classes=[0, 1]) <= 1.0


def test_worldcup_optuna_uses_predict_proba_for_probability_objective(monkeypatch):
    optuna = pytest.importorskip("optuna")
    from src.worldcup import training

    calls = {"proba": 0}

    class FakeClassifier:
        def predict(self, x):
            return np.asarray([0 for _ in range(len(x))])

        def predict_proba(self, x):
            calls["proba"] += 1
            return np.tile(np.asarray([[0.65, 0.35]]), (len(x), 1))

    def fake_fit(**kwargs):
        return {"classifier": FakeClassifier(), "device": "cpu", "warnings": []}

    monkeypatch.setattr(training, "fit_configured_classifier", fake_fit)
    config = {
        "tuning_enabled": True,
        "model_type": "lightgbm",
        "model_profile": training.XG_LIGHTGBM_PROFILE,
        "params": training.worldcup_model_defaults("lightgbm"),
        "n_jobs": 1,
        "device": "cpu",
        "seed": 7,
        "n_trials": 1,
        "optuna_sampler": "random",
        "optuna_pruner": "none",
        "objective": "LogLoss",
        "tune_params": "n_estimators",
        "training_target": "over_under_25",
    }
    x_train = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [1, 1, 0, 0, 1, 0]})
    y_train = pd.Series([0, 1, 0, 1, 0, 1])
    x_validation = pd.DataFrame({"a": [6, 7, 8, 9], "b": [1, 0, 1, 0]})
    y_validation = pd.Series([0, 1, 0, 1])

    result = training.tune_model_if_requested(
        config,
        x_train,
        y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    assert result["enabled"] is True
    assert result["objective"] == "LogLoss"
    assert result["uses_predict_proba"] is True
    assert result["validation_source"] == "temporal_validation"
    assert calls["proba"] > 0
    assert optuna is not None


def test_worldcup_calibration_disables_when_classes_are_insufficient():
    from src.worldcup import training

    result = training.calibrate_classifier_if_requested(
        config={
            "calibration_enabled": True,
            "calibration_method": "sigmoid",
            "model_type": "lightgbm",
            "params": training.worldcup_model_defaults("lightgbm"),
            "n_jobs": 1,
            "device": "cpu",
            "seed": 7,
        },
        x_fit=pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
        y_fit=pd.Series([0, 0, 0]),
        sample_weight=None,
        classes=[0, 1],
    )

    assert result["enabled"] is True
    assert result["applied"] is False
    assert "clases suficientes" in result["warnings"][0]


def test_worldcup_feature_selection_supervised_falls_back_without_validation():
    from src.worldcup import training

    x_train = pd.DataFrame({
        "rating_home": [1.0, 2.0, 3.0],
        "rating_away": [3.0, 2.0, 1.0],
        **{f"noise_{index}": [float(index), 0.0, 1.0] for index in range(130)},
    })

    result = training.select_feature_columns_for_profile(
        feature_columns=list(x_train.columns),
        x_train=x_train,
        feature_profile=training.FEATURE_PROFILE_BALANCED,
        max_features=2,
        mode=training.FEATURE_SELECTION_SUPERVISED_MODEL,
        y_train=pd.Series([0, 1, 0]),
    )

    assert result["selected_mode"] == training.FEATURE_SELECTION_FAMILY_BALANCED
    assert result["fallback_used"] is True
    assert result["fallback_reason"] == "validation_unavailable"


def test_worldcup_shap_interpretability_payload_flags_market_and_redundancy():
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestClassifier

    from src.worldcup import training

    x_train = pd.DataFrame({
        "rating_diff": [-2.0, -2.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0],
        "rating_ratio": [-2.0, -2.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0],
        "market_home_prob": [0.10, 0.90, 0.20, 0.80, 0.30, 0.70, 0.40, 0.95],
        "noise": [0.0, 1.0, 0.5, 0.2, 0.1, 1.3, 0.7, 0.4],
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=12, max_depth=3, random_state=7)
    clf.fit(x_train, y_train)

    audit = training.shap_interpretability_payload(
        clf,
        list(x_train.columns),
        x_train,
        y_train,
        classes=[0, 1],
        target="over_under_25",
        enabled=True,
        sample_rows=64,
    )

    shap_payload = audit["shap"]
    assert shap_payload["applied"] is True
    assert shap_payload["validation_source"] == "temporal_validation_pre_test"
    assert shap_payload["top_features"]
    assert any(flag["feature"] == "market_home_prob" for flag in shap_payload["leakage_flags"])
    groups = training.shap_redundant_feature_groups(
        x_train,
        [
            {"feature": "rating_diff", "mean_abs_shap": 1.0},
            {"feature": "rating_ratio", "mean_abs_shap": 0.9},
        ],
    )
    assert any({"rating_diff", "rating_ratio"}.issubset(set(group["features"])) for group in groups)


def test_worldcup_xg_lightgbm_synthetic_bundle_keeps_test_as_final_report(tmp_path, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier

    from src.worldcup import training
    from src.worldcup.data import fallback_tournament_2026
    from tests.test_web_local import patch_worldcup_international, worldcup_international_sample

    monkeypatch.setattr(training, "WORLD_CUP_MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(training, "HYBRID_MODEL_FILE", tmp_path / "models" / "hybrid.pkl")
    monkeypatch.setattr(training, "HYBRID_MODEL_META_FILE", tmp_path / "models" / "hybrid.json")
    monkeypatch.setattr(training, "PREPARED_DATASET_FILE", tmp_path / "cache" / "prepared.pkl")
    monkeypatch.setattr(training, "PREPARED_DATASET_META_FILE", tmp_path / "cache" / "prepared.json")
    monkeypatch.setattr(training, "FEATURE_STORE_ROOT", tmp_path / "features")
    patch_worldcup_international(monkeypatch, training, worldcup_international_sample(rows=60))

    original_build = training.build_worldcup_classifier

    def fake_build_worldcup_classifier(model_key, params, n_jobs, device, seed, num_classes):
        if model_key == "lightgbm":
            return RandomForestClassifier(n_estimators=8, max_depth=4, random_state=seed, n_jobs=1)
        return original_build(model_key, params, n_jobs, device, seed, num_classes)

    monkeypatch.setattr(training, "build_worldcup_classifier", fake_build_worldcup_classifier)
    training.prepare_training_dataset(force=True)

    result = training.train_hybrid_model(
        fallback_tournament_2026(),
        payload={
            "model_type": "xg_lightgbm",
            "model_id": "xg-sklearn-test",
            "n_estimators": 8,
            "tuning_enabled": False,
            "calibration_enabled": True,
            "calibration_method": "sigmoid",
            "feature_selection_mode": "family_balanced",
            "seed": 17,
        },
    )

    expected_markets = {"result", "over_under_05", "over_under_15", "over_under_25", "over_under_35", "goals_distribution"}

    assert result["model"]["bundle"] is True
    assert result["model"]["model_profile"] == training.XG_LIGHTGBM_PROFILE
    assert result["eval_rows"] == result["model"]["markets"]["result"]["eval_rows"]
    assert set(result["model"]["markets"]) == expected_markets
    for market in expected_markets:
        summary = result["model"]["markets"][market]
        assert summary["metrics"]["eval"]["Brier"] >= 0.0
        assert summary["metrics"]["eval"]["LogLoss"] >= 0.0
        assert summary["metrics"]["eval"]["BalancedAccuracy"] >= 0.0
        assert summary["calibration"]["enabled"] is True
        assert summary["raw_metrics"]["eval"]["Brier"] >= 0.0
        assert summary["confusion_matrix"]["matrix"]
