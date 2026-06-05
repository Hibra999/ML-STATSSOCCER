import warnings

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor

from src.models.model import ClassificationModel
from src.preprocessing.utils.sampling import SamplerType
from src.preprocessing.utils.target import TargetType


PROBABILITY_EPSILON = 1e-12


def sanitize_probabilities(probabilities: np.ndarray, epsilon: float = PROBABILITY_EPSILON) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {probs.shape}.")

    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, epsilon, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    invalid_rows = ~np.isfinite(row_sums[:, 0]) | (row_sums[:, 0] <= 0.0)
    if invalid_rows.any():
        probs[invalid_rows] = 1.0 / probs.shape[1]
        row_sums = probs.sum(axis=1, keepdims=True)
    return probs / row_sums


def _filter_ngboost_categorical_warning():
    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in log",
        category=RuntimeWarning,
        module=r"ngboost\.distns\.categorical",
    )


def _estimator_feature_importances(estimator) -> np.ndarray:
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_)
    if hasattr(estimator, "get_feature_importance"):
        return np.asarray(estimator.get_feature_importance())
    raise ValueError("Este modelo no expone importancias de variables.")


class ProbabilitySanitizingModel(ClassificationModel):
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return sanitize_probabilities(super().predict_proba(df))


class NGBoost(ProbabilitySanitizingModel):
    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            normalizer: Optional[TransformerMixin] = None,
            sampler: Optional[SamplerType] = None,
            n_estimators: int = 300,
            max_depth: int = 3,
            learning_rate: float = 0.02,
            minibatch_frac: float = 1.0,
            natural_gradient: bool = True,
            calibrate_probabilities: bool = False,
            **kwargs
    ):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._minibatch_frac = minibatch_frac
        self._natural_gradient = natural_gradient
        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            normalizer=normalizer,
            sampler=sampler,
            calibrate_probabilities=calibrate_probabilities,
            **kwargs
        )

    def fit(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        with warnings.catch_warnings():
            _filter_ngboost_categorical_warning()
            return super().fit(train_df=train_df, eval_df=eval_df)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            _filter_ngboost_categorical_warning()
            return super().predict_proba(df)

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        try:
            from ngboost import NGBClassifier
            from ngboost.distns import k_categorical
        except ImportError as exc:
            raise RuntimeError("NGBoost no esta instalado. Ejecuta pip install -r requirements.txt.") from exc

        return NGBClassifier(
            Dist=k_categorical(num_classes),
            Base=DecisionTreeRegressor(max_depth=self._max_depth, random_state=0),
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            minibatch_frac=self._minibatch_frac,
            natural_gradient=self._natural_gradient,
            random_state=0,
            verbose=False,
        )

    def get_feature_importances(self) -> np.ndarray:
        estimator = self._classifier
        if self._calibrate_probabilities:
            estimator = self._classifier.calibrated_classifiers_[0].estimator
        return _estimator_feature_importances(estimator)

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == "n_estimators":
            return {"low": 100, "high": 800, "step": 100}
        if param == "max_depth":
            return {"low": 2, "high": 5, "step": 1}
        if param == "learning_rate":
            return {"low": 0.005, "high": 0.08, "step": 0.005}
        if param == "minibatch_frac":
            return {"low": 0.5, "high": 1.0, "step": 0.1}
        if param == "natural_gradient":
            return [True, False]
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "minibatch_frac": self._minibatch_frac,
            "natural_gradient": self._natural_gradient,
        })
        return model_config


class CatBoost(ProbabilitySanitizingModel):
    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            calibrate_probabilities: bool,
            normalizer: Optional[TransformerMixin] = None,
            sampler: Optional[SamplerType] = None,
            n_estimators: int = 300,
            max_depth: int = 6,
            learning_rate: float = 0.05,
            l2_leaf_reg: float = 3.0,
            random_strength: float = 1.0,
            **kwargs
    ):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._l2_leaf_reg = l2_leaf_reg
        self._random_strength = random_strength
        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=calibrate_probabilities,
            normalizer=normalizer,
            sampler=sampler,
            **kwargs
        )

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("CatBoost no esta instalado. Ejecuta pip install -r requirements.txt.") from exc

        return CatBoostClassifier(
            iterations=self._n_estimators,
            depth=self._max_depth,
            learning_rate=self._learning_rate,
            l2_leaf_reg=self._l2_leaf_reg,
            random_strength=self._random_strength,
            loss_function="MultiClass" if num_classes > 2 else "Logloss",
            random_seed=0,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    def get_feature_importances(self) -> np.ndarray:
        estimator = self._classifier
        if self._calibrate_probabilities:
            estimator = self._classifier.calibrated_classifiers_[0].estimator
        return _estimator_feature_importances(estimator)

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == "n_estimators":
            return {"low": 100, "high": 800, "step": 100}
        if param == "max_depth":
            return {"low": 3, "high": 10, "step": 1}
        if param == "learning_rate":
            return {"low": 0.01, "high": 0.3, "step": 0.01}
        if param == "l2_leaf_reg":
            return {"low": 1.0, "high": 10.0, "step": 0.5}
        if param == "random_strength":
            return {"low": 0.0, "high": 5.0, "step": 0.5}
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "l2_leaf_reg": self._l2_leaf_reg,
            "random_strength": self._random_strength,
        })
        return model_config


class LightGBM(ProbabilitySanitizingModel):
    def __init__(
            self,
            league_id: str,
            model_id: str,
            target_type: TargetType,
            calibrate_probabilities: bool,
            normalizer: Optional[TransformerMixin] = None,
            sampler: Optional[SamplerType] = None,
            n_estimators: int = 300,
            num_leaves: int = 31,
            max_depth: int = -1,
            learning_rate: float = 0.05,
            min_child_samples: int = 20,
            lambda_regularization: float = 0.0,
            alpha_regularization: float = 0.0,
            **kwargs
    ):
        self._n_estimators = n_estimators
        self._num_leaves = num_leaves
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._min_child_samples = min_child_samples
        self._lambda_regularization = lambda_regularization
        self._alpha_regularization = alpha_regularization
        super().__init__(
            league_id=league_id,
            model_id=model_id,
            target_type=target_type,
            calibrate_probabilities=calibrate_probabilities,
            normalizer=normalizer,
            sampler=sampler,
            **kwargs
        )

    def build_classifier(self, input_size: int, num_classes: int) -> BaseEstimator:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM no esta instalado. Ejecuta pip install -r requirements.txt.") from exc

        return LGBMClassifier(
            objective="multiclass" if num_classes > 2 else "binary",
            n_estimators=self._n_estimators,
            num_leaves=self._num_leaves,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            min_child_samples=self._min_child_samples,
            reg_lambda=self._lambda_regularization,
            reg_alpha=self._alpha_regularization,
            random_state=0,
            n_jobs=-1,
            verbosity=-1,
        )

    def get_feature_importances(self) -> np.ndarray:
        estimator = self._classifier
        if self._calibrate_probabilities:
            estimator = self._classifier.calibrated_classifiers_[0].estimator
        return _estimator_feature_importances(estimator)

    @classmethod
    def _get_suggested_param_values(cls, param: str) -> Union[List[Any], Dict[str, Any]]:
        if param == "n_estimators":
            return {"low": 100, "high": 800, "step": 100}
        if param == "num_leaves":
            return {"low": 15, "high": 127, "step": 8}
        if param == "max_depth":
            return {"low": -1, "high": 12, "step": 1}
        if param == "learning_rate":
            return {"low": 0.01, "high": 0.3, "step": 0.01}
        if param == "min_child_samples":
            return {"low": 5, "high": 80, "step": 5}
        if param == "lambda_regularization":
            return {"low": 0.0, "high": 5.0, "step": 0.5}
        if param == "alpha_regularization":
            return {"low": 0.0, "high": 5.0, "step": 0.5}
        raise ValueError(f'Undefined parameter: "{param}".')

    def _get_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        model_config.update({
            "n_estimators": self._n_estimators,
            "num_leaves": self._num_leaves,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "min_child_samples": self._min_child_samples,
            "lambda_regularization": self._lambda_regularization,
            "alpha_regularization": self._alpha_regularization,
        })
        return model_config
