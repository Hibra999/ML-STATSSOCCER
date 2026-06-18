from __future__ import annotations

from importlib import import_module
from typing import Any


_CLASS_IMPORTS = {
    "DecisionTree": "src.models.classifiers.decisiontree",
    "DiscriminantAnalysisClassifier": "src.models.classifiers.discriminant",
    "KNN": "src.models.classifiers.knn",
    "LogisticRegressor": "src.models.classifiers.logistic",
    "NaiveBayes": "src.models.classifiers.naivebayes",
    "NeuralNetwork": "src.models.classifiers.neuralnets.nn",
    "RandomForest": "src.models.classifiers.randomforest",
    "SVM": "src.models.classifiers.svm",
    "XGBoost": "src.models.classifiers.extremeboosting",
}

__all__ = sorted(_CLASS_IMPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _CLASS_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
