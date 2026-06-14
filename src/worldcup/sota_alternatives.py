from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


ALTERNATIVES_BENCHMARK_PIPELINE_MODE = "alternatives_benchmark"
ALTERNATIVES_BENCHMARK_LABEL = "Benchmark alternativas"
ALTERNATIVES_EVIDENCE_POLICY = "papers_benchmarks"


ALTERNATIVE_BENCHMARK_SOURCES: List[Dict[str, Any]] = [
    {
        "rank": 1,
        "model_name": "CatBoost + pi-ratings",
        "family": "gradient_boosted_trees",
        "reported_better_than": ["2017 Soccer Prediction Challenge models", "goal-only baselines"],
        "metric": "Win/draw/loss predictive performance on validation splits",
        "reported_result": "Strong and stable validation performance using CatBoost with pi-ratings features.",
        "dataset_context": "2023 Soccer Prediction Challenge; recent five-year training window plus short pre-event update window.",
        "source_title": "Evaluating Soccer Match Prediction Models: A Deep Learning Approach and Feature Optimization for Gradient-Boosted Trees",
        "source_url": "https://arxiv.org/abs/2309.14807",
        "implementation_cost": "media",
        "data_requirements": ["match results", "pi-ratings or rating features", "temporal validation"],
        "limitations": ["Reported on challenge data, not specifically World Cup 2026.", "Requires rating feature reproduction."],
        "recommended_priority": "alta",
    },
    {
        "rank": 2,
        "model_name": "Gradient boosted trees with soccer-specific ratings",
        "family": "gradient_boosted_trees",
        "reported_better_than": ["typical goal-only model sets"],
        "metric": "Literature review of benchmark performance",
        "reported_result": "Review identifies CatBoost with pi-ratings as currently best-performing on datasets containing only goals as match features.",
        "dataset_context": "Survey chapter covering soccer match result prediction datasets, model families and evaluation.",
        "source_title": "Machine Learning for Soccer Match Result Prediction",
        "source_url": "https://arxiv.org/abs/2403.07669",
        "implementation_cost": "media",
        "data_requirements": ["match results", "rating features", "proper scoring evaluation"],
        "limitations": ["Survey-level evidence; exact advantage depends on dataset and feature construction."],
        "recommended_priority": "alta",
    },
    {
        "rank": 3,
        "model_name": "Random Forest + ranking ability parameters",
        "family": "random_forest_ensemble",
        "reported_better_than": ["Poisson regression", "standalone ranking methods"],
        "metric": "Predictive performance on training data before tournament simulation",
        "reported_result": "Ranking methods and random forests were best-performing, and combining random forest with ranking covariates improved predictive power substantially.",
        "dataset_context": "FIFA World Cups 2002-2014 used to predict and simulate FIFA World Cup 2018.",
        "source_title": "Prediction of the FIFA World Cup 2018 - A random forest approach with an emphasis on estimated team ability parameters",
        "source_url": "https://arxiv.org/abs/1806.03208",
        "implementation_cost": "media",
        "data_requirements": ["World Cup historical matches", "team covariates", "ranking or ability parameters"],
        "limitations": ["Training-data comparison; needs strict temporal validation before production use."],
        "recommended_priority": "alta",
    },
    {
        "rank": 4,
        "model_name": "Bayesian weighted dynamic goal models",
        "family": "bayesian_dynamic_goals",
        "reported_better_than": ["other discrete-time dynamic goal models"],
        "metric": "Predictive performance across recent league seasons",
        "reported_result": "Adaptive period-specific shrinkage reported better predictive performance than other discrete-time dynamic models.",
        "dataset_context": "Last five seasons of Bundesliga, Premier League and La Liga; implemented in footBayes.",
        "source_title": "Bayesian weighted discrete-time dynamic models for association football prediction",
        "source_url": "https://arxiv.org/abs/2508.05891",
        "implementation_cost": "alta",
        "data_requirements": ["longitudinal match results", "league periods", "Bayesian fitting stack"],
        "limitations": ["League evidence, not national-team tournament evidence.", "Heavier computation than current SOTA path."],
        "recommended_priority": "media",
    },
    {
        "rank": 5,
        "model_name": "Historical + bookmaker-odds Bayesian Poisson",
        "family": "odds_blended_bayesian_poisson",
        "reported_better_than": ["historical-only football score models"],
        "metric": "Predictive accuracy checks for the tenth season after a nine-year training set",
        "reported_result": "Scoring rates are blended from historical estimates and bookmaker odds to improve fit and predictive accuracy.",
        "dataset_context": "Nine-year dataset across popular European leagues, predicting the following season.",
        "source_title": "Combining historical data and bookmakers'odds in modelling football scores",
        "source_url": "https://arxiv.org/abs/1802.08848",
        "implementation_cost": "media",
        "data_requirements": ["historical scores", "pre-match bookmaker odds", "odds de-vigging"],
        "limitations": ["Depends on reliable pre-match odds availability.", "Still Poisson-family, but stronger input source."],
        "recommended_priority": "media",
    },
    {
        "rank": 6,
        "model_name": "Bayesian Bradley-Terry-Davidson ranking features",
        "family": "ranking_bayesian",
        "reported_better_than": ["FIFA ranking-only inputs in some tournament settings"],
        "metric": "Comparative predictive performance for international tournaments",
        "reported_result": "Alternative ranking summaries were evaluated inside statistical and ML models for World Cup and Africa Cup contexts.",
        "dataset_context": "2022 FIFA World Cup and 2023 CAF Africa Cup of Nations.",
        "source_title": "Alternative ranking measures to predict international football results",
        "source_url": "https://arxiv.org/abs/2405.10247",
        "implementation_cost": "media",
        "data_requirements": ["international match history", "paired-comparison rankings", "tournament-specific validation"],
        "limitations": ["Best use may be as features for another predictor rather than a standalone predictor."],
        "recommended_priority": "media",
    },
    {
        "rank": 7,
        "model_name": "Bayesian multinomial-Dirichlet outcome models",
        "family": "bayesian_multinomial",
        "reported_better_than": ["some standard probabilistic approaches"],
        "metric": "Proper scoring rules, error proportion and calibration assessment",
        "reported_result": "Reported competitive predictive power, good calibration and reasonable goodness of fit.",
        "dataset_context": "1710 Brazilian first-division matches.",
        "source_title": "Comparing probabilistic predictive models applied to football",
        "source_url": "https://arxiv.org/abs/1705.04356",
        "implementation_cost": "media",
        "data_requirements": ["match outcomes", "calibration evaluation", "proper scoring rules"],
        "limitations": ["Outcome-only; does not directly produce score distributions or totals markets."],
        "recommended_priority": "baja",
    },
    {
        "rank": 8,
        "model_name": "Neural networks and Random Forest vs Poisson",
        "family": "ml_comparison",
        "reported_better_than": ["Poisson approaches in selected single-match settings"],
        "metric": "Single-match prediction quality across five European top leagues",
        "reported_result": "Compares neural-network and random-forest approaches against Poisson and discusses possible improvements.",
        "dataset_context": "Five European top leagues, season-level match-result data.",
        "source_title": "Match predictions in soccer: Machine learning vs. Poisson approaches",
        "source_url": "https://arxiv.org/abs/2408.08331",
        "implementation_cost": "media",
        "data_requirements": ["season match results", "team-strength features", "temporal holdout"],
        "limitations": ["Paper reports only minor influence from exact feature/model choice; not a guaranteed upgrade."],
        "recommended_priority": "baja",
    },
]


BASELINE_CONTEXT_SOURCES: List[Dict[str, Any]] = [
    {
        "model_name": "Bradley-Terry extensions and hierarchical Poisson",
        "source_title": "Modeling outcomes of soccer matches",
        "source_url": "https://arxiv.org/abs/1807.01623",
        "note": "Useful caution: direct outcome models and hierarchical Poisson showed similar predictive behavior under temporal validation.",
    },
    {
        "model_name": "Weighted maximum-likelihood strength models",
        "source_title": "Ranking soccer teams on basis of their current strength: a comparison of maximum likelihood approaches",
        "source_url": "https://arxiv.org/abs/1705.09575",
        "note": "Useful baseline: independent and bivariate Poisson were best among the compared strength-ranking models.",
    },
]


def sota_alternatives_catalog() -> List[Dict[str, Any]]:
    return deepcopy(ALTERNATIVE_BENCHMARK_SOURCES)


def sota_baseline_context() -> List[Dict[str, Any]]:
    return deepcopy(BASELINE_CONTEXT_SOURCES)


def alternatives_table_rows(alternatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in alternatives:
        rows.append({
            "Rank": item.get("rank", ""),
            "Modelo": item.get("model_name", ""),
            "Familia": item.get("family", ""),
            "Prioridad": item.get("recommended_priority", ""),
            "Costo": item.get("implementation_cost", ""),
            "Mejor que": ", ".join(item.get("reported_better_than", [])),
            "Metrica": item.get("metric", ""),
            "Resultado reportado": item.get("reported_result", ""),
            "Contexto": item.get("dataset_context", ""),
            "Fuente": item.get("source_title", ""),
            "URL": item.get("source_url", ""),
        })
    return rows
