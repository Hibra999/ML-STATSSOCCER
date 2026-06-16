#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SPLIT_ORDER = ["train", "validation", "test"]
OVER_MARKETS: List[Tuple[str, str, str]] = [
    ("over_under_05", "OverUnder05", "U/O 0.5"),
    ("over_under_15", "OverUnder15", "U/O 1.5"),
    ("over_under_25", "OverUnder25", "U/O 2.5"),
    ("over_under_35", "OverUnder35", "U/O 3.5"),
]
RESULT_ORDER = ["H", "D", "A"]
MIN_DRIFT_PERIOD_ROWS = 20


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind", context="notebook")

    prepared = load_pickle(Path(args.prepared))
    matches = prepared_matches(prepared)
    api = prepared.get("api_football", {}) if isinstance(prepared, dict) else {}
    model_meta = load_model_metadata(Path(args.models_root))

    figures = {
        "target_distribution": plot_target_distribution(matches, output_dir),
        "temporal_drift": plot_temporal_drift(matches, output_dir),
        "goal_distribution": plot_goal_distribution(matches, output_dir),
        "correlation_heatmap": plot_correlation_heatmap(matches, output_dir),
        "source_coverage": plot_source_coverage(prepared, api, matches, output_dir),
        "api_team_outcomes": plot_api_team_outcomes(api, output_dir),
        "market_errors": plot_market_errors(model_meta, output_dir),
    }
    findings = compute_findings(prepared, matches, api, model_meta, figures)
    (output_dir / "summary.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "findings.md").write_text(render_markdown(findings), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "figures": figures, "findings": str(output_dir / "findings.md")}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate seaborn EDA for the World Cup predictive dataset.")
    parser.add_argument("--prepared", default="storage/worldcup/cache/worldcup_training_prepared.pkl")
    parser.add_argument("--models-root", default="storage/worldcup/models")
    parser.add_argument("--output", default="outputs/worldcup_seaborn_eda")
    return parser.parse_args()


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepared_matches(prepared: Dict[str, Any]) -> pd.DataFrame:
    frames = []
    for split in SPLIT_ORDER:
        frame = prepared.get(split, pd.DataFrame())
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            working = frame.copy()
            working["split"] = split
            frames.append(working)
    if not frames:
        return pd.DataFrame()
    matches = pd.concat(frames, ignore_index=True)
    matches["Date"] = pd.to_datetime(matches.get("Date"), errors="coerce", utc=True).dt.tz_convert(None)
    for column in ["HG", "AG", *[column for _, column, _ in OVER_MARKETS], "sample_weight"]:
        if column in matches.columns:
            matches[column] = pd.to_numeric(matches[column], errors="coerce")
    matches["total_goals"] = matches["HG"].fillna(0.0) + matches["AG"].fillna(0.0)
    matches["goal_diff"] = matches["HG"].fillna(0.0) - matches["AG"].fillna(0.0)
    matches["abs_goal_diff"] = matches["goal_diff"].abs()
    matches["home_win"] = (matches["Label"] == "H").astype(int)
    matches["draw"] = (matches["Label"] == "D").astype(int)
    matches["away_win"] = (matches["Label"] == "A").astype(int)
    matches["is_worldcup_match"] = matches.get("is_worldcup_match", False).astype(bool).astype(int)
    matches["knockout"] = matches.get("knockout", False).astype(bool).astype(int)
    matches["quarter"] = matches["Date"].dt.to_period("Q").dt.to_timestamp()
    matches["year"] = matches["Date"].dt.year
    return matches


def save_current(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return str(path)


def split_palette() -> Dict[str, Any]:
    colors = sns.color_palette("colorblind", n_colors=len(SPLIT_ORDER))
    return dict(zip(SPLIT_ORDER, colors))


def plot_target_distribution(matches: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "01_target_distribution_by_split.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    sns.countplot(data=matches, x="split", hue="Label", order=SPLIT_ORDER, hue_order=RESULT_ORDER, ax=axes[0])
    axes[0].set(title="1X2 target distribution by temporal split", xlabel="", ylabel="Matches")

    over_long = matches.melt(
        id_vars=["split"],
        value_vars=[column for _, column, _ in OVER_MARKETS],
        var_name="market",
        value_name="over",
    )
    market_labels = {column: label for _, column, label in OVER_MARKETS}
    over_long["market"] = over_long["market"].map(market_labels)
    over_summary = over_long.groupby(["split", "market"], observed=True)["over"].mean().reset_index()
    over_summary["over_rate_pct"] = over_summary["over"] * 100.0
    sns.barplot(data=over_summary, x="market", y="over_rate_pct", hue="split", hue_order=SPLIT_ORDER, ax=axes[1])
    axes[1].set(title="Over rate by market and split", xlabel="", ylabel="Over rate (%)")
    axes[1].axhline(50, color="0.3", linewidth=1, linestyle="--")
    return save_current(path)


def plot_temporal_drift(matches: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "02_temporal_drift_quarterly.png"
    quarterly = matches.dropna(subset=["quarter"]).copy()
    if quarterly.empty:
        return empty_figure(path, "Temporal drift unavailable: no dates")
    period_counts = quarterly.groupby("quarter", observed=True).size().rename("rows").reset_index()
    quarterly = quarterly.merge(period_counts[period_counts["rows"] >= MIN_DRIFT_PERIOD_ROWS][["quarter"]], on="quarter", how="inner")
    if quarterly.empty:
        return empty_figure(path, f"Temporal drift unavailable: no periods with at least {MIN_DRIFT_PERIOD_ROWS} rows")
    result = quarterly.groupby("quarter", observed=True)[["home_win", "draw", "away_win"]].mean().reset_index().melt(id_vars="quarter", var_name="metric", value_name="rate")
    result["metric"] = result["metric"].map({"home_win": "Home win", "draw": "Draw", "away_win": "Away win"})
    over_cols = [column for _, column, _ in OVER_MARKETS]
    over = quarterly.groupby("quarter", observed=True)[over_cols].mean().reset_index().melt(id_vars="quarter", var_name="metric", value_name="rate")
    over["metric"] = over["metric"].map({column: label for _, column, label in OVER_MARKETS})
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    sns.lineplot(data=result, x="quarter", y="rate", hue="metric", marker="o", ax=axes[0])
    axes[0].set(title=f"Quarterly 1X2 drift (N >= {MIN_DRIFT_PERIOD_ROWS})", xlabel="", ylabel="Rate")
    axes[0].set_ylim(0, 1)
    sns.lineplot(data=over, x="quarter", y="rate", hue="metric", marker="o", ax=axes[1])
    axes[1].set(title=f"Quarterly over/under target drift (N >= {MIN_DRIFT_PERIOD_ROWS})", xlabel="Quarter", ylabel="Over rate")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
    return save_current(path)


def plot_goal_distribution(matches: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "03_goal_distribution_by_result.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=matches, x="total_goals", hue="split", hue_order=SPLIT_ORDER, bins=range(0, 13), multiple="dodge", ax=axes[0])
    axes[0].set(title="Total goals distribution by split", xlabel="Total goals", ylabel="Matches")
    sns.boxplot(data=matches, x="Label", y="total_goals", hue="split", order=RESULT_ORDER, hue_order=SPLIT_ORDER, ax=axes[1])
    axes[1].set(title="Total goals by 1X2 result", xlabel="Result", ylabel="Total goals")
    return save_current(path)


def plot_correlation_heatmap(matches: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "04_label_correlation_heatmap.png"
    columns = [
        "HG",
        "AG",
        "total_goals",
        "goal_diff",
        "abs_goal_diff",
        "home_win",
        "draw",
        "away_win",
        "OverUnder05",
        "OverUnder15",
        "OverUnder25",
        "OverUnder35",
        "is_worldcup_match",
        "knockout",
        "sample_weight",
    ]
    available = [column for column in columns if column in matches.columns]
    corr = matches[available].apply(pd.to_numeric, errors="coerce").corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlations among labels, split metadata and goal outcomes")
    return save_current(path)


def plot_source_coverage(prepared: Dict[str, Any], api: Dict[str, Any], matches: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "05_predictive_source_coverage.png"
    rows = []
    total = max(int(matches.shape[0]), 1)
    rows.append({"source": "prepared_matches", "rows": int(matches.shape[0]), "coverage_pct": 100.0})
    for key in ["market_rows", "api_football_fixture_rows", "api_football_stat_rows", "api_football_market_rows", "qualifier_feature_rows"]:
        rows.append({"source": key, "rows": int(prepared.get(key, 0) or 0), "coverage_pct": float(prepared.get(key, 0) or 0) / total * 100.0})
    for key in ["statistics", "lineups", "injuries", "odds", "market_rows"]:
        value = api.get(key, pd.DataFrame()) if isinstance(api, dict) else pd.DataFrame()
        if isinstance(value, pd.DataFrame):
            rows.append({"source": f"api_{key}", "rows": int(value.shape[0]), "coverage_pct": float(value.shape[0]) / total * 100.0})
    coverage = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=coverage, y="source", x="rows", ax=axes[0])
    axes[0].set(title="Available rows by predictive source", xlabel="Rows", ylabel="")
    sns.barplot(data=coverage, y="source", x="coverage_pct", ax=axes[1])
    axes[1].set(title="Coverage vs prepared match rows", xlabel="Coverage (%)", ylabel="")
    return save_current(path)


def plot_api_team_outcomes(api: Dict[str, Any], output_dir: Path) -> str:
    path = output_dir / "06_api_team_outcome_distributions.png"
    team_stats = api.get("team_stats", pd.DataFrame()) if isinstance(api, dict) else pd.DataFrame()
    if not isinstance(team_stats, pd.DataFrame) or team_stats.empty:
        return empty_figure(path, "API team outcomes unavailable")
    working = team_stats.copy()
    for column in ["GF", "GA", "GoalDiff", "Points", "Over25", "BTTS", "CleanSheet"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working["Outcome"] = np.select(
        [working.get("Win", 0).astype(float) > 0, working.get("Draw", 0).astype(float) > 0],
        ["Win", "Draw"],
        default="Loss",
    )
    long = working.melt(id_vars=["Outcome", "Side"], value_vars=[c for c in ["GF", "GA", "GoalDiff", "Points"] if c in working], var_name="metric", value_name="value")
    g = sns.catplot(data=long, x="Outcome", y="value", hue="Side", col="metric", kind="box", col_wrap=2, height=3.2, aspect=1.2, sharey=False)
    g.fig.suptitle("API Football team outcome distributions (available coverage only)", y=1.03)
    g.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(g.fig)
    return str(path)


def load_model_metadata(models_root: Path) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    if not models_root.exists():
        return output
    for path in sorted(models_root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data["_path"] = str(path)
            output.append(data)
    return output


def plot_market_errors(model_meta: List[Dict[str, Any]], output_dir: Path) -> str:
    path = output_dir / "07_market_error_metrics.png"
    metric_rows = []
    for meta in model_meta:
        markets = meta.get("markets", {}) if isinstance(meta, dict) else {}
        if not isinstance(markets, dict):
            continue
        for market, summary in markets.items():
            eval_metrics = ((summary or {}).get("metrics") or {}).get("eval", {})
            for metric in ["Accuracy", "F1", "BalancedAccuracy", "LogLoss", "Brier"]:
                if metric in eval_metrics:
                    metric_rows.append({"model": meta.get("model_id", ""), "market": market, "metric": metric, "value": eval_metrics[metric]})
    if not metric_rows:
        return empty_figure(path, "No trained World Cup market model metadata found; rerun after training to plot errors by market.")
    metrics = pd.DataFrame(metric_rows)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics, x="market", y="value", hue="metric")
    plt.xticks(rotation=25, ha="right")
    plt.title("Final report metrics by trained market")
    plt.xlabel("Market")
    plt.ylabel("Metric value")
    return save_current(path)


def empty_figure(path: Path, message: str) -> str:
    plt.figure(figsize=(9, 3))
    plt.axis("off")
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return save_current(path)


def compute_findings(prepared: Dict[str, Any], matches: pd.DataFrame, api: Dict[str, Any], model_meta: List[Dict[str, Any]], figures: Dict[str, str]) -> Dict[str, Any]:
    result_rates = pd.crosstab(matches["split"], matches["Label"], normalize="index").reindex(SPLIT_ORDER).fillna(0.0) * 100.0
    over_rates = matches.groupby("split", observed=True)[[column for _, column, _ in OVER_MARKETS]].mean().reindex(SPLIT_ORDER).fillna(0.0) * 100.0
    total_goals = matches.groupby("split", observed=True)["total_goals"].agg(["count", "mean", "median", "std"]).reindex(SPLIT_ORDER)
    drift = drift_summary(result_rates, over_rates)
    period_counts = matches.dropna(subset=["quarter"]).groupby("quarter", observed=True).size()
    corr_pairs = top_correlation_pairs(matches)
    api_team_stats = api.get("team_stats", pd.DataFrame()) if isinstance(api, dict) else pd.DataFrame()
    api_columns = list(api_team_stats.columns) if isinstance(api_team_stats, pd.DataFrame) else []
    model_market_count = sum(len((meta.get("markets") or {})) for meta in model_meta if isinstance(meta.get("markets"), dict))
    return {
        "dataset": {
            "rows": int(matches.shape[0]),
            "date_min": str(matches["Date"].min()) if not matches.empty else "",
            "date_max": str(matches["Date"].max()) if not matches.empty else "",
            "training_start_year": int(prepared.get("training_start_year", 2014) or 2014),
            "max_label_cutoff": str(prepared.get("max_label_cutoff", "")),
            "split_policy": str(prepared.get("split_policy", "")),
            "split_counts": {split: int((matches["split"] == split).sum()) for split in SPLIT_ORDER},
            "drift_period_min_rows": MIN_DRIFT_PERIOD_ROWS,
            "drift_periods_total": int(period_counts.shape[0]),
            "drift_periods_used": int((period_counts >= MIN_DRIFT_PERIOD_ROWS).sum()),
        },
        "result_rates_pct": dataframe_to_rounded_dict(result_rates),
        "over_rates_pct": dataframe_to_rounded_dict(over_rates),
        "total_goals_by_split": dataframe_to_rounded_dict(total_goals),
        "drift": drift,
        "top_correlations": corr_pairs,
        "coverage": {
            "market_rows": int(prepared.get("market_rows", 0) or 0),
            "api_football_fixture_rows": int(prepared.get("api_football_fixture_rows", 0) or 0),
            "api_football_stat_rows": int(prepared.get("api_football_stat_rows", 0) or 0),
            "api_football_market_rows": int(prepared.get("api_football_market_rows", 0) or 0),
            "qualifier_feature_rows": int(prepared.get("qualifier_feature_rows", 0) or 0),
            "api_team_stat_columns": api_columns,
            "has_xg_columns": any("xg" in column.lower() for column in api_columns),
            "has_shot_columns": any("shot" in column.lower() or "sot" in column.lower() for column in api_columns),
            "trained_market_models": model_market_count,
        },
        "figures": figures,
        "recommendations": recommendations(prepared, drift, corr_pairs, api_columns, model_market_count),
    }


def drift_summary(result_rates: pd.DataFrame, over_rates: pd.DataFrame) -> Dict[str, Any]:
    output: Dict[str, Any] = {"result_abs_pp_vs_train": {}, "over_abs_pp_vs_train": {}}
    if "train" not in result_rates.index:
        return output
    for split in ["validation", "test"]:
        if split in result_rates.index:
            diff = (result_rates.loc[split] - result_rates.loc["train"]).abs().sort_values(ascending=False)
            output["result_abs_pp_vs_train"][split] = dataframe_series_to_dict(diff)
        if split in over_rates.index:
            diff = (over_rates.loc[split] - over_rates.loc["train"]).abs().sort_values(ascending=False)
            output["over_abs_pp_vs_train"][split] = dataframe_series_to_dict(diff)
    return output


def top_correlation_pairs(matches: pd.DataFrame, limit: int = 12) -> List[Dict[str, Any]]:
    columns = [
        "HG",
        "AG",
        "total_goals",
        "goal_diff",
        "abs_goal_diff",
        "home_win",
        "draw",
        "away_win",
        "OverUnder05",
        "OverUnder15",
        "OverUnder25",
        "OverUnder35",
        "is_worldcup_match",
        "knockout",
        "sample_weight",
    ]
    available = [column for column in columns if column in matches.columns]
    corr = matches[available].apply(pd.to_numeric, errors="coerce").corr().abs()
    rows = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                rows.append({"left": left, "right": right, "abs_corr": round(float(value), 4)})
    return sorted(rows, key=lambda item: item["abs_corr"], reverse=True)[:limit]


def recommendations(prepared: Dict[str, Any], drift: Dict[str, Any], corr_pairs: List[Dict[str, Any]], api_columns: Iterable[str], model_market_count: int) -> List[str]:
    recs = []
    if int(prepared.get("market_rows", 0) or 0) == 0:
        recs.append("Gate or drop market_* and xg_vs_market_delta features for this run; market coverage is 0 rows.")
    if not any("xg" in str(column).lower() for column in api_columns):
        recs.append("Treat xG features as coverage indicators unless a real pre-match/as-of xG source is added; current API team stats do not expose xG columns.")
    if not any("shot" in str(column).lower() or "sot" in str(column).lower() for column in api_columns):
        recs.append("Treat shots/SOT quality features as sparse or zero-filled in this prepared dataset; add source coverage checks before selection.")
    result_drift = drift.get("result_abs_pp_vs_train", {}).get("test", {})
    if result_drift:
        largest = max(result_drift.items(), key=lambda item: item[1])
        if largest[1] >= 3.0:
            recs.append(f"Monitor temporal calibration for result class {largest[0]}; test drift vs train is {largest[1]:.2f} percentage points.")
    over_drift = drift.get("over_abs_pp_vs_train", {}).get("test", {})
    if over_drift:
        largest = max(over_drift.items(), key=lambda item: item[1])
        if largest[1] >= 3.0:
            recs.append(f"Use market-specific calibration for {largest[0]}; test over-rate drift vs train is {largest[1]:.2f} percentage points.")
    if model_market_count == 0:
        recs.append("Train xg_lightgbm and rerun this EDA to populate error-by-market plots from saved confusion matrices and Brier/LogLoss.")
    if any(pair["abs_corr"] >= 0.95 for pair in corr_pairs):
        recs.append("Use correlation pruning or SHAP redundancy checks; several label/goal variables are near-duplicates and similar redundancy can appear in engineered features.")
    return recs


def dataframe_to_rounded_dict(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    for index, row in frame.iterrows():
        output[str(index)] = {str(column): round(float(value), 4) if pd.notna(value) else 0.0 for column, value in row.items()}
    return output


def dataframe_series_to_dict(series: pd.Series) -> Dict[str, float]:
    return {str(index): round(float(value), 4) if pd.notna(value) else 0.0 for index, value in series.items()}


def render_markdown(findings: Dict[str, Any]) -> str:
    dataset = findings["dataset"]
    coverage = findings["coverage"]
    lines = [
        "# World Cup Seaborn EDA",
        "",
        "## Dataset",
        f"- Rows: {dataset['rows']}",
        f"- Date range: {dataset['date_min']} to {dataset['date_max']}",
        f"- Max label cutoff: {dataset['max_label_cutoff']}",
        f"- Split policy: {dataset['split_policy']}",
        f"- Split counts: {dataset['split_counts']}",
        f"- Drift periods used: {dataset['drift_periods_used']} of {dataset['drift_periods_total']} quarters with N >= {dataset['drift_period_min_rows']}",
        "",
        "## Coverage",
        f"- Market rows: {coverage['market_rows']}",
        f"- API Football fixtures: {coverage['api_football_fixture_rows']}",
        f"- API Football team stat rows: {coverage['api_football_stat_rows']}",
        f"- API Football market rows: {coverage['api_football_market_rows']}",
        f"- Has xG columns: {coverage['has_xg_columns']}",
        f"- Has shot/SOT columns: {coverage['has_shot_columns']}",
        f"- Trained market models with metadata: {coverage['trained_market_models']}",
        "",
        "## Largest Drift vs Train",
    ]
    for group, values in findings["drift"].items():
        lines.append(f"- {group}: {values}")
    lines.extend([
        "",
        "## Top Correlations",
    ])
    for item in findings["top_correlations"][:10]:
        lines.append(f"- {item['left']} vs {item['right']}: abs corr {item['abs_corr']}")
    lines.extend([
        "",
        "## Recommendations",
    ])
    for rec in findings["recommendations"]:
        lines.append(f"- {rec}")
    lines.extend([
        "",
        "## Figures",
    ])
    for name, path in findings["figures"].items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
