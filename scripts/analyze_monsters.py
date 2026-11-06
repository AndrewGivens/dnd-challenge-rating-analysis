"""Summarize acquired monster data, fit an official-SRD baseline, and save figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


FEATURES = [
    "log_hit_points",
    "armor_class",
    "strength",
    "dexterity",
    "constitution",
    "intelligence",
    "wisdom",
    "charisma",
]
SOURCE_ORDER = ["srd-2014", "ccdx", "tob", "tob2", "tob3"]
SOURCE_LABELS = {
    "srd-2014": "2014 SRD",
    "ccdx": "Creature Codex",
    "tob": "Tome of Beasts",
    "tob2": "Tome of Beasts 2",
    "tob3": "Tome of Beasts 3",
}
GROUP_COLORS = {"Wizards of the Coast": "#3569a8", "Kobold Press": "#b65f32"}


def load_data(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    required = {"key", "name", "document_key", "challenge_rating", "hit_points", "armor_class"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input CSVs are missing required columns: {', '.join(missing)}")
    numeric = ["challenge_rating", "hit_points", "armor_class", "strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["publisher_name"] = frame["publisher_name"].fillna("Unknown")
    frame["source_group"] = np.where(
        frame["document_key"].eq("srd-2014"), "Wizards of the Coast", "Kobold Press"
    )
    frame["log_hit_points"] = np.log1p(frame["hit_points"])
    return frame


def fit_baseline(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    official = frame[frame["document_key"].eq("srd-2014")].copy()
    third_party = frame[~frame["document_key"].eq("srd-2014")].copy()
    if len(official) < 10 or third_party.empty:
        raise ValueError("Need at least 10 official SRD rows and some third-party rows")
    if official[FEATURES + ["challenge_rating"]].isna().any().any():
        raise ValueError("Model fields contain missing or invalid values; run validation and clean first")

    x_official = official[FEATURES]
    y_log = np.log1p(official["challenge_rating"])
    cv = KFold(n_splits=5, shuffle=True, random_state=386)
    cv_prediction_log = cross_val_predict(LinearRegression(), x_official, y_log, cv=cv)
    official_prediction = np.expm1(np.maximum(cv_prediction_log, 0))

    model = LinearRegression().fit(x_official, y_log)
    third_prediction = np.expm1(np.maximum(model.predict(third_party[FEATURES]), 0))
    official["predicted_cr"] = official_prediction
    third_party["predicted_cr"] = third_prediction
    official["prediction_method"] = "5-fold out-of-fold"
    third_party["prediction_method"] = "full official-baseline fit"
    frame_with_predictions = pd.concat([official, third_party]).sort_index()
    frame_with_predictions["cr_residual"] = (
        frame_with_predictions["challenge_rating"] - frame_with_predictions["predicted_cr"]
    )
    metrics = {
        "official_cv_r2_log_cr": float(r2_score(y_log, cv_prediction_log)),
        "official_cv_mae_cr": float(mean_absolute_error(official["challenge_rating"], official_prediction)),
    }
    return frame_with_predictions, metrics


def cr_bands(cr: pd.Series) -> pd.Series:
    bins = [-0.000001, 0, 1, 4, 10, 16, 30.000001]
    labels = ["0", "(0, 1]", "(1, 4]", "(4, 10]", "(10, 16]", "(16, 30]"]
    return pd.cut(cr, bins=bins, labels=labels, include_lowest=True, right=True)


def create_figures(frame: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # Share of monsters in each CR band helps compare the differently sized groups.
    plotted = frame.copy()
    plotted["cr_band"] = cr_bands(plotted["challenge_rating"])
    distribution = pd.crosstab(plotted["cr_band"], plotted["source_group"], normalize="columns")
    distribution = distribution.reindex(["0", "(0, 1]", "(1, 4]", "(4, 10]", "(10, 16]", "(16, 30]"])
    ax = distribution.T.plot(
        kind="bar",
        stacked=True,
        figsize=(8.5, 5.2),
        color=["#8b9ba8", "#638caa", "#4d9b8c", "#e0a04c", "#cf704c", "#9f5e7b"],
        width=0.66,
    )
    ax.set_title("Challenge-rating mix by publisher group")
    ax.set_xlabel("")
    ax.set_ylabel("Share of monsters")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2), labels=[f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
    ax.legend(title="Listed CR band", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "cr-distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Plot HP and AC against CR; log1p x positions preserve fractional low CRs.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True)
    for group, group_frame in frame.groupby("source_group", sort=False):
        color = GROUP_COLORS.get(group, "#707070")
        x = np.log1p(group_frame["challenge_rating"])
        axes[0].scatter(x, group_frame["hit_points"], s=14, alpha=0.30, color=color, label=group, edgecolors="none")
        jitter = np.random.default_rng(386).uniform(-0.12, 0.12, len(group_frame))
        axes[1].scatter(x, group_frame["armor_class"] + jitter, s=14, alpha=0.27, color=color, label=group, edgecolors="none")
    cr_ticks = np.array([0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 30])
    for ax in axes:
        ax.set_xticks(np.log1p(cr_ticks), labels=["0", "1/8", "1/4", "1/2", "1", "2", "4", "8", "16", "30"])
        ax.set_xlabel("Listed challenge rating (log1p-spaced axis)")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    axes[0].set_title("Hit points and CR")
    axes[0].set_ylabel("Hit points")
    axes[0].set_yscale("log")
    axes[1].set_title("Armor class and CR")
    axes[1].set_ylabel("Armor class")
    axes[1].set_yticks(range(5, 27, 5))
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Monster statistics by publisher group", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "stats-vs-cr.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Residuals are listed CR minus model prediction; positive values exceed the baseline estimate.
    order = [key for key in SOURCE_ORDER if key in set(frame["document_key"])]
    residual_values = [frame.loc[frame["document_key"].eq(key), "cr_residual"].dropna().to_numpy() for key in order]
    labels = [SOURCE_LABELS.get(key, key) for key in order]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    boxes = ax.boxplot(residual_values, tick_labels=labels, patch_artist=True, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.5})
    for i, key in enumerate(order):
        boxes["boxes"][i].set_facecolor("#6b91b3" if key == "srd-2014" else "#d78a62")
        boxes["boxes"][i].set_alpha(0.78)
    ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
    ax.set_title("Listed CR relative to the official-stat baseline")
    ax.set_ylabel("Listed CR − predicted CR")
    ax.set_xlabel("Source document")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.text(0.01, -0.22, "Positive residuals have higher listed CR than the model predicts. Official points use out-of-fold predictions; Kobold points use the model fitted to all official records.", transform=ax.transAxes, fontsize=8.5, color="#444444")
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline-residuals.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(frame: pd.DataFrame, metrics: dict[str, float]) -> dict[str, Any]:
    source_summary: dict[str, Any] = {}
    for source, subset in frame.groupby("document_key", sort=False):
        source_summary[str(source)] = {
            "n": int(len(subset)),
            "cr_median": float(subset["challenge_rating"].median()),
            "cr_mean": float(subset["challenge_rating"].mean()),
            "cr_q1": float(subset["challenge_rating"].quantile(0.25)),
            "cr_q3": float(subset["challenge_rating"].quantile(0.75)),
            "hp_median": float(subset["hit_points"].median()),
            "ac_median": float(subset["armor_class"].median()),
            "median_cr_residual": float(subset["cr_residual"].median()),
            "mean_cr_residual": float(subset["cr_residual"].mean()),
            "positive_residual_percent": float((subset["cr_residual"] > 0).mean() * 100),
        }

    group_summary: dict[str, Any] = {}
    for group, subset in frame.groupby("source_group", sort=False):
        group_summary[str(group)] = {
            "n": int(len(subset)),
            "cr_median": float(subset["challenge_rating"].median()),
            "cr_mean": float(subset["challenge_rating"].mean()),
            "hp_median": float(subset["hit_points"].median()),
            "ac_median": float(subset["armor_class"].median()),
            "median_cr_residual": float(subset["cr_residual"].median()),
            "mean_cr_residual": float(subset["cr_residual"].mean()),
            "positive_residual_percent": float((subset["cr_residual"] > 0).mean() * 100),
        }

    feature_correlations = {}
    for group, subset in frame.groupby("source_group", sort=False):
        feature_correlations[str(group)] = {
            "spearman_cr_hp": float(subset["challenge_rating"].corr(subset["hit_points"], method="spearman")),
            "spearman_cr_ac": float(subset["challenge_rating"].corr(subset["armor_class"], method="spearman")),
        }

    # Check how often third-party feature values extend beyond the official training ranges.
    official = frame[frame["document_key"].eq("srd-2014")]
    third_party = frame[~frame["document_key"].eq("srd-2014")]
    range_masks = []
    for feature in FEATURES:
        base_feature = "hit_points" if feature == "log_hit_points" else feature
        values = np.log1p(official[base_feature]) if feature == "log_hit_points" else official[base_feature]
        outside = (third_party[feature] < values.min()) | (third_party[feature] > values.max())
        range_masks.append(outside)
    outside_any = pd.concat(range_masks, axis=1).any(axis=1) if range_masks else pd.Series(False, index=third_party.index)

    return {
        "n_total": int(len(frame)),
        "source_counts": {str(key): int(value) for key, value in frame["document_key"].value_counts().items()},
        "group_summary": group_summary,
        "source_summary": source_summary,
        "official_model_cv": metrics,
        "official_training_ranges": {
            feature: {
                "min": float(official[feature].min()),
                "max": float(official[feature].max()),
            }
            for feature in FEATURES
        },
        "third_party_rows_outside_official_feature_range": int(outside_any.sum()),
        "third_party_rows_outside_official_feature_range_percent": float(outside_any.mean() * 100),
        "feature_correlations": feature_correlations,
        "interpretation": "Residuals are exploratory deviations from a simple model trained on official SRD monsters; they are not determinations of balance or correctness.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="acquired source CSVs; include srd-2014 and selected third-party sources")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    frame = load_data(args.csv)
    frame_with_predictions, metrics = fit_baseline(frame)
    create_figures(frame_with_predictions, args.figures_dir)
    summary = summarize(frame_with_predictions, metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    review_columns = ["key", "name", "document_key", "challenge_rating", "predicted_cr", "cr_residual", "prediction_method"]
    frame_with_predictions.assign(abs_residual=frame_with_predictions["cr_residual"].abs()).sort_values(
        "abs_residual", ascending=False
    )[review_columns].to_csv(args.output_dir / "residual_review.csv", index=False)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Saved figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
