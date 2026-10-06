"""Visualization functions for experiment results.

Uses matplotlib with the Agg backend (no GUI required).
Generates multibar plots for classifier performance comparisons,
feature importance bar charts, and classifier parameter visualizations.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot styling
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
})

# Color palette for classifiers
CLF_COLORS = {
    "logreg": "#1f77b4",
    "random_forest": "#2ca02c",
    "gradient_boosting": "#ff7f0e",
    "gaussian_nb": "#d62728",
}

# Color palette for feature subsets
SUBSET_COLORS = {
    "full": "#1f77b4",
    "no_tcp_flags": "#aec7e8",
    "no_statistical": "#ff7f0e",
    "no_directional": "#2ca02c",
    "no_byte_counts": "#98df8a",
    "no_iat": "#d62728",
    "minimal": "#ff9896",
}


def plot_avg_performance_by_subset(
    results_df: pd.DataFrame,
    channel: str,
    output_path: str,
):
    """Multibar plot: average AUROC across feature sets for each classifier.

    X-axis = classifiers, Y-axis = AUROC, grouped bars = feature subsets.
    One plot per channel.

    Args:
        results_df: DataFrame with columns channel, feature_subset, classifier, mean_auroc.
        channel: Channel name to filter on.
        output_path: Path to save the PNG file.
    """
    channel_df = results_df[results_df["channel"] == channel].copy()
    if channel_df.empty:
        return

    classifiers = sorted(channel_df["classifier"].unique())
    subsets = sorted(channel_df["feature_subset"].unique())

    n_classifiers = len(classifiers)
    n_subsets = len(subsets)
    bar_width = 0.8 / n_subsets
    x = np.arange(n_classifiers)

    fig, ax = plt.subplots(figsize=(max(8, n_classifiers * 2), 5))

    for i, subset in enumerate(subsets):
        offset = (i - n_subsets / 2 + 0.5) * bar_width
        means = []
        stds = []
        for clf in classifiers:
            row = channel_df[
                (channel_df["classifier"] == clf)
                & (channel_df["feature_subset"] == subset)
            ]
            if not row.empty:
                means.append(row["mean_auroc"].values[0])
                stds.append(row["std_auroc"].values[0])
            else:
                means.append(0)
                stds.append(0)

        color = SUBSET_COLORS.get(subset, None)
        ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=2,
            label=subset, color=color,
            edgecolor="white", linewidth=0.3,
        )

    ax.set_xlabel("Classifier")
    ax.set_ylabel("Mean AUROC")
    ax.set_title(f"Average Performance by Feature Subset — {channel}")
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=min(n_subsets, 3), framealpha=0.9)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_performance_by_classifier(
    results_df: pd.DataFrame,
    channel: str,
    feature_subset: str,
    output_path: str,
):
    """Multibar plot: AUROC for each classifier within a feature subset.

    X-axis = classifiers, Y-axis = AUROC with error bars (std).

    Args:
        results_df: DataFrame with classifier results.
        channel: Channel name.
        feature_subset: Feature subset name.
        output_path: Path to save the PNG.
    """
    channel_df = results_df[
        (results_df["channel"] == channel)
        & (results_df["feature_subset"] == feature_subset)
    ].copy()

    if channel_df.empty:
        return

    classifiers = channel_df["classifier"].tolist()
    means = channel_df["mean_auroc"].values
    stds = channel_df["std_auroc"].values

    fig, ax = plt.subplots(figsize=(max(6, len(classifiers) * 1.5), 4.5))

    colors = [CLF_COLORS.get(c, "#888888") for c in classifiers]
    bars = ax.bar(
        range(len(classifiers)), means, yerr=stds,
        capsize=3, color=colors, edgecolor="white", linewidth=0.3,
    )

    ax.set_xlabel("Classifier")
    ax.set_ylabel("Mean AUROC (5-fold CV)")
    ax.set_title(f"Classifier Performance — {channel} / {feature_subset}")
    ax.set_xticks(range(len(classifiers)))
    ax.set_xticklabels(classifiers, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Annotate bars with values
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    channel: str,
    classifier: str,
    feature_subset: str,
    output_path: str,
):
    """Bar plot: top-N feature importance for a channel x classifier.

    Args:
        importance_df: DataFrame with columns feature, importance, rank.
        channel: Channel name.
        classifier: Classifier name.
        feature_subset: Feature subset name.
        output_path: Path to save the PNG.
    """
    if importance_df.empty:
        return

    # Sort by rank (ascending importance)
    df = importance_df.sort_values("rank").copy()
    df = df.iloc[::-1]  # Reverse so highest importance is on top

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.25)))
    ax.barh(
        df["feature"], df["importance"],
        color=CLF_COLORS.get(classifier, "#1f77b4"),
        edgecolor="white", linewidth=0.3,
    )
    ax.set_xlabel("Importance Score")
    ax.set_title(
        f"Feature Importance — {channel} / {classifier} / {feature_subset}",
        fontsize=10,
    )
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_classifier_params(
    classifier_name: str,
    best_params: dict,
    mean_auroc: float,
    std_auroc: float,
    n_positive: int,
    n_negative: int,
    output_path: str,
):
    """Visualize classifier parameters for the top-performing classifier.

    Creates a table-style figure showing the classifier name, its
    hyperparameters, and the AUROC score achieved.

    Args:
        classifier_name: Name of the classifier.
        best_params: Dictionary of hyperparameters (from estimator.get_params()).
        mean_auroc: Mean AUROC score.
        std_auroc: Std of AUROC.
        n_positive: Number of positive samples.
        n_negative: Number of negative samples.
        output_path: Path to save the PNG.
    """
    # Filter to the most relevant parameters
    relevant_keys = [
        "n_estimators", "max_depth", "min_samples_leaf",
        "max_iter", "C", "penalty", "solver",
        "learning_rate", "n_neighbors", "weights",
        "variance_smoothing", "var_smoothing",
    ]
    displayed_params = {
        k: v for k, v in best_params.items()
        if k in relevant_keys and v is not None
    }

    if not displayed_params:
        displayed_params = {
            k: v for k, v in best_params.items()
            if not k.startswith("_") and not callable(v)
        }

    # Build table rows
    rows = []
    for k, v in displayed_params.items():
        v_str = str(v)
        if len(v_str) > 50:
            v_str = v_str[:47] + "..."
        rows.append([k, v_str])

    fig, ax = plt.subplots(figsize=(7, max(3, len(rows) * 0.35 + 1.5)))
    ax.axis("off")
    ax.set_title(
        f"Best Classifier Parameters — {classifier_name}",
        fontsize=13, fontweight="bold", pad=10,
    )

    # Performance info box
    info_text = (
        f"Mean AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}\n"
        f"Samples: {n_positive + n_negative} "
        f"(HCS: {n_positive}, TGEN: {n_negative})"
    )
    ax.text(
        0.5, 0.95, info_text,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    # Parameter table
    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=["Parameter", "Value"],
            cellLoc="left",
            loc="center",
            bbox=[0.1, 0.1, 0.8, 0.75],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)

        # Style the header
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
