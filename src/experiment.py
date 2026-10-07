"""Experiment runner: orchestrates classifier evaluation across channels,
feature subsets, and classifiers.

Loads experiment plans from JSON, runs stratified k-fold CV, collects
results with train/test sample counts, and produces visualizations.
"""

import json
import os
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone

from .classifiers import CLASSIFIER_NAMES, get_classifier
from .data_prep import get_available_channels, handle_missing_values, prepare_dataset
from .features import FEATURE_SUBSET_NAMES, get_feature_columns, get_all_feature_columns
from .feature_importance import extract_feature_importance
from .plotting import (
    plot_avg_performance_by_subset,
    plot_performance_by_classifier,
    plot_feature_importance,
    plot_classifier_params,
)


@dataclass
class ExperimentResult:
    """Result of a single experiment evaluation run."""

    channel: str
    feature_subset: str
    classifier: str
    fold_scores: list[float] = field(default_factory=list)
    mean_auroc: float = 0.0
    std_auroc: float = 0.0
    n_folds: int = 0
    n_positive: int = 0
    n_negative: int = 0
    train_n_positive: int = 0
    train_n_negative: int = 0
    test_n_positive: int = 0
    test_n_negative: int = 0
    warnings: list[str] = field(default_factory=list)


def load_experiment_plan(plan_path: str) -> dict:
    """Load an experiment plan from a JSON file.

    Args:
        plan_path: Path to the JSON experiment plan.

    Returns:
        Dict with keys: name, description, channels, pooled, feature_subsets,
            classifiers, n_splits, max_samples, top_n_features.
    """
    with open(plan_path) as f:
        plan = json.load(f)
    return plan


def default_experiment_plan() -> dict:
    """Return the default experiment plan configuration."""
    return {
        "name": "default_experiment",
        "description": "Full analysis: all channels, all subsets, all classifiers",
        "pooled": True,
        "channels": ["racetunnel", "sky", "obfs", "iodine", "mastodon"],
        "feature_subsets": FEATURE_SUBSET_NAMES,
        "classifiers": CLASSIFIER_NAMES,
        "n_splits": 5,
        "max_samples": 50000,
        "top_n_features": 20,
    }


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    classifier_name: str,
    n_splits: int,
) -> tuple[list[float], int, str | None]:
    """Run stratified k-fold CV and return fold scores.

    Args:
        X: Feature matrix.
        y: Binary labels.
        classifier_name: Classifier key.
        n_splits: Requested number of folds.

    Returns:
        Tuple of (fold_scores, actual_folds, warning).
    """
    n_positive = int(y.sum())
    n_negative = int((y == 0).sum())
    min_class = min(n_positive, n_negative)
    actual_folds = min(n_splits, min_class)

    if actual_folds < 2:
        return [], 0, f"Only {n_positive} positive / {n_negative} negative; cannot CV"

    warning = None
    if actual_folds < n_splits:
        warning = f"Reduced to {actual_folds}-fold CV (min class: {min_class})"

    clf = get_classifier(classifier_name)
    skf = StratifiedKFold(
        n_splits=actual_folds, shuffle=True, random_state=42
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                clf, X, y, cv=skf, scoring="roc_auc", n_jobs=-1
            )
        return scores.tolist(), actual_folds, warning
    except Exception as e:
        return [], 0, f"CV failed: {e}"


def compute_train_test_counts(
    n_positive: int,
    n_negative: int,
    n_folds: int,
) -> tuple[int, int, int, int]:
    """Compute approximate train/test sample counts from CV.

    With k-fold CV: train gets (k-1)/k of each class, test gets 1/k.

    Args:
        n_positive: Total positive samples.
        n_negative: Total negative samples.
        n_folds: Number of CV folds.

    Returns:
        Tuple of (train_pos, train_neg, test_pos, test_neg).
    """
    if n_folds == 0:
        return 0, 0, 0, 0
    train_pos = round(n_positive * (n_folds - 1) / n_folds)
    train_neg = round(n_negative * (n_folds - 1) / n_folds)
    test_pos = n_positive - train_pos
    test_neg = n_negative - train_neg
    return train_pos, train_neg, test_pos, test_neg


def run_experiment(
    df: pd.DataFrame,
    plan: dict,
    results_dir: str,
) -> pd.DataFrame:
    """Run a complete experiment based on a plan.

    Args:
        df: Full flows DataFrame.
        plan: Experiment plan dict.
        results_dir: Directory to save results.

    Returns:
        DataFrame with all experiment results.
    """
    channels = ["pooled"] if plan.get("pooled") else []
    channels += plan.get("channels", [])
    tgen_types = plan.get("tgen_types", [])
    feature_subsets = plan.get("feature_subsets", FEATURE_SUBSET_NAMES)
    classifiers = plan.get("classifiers", CLASSIFIER_NAMES)
    n_splits = plan.get("n_splits", 5)
    max_samples = plan.get("max_samples", 50000)
    top_n = plan.get("top_n_features", 20)
    skip_importance = plan.get("skip_importance", False)
    skip_plots = plan.get("skip_plots", False)
    balanced = plan.get("balanced", False)

    from .data_prep import get_available_tgen_types

    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    imp_dir = os.path.join(results_dir, "feature_importance")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(imp_dir, exist_ok=True)

    # Build the list of evaluation targets
    results: list[ExperimentResult] = []
    evaluation_targets = []

    if plan.get("pooled"):
        evaluation_targets.append(("pooled", None, None))

    for ch in channels:
        if ch == "pooled":
            continue
        evaluation_targets.append((ch, ch, None))

    if tgen_types:
        available_tgen = get_available_tgen_types(df)
        for tt in tgen_types:
            tt_resolved = None
            for at in available_tgen:
                if tt.lower() == at.lower():
                    tt_resolved = at
                    break
            if not tt_resolved and (tt.upper() in available_tgen):
                tt_resolved = tt.upper()
            if not tt_resolved and (tt in available_tgen):
                tt_resolved = tt
            if tt_resolved:
                evaluation_targets.append((f"TGEN_{tt_resolved}", None, tt_resolved))

    # Handle paired HCS/TGEN evaluations
    paired_evals = plan.get("paired_evaluations", [])
    if not paired_evals:
        # Try loading from default mapping file
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mapping_file = os.path.join(repo_root, "tgen_hcs_map.json")
        if os.path.exists(mapping_file):
            with open(mapping_file) as f:
                paired_evals = json.load(f)

    for pair in paired_evals:
        hcs_channel = pair.get("hcs_channel") or pair.get("hcs")
        tgen_type = pair.get("tgen_type") or pair.get("tgen")
        if hcs_channel and tgen_type:
            # Normalize to strings for label generation
            hcs_str = hcs_channel if isinstance(hcs_channel, str) else "+".join(hcs_channel)
            tgen_str = tgen_type if isinstance(tgen_type, str) else "+".join(tgen_type)
            target_label = f"HCS_{hcs_str}_vs_TGEN_{tgen_str}"
            evaluation_targets.append((target_label, hcs_channel, tgen_type))

    total = len(evaluation_targets) * len(feature_subsets) * len(classifiers)
    print(f"\nRunning {total} experiment combinations...")

    for target_label, channel_arg, tgen_type_arg in evaluation_targets:
        channel_label = target_label

        print(f"\n{'='*60}")
        print(f"Channel: {channel_label}")
        print(f"{'='*60}")

        for subset in feature_subsets:
            feature_columns = get_feature_columns(
                list(df.columns), subset
            )

            for clf_name in classifiers:
                print(f"  [{channel_label} / {subset} / {clf_name}]", end=" ")

                X, y, desc = prepare_dataset(
                    df, feature_columns,
                    channel=channel_arg,
                    tgen_type=tgen_type_arg,
                    max_samples=max_samples,
                    balanced=balanced,
                )
                print(desc)

                n_pos = int(y.sum())
                n_neg = int((y == 0).sum())

                result = ExperimentResult(
                    channel=channel_label,
                    feature_subset=subset,
                    classifier=clf_name,
                    n_positive=n_pos,
                    n_negative=n_neg,
                )

                if n_pos == 0 or n_neg == 0:
                    result.warnings.append("No positive or negative samples")
                    results.append(result)
                    continue

                scores, n_folds, warning = run_cv(
                    X, y, clf_name, n_splits
                )

                result.fold_scores = scores
                result.n_folds = n_folds
                if scores:
                    result.mean_auroc = float(np.mean(scores))
                    result.std_auroc = float(np.std(scores))

                if warning:
                    result.warnings.append(warning)

                train_pos, train_neg, test_pos, test_neg = compute_train_test_counts(
                    n_pos, n_neg, n_folds
                )
                result.train_n_positive = train_pos
                result.train_n_negative = train_neg
                result.test_n_positive = test_pos
                result.test_n_negative = test_neg

                results.append(result)
                if scores:
                    print(
                        f"    -> AUROC: {result.mean_auroc:.4f} "
                        f"± {result.std_auroc:.4f} (folds: {n_folds})"
                    )

        # Feature importance for this channel (using full subset)
        if not skip_importance:
            print(f"\n  Computing feature importance for {channel_label}...")
            for clf_name in classifiers:
                key = f"{channel_label}_{clf_name}"
                try:
                    imp_df = extract_feature_importance(
                        df, channel_arg, clf_name,
                        feature_subset="full", top_n=top_n,
                        max_samples=max_samples,
                        tgen_type=tgen_type_arg,
                    )
                    if not imp_df.empty:
                        imp_path = os.path.join(imp_dir, f"{key}.csv")
                        imp_df.to_csv(imp_path, index=False)
                        print(f"    Saved: {imp_path}")

                        # Plot feature importance
                        if not skip_plots:
                            plot_path = os.path.join(
                                plots_dir,
                                f"{key}_feature_importance.png",
                            )
                            plot_feature_importance(
                                imp_df, channel_label, clf_name, "full", plot_path,
                            )
                except Exception as e:
                    print(f"    Failed {key}: {e}")

        if not skip_plots:
            channel_results = [
                r for r in results if r.channel == channel_label
            ]
            if channel_results:
                # Build results DataFrame for plotting
                plot_df = pd.DataFrame([
                    {
                        "channel": r.channel,
                        "feature_subset": r.feature_subset,
                        "classifier": r.classifier,
                        "mean_auroc": r.mean_auroc,
                        "std_auroc": r.std_auroc,
                    }
                    for r in channel_results if r.n_folds > 0
                ])
                if not plot_df.empty:
                    plot_path = os.path.join(
                        plots_dir,
                        f"{channel_label}_performance_by_subset.png",
                    )
                    plot_avg_performance_by_subset(plot_df, channel_label, plot_path)

                    # Plots: performance by classifier per subset
                    for subset in feature_subsets:
                        plot_path = os.path.join(
                            plots_dir,
                            f"{channel_label}_{subset}_performance.png",
                        )
                        plot_performance_by_classifier(
                            plot_df, channel_label, subset, plot_path,
                        )

                # Find top classifier and plot params
                valid_results = [r for r in channel_results if r.n_folds > 0]
                if valid_results:
                    best = max(valid_results, key=lambda r: r.mean_auroc)
                    try:
                        clf = get_classifier(best.classifier)
                        params = clf.get_params()
                        plot_path = os.path.join(
                            plots_dir,
                            f"{channel_label}_{best.classifier}_params.png",
                        )
                        plot_classifier_params(
                            best.classifier, params,
                            best.mean_auroc, best.std_auroc,
                            best.n_positive, best.n_negative,
                            plot_path,
                        )
                    except Exception as e:
                        print(f"  Failed to plot params: {e}")
        else:
            channel_results = [
                r for r in results if r.channel == channel_label
            ]

    # Convert results to DataFrame
    results_df = pd.DataFrame([asdict(r) for r in results])

    # Save CSV
    csv_path = os.path.join(results_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV: {csv_path}")

    # Save JSON
    json_path = os.path.join(results_dir, "results.json")
    json_results = [
        {
            "channel": r.channel,
            "feature_subset": r.feature_subset,
            "classifier": r.classifier,
            "fold_scores": r.fold_scores,
            "mean_auroc": r.mean_auroc,
            "std_auroc": r.std_auroc,
            "n_folds": r.n_folds,
            "n_positive": r.n_positive,
            "n_negative": r.n_negative,
            "train_n_positive": r.train_n_positive,
            "train_n_negative": r.train_n_negative,
            "test_n_positive": r.test_n_positive,
            "test_n_negative": r.test_n_negative,
            "warnings": r.warnings,
        }
        for r in results
    ]
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results JSON: {json_path}")

    # Save sample counts
    counts_df = results_df[
        ["channel", "feature_subset", "classifier",
         "n_positive", "n_negative", "n_folds",
         "train_n_positive", "train_n_negative",
         "test_n_positive", "test_n_negative"]
    ].copy()
    counts_csv = os.path.join(results_dir, "sample_counts.csv")
    counts_df.to_csv(counts_csv, index=False)
    print(f"Saved sample counts: {counts_csv}")

    return results_df
