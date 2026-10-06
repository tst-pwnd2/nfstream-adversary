"""Cross-validation pipeline for evaluating classifiers on flow data.

Runs stratified k-fold CV across channels, feature subsets, and classifiers.
Collects AUROC scores and metadata for analysis.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from .classifiers import CLASSIFIER_NAMES, get_classifier
from .data_prep import get_available_channels, handle_missing_values, prepare_dataset
from .features import FEATURE_SUBSET_NAMES, get_feature_columns, get_all_feature_columns


@dataclass
class CVResult:
    """Result of a single CV evaluation run."""

    channel: str
    feature_subset: str
    classifier: str
    fold_scores: list[float] = field(default_factory=list)
    mean_auroc: float = 0.0
    std_auroc: float = 0.0
    n_positive: int = 0
    n_negative: int = 0
    n_folds: int = 0
    warning: str | None = None


def evaluate_single(
    X: pd.DataFrame,
    y: pd.Series,
    classifier_name: str,
    channel: str,
    feature_subset: str,
    n_splits: int = 5,
) -> CVResult:
    """Run stratified k-fold CV for a single configuration.

    Args:
        X: Feature DataFrame.
        y: Binary label Series.
        classifier_name: Key into CLASSIFIERS.
        channel: Channel name for the result record.
        feature_subset: Feature subset name for the result record.
        n_splits: Number of CV folds.

    Returns:
        CVResult with fold scores and summary statistics.
    """
    n_positive = int(y.sum())
    n_negative = int((y == 0).sum())

    result = CVResult(
        channel=channel,
        feature_subset=feature_subset,
        classifier=classifier_name,
        n_positive=n_positive,
        n_negative=n_negative,
    )

    # Determine number of folds based on smallest class
    min_class_size = min(n_positive, n_negative)
    actual_folds = min(n_splits, min_class_size)

    if actual_folds < 2:
        result.warning = (
            f"Only {n_positive} positive / {n_negative} negative samples; "
            f"cannot perform CV"
        )
        result.n_folds = 0
        return result

    if actual_folds < n_splits:
        result.warning = (
            f"Reduced to {actual_folds}-fold CV (min class size: {min_class_size})"
        )

    result.n_folds = actual_folds

    classifier = get_classifier(classifier_name)
    skf = StratifiedKFold(
        n_splits=actual_folds,
        shuffle=True,
        random_state=42,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                classifier, X, y, cv=skf, scoring="roc_auc", n_jobs=-1
            )
        result.fold_scores = scores.tolist()
        result.mean_auroc = float(np.mean(scores))
        result.std_auroc = float(np.std(scores))
    except Exception as e:
        result.warning = f"CV failed: {str(e)}"
        result.n_folds = 0

    return result


def evaluate_channel(
    df: pd.DataFrame,
    channel: str | None,
    feature_subset: str,
    classifier_name: str,
    max_samples: int | None = None,
) -> CVResult:
    """Evaluate a single channel + subset + classifier combination.

    Args:
        df: Full flows DataFrame.
        channel: HCS channel name, or None for pooled.
        feature_subset: Feature subset name.
        classifier_name: Classifier key.
        max_samples: Max total samples (subsample if exceeded).

    Returns:
        CVResult.
    """
    channel_label = channel if channel else "pooled"

    feature_columns = get_feature_columns(list(df.columns), feature_subset)

    X, y, desc = prepare_dataset(
        df, feature_columns, channel=channel, max_samples=max_samples
    )

    print(f"  [{channel_label} / {feature_subset} / {classifier_name}] {desc}")
    print(f"    Features: {len(feature_columns)}, "
          f"Positive: {y.sum()}, Negative: {(y==0).sum()}")

    if y.sum() == 0 or (y == 0).sum() == 0:
        return CVResult(
            channel=channel_label,
            feature_subset=feature_subset,
            classifier=classifier_name,
            n_positive=int(y.sum()),
            n_negative=int((y == 0).sum()),
            warning="No positive or negative samples",
        )

    return evaluate_single(X, y, classifier_name, channel_label, feature_subset)


def run_full_evaluation(
    df: pd.DataFrame,
    channels: list[str | None],
    feature_subsets: list[str],
    classifiers: list[str],
    n_splits: int = 5,
    max_samples: int | None = None,
) -> list[CVResult]:
    """Run the complete evaluation matrix.

    Args:
        df: Full flows DataFrame.
        channels: List of channel names, or [None] for pooled.
        feature_subsets: List of subset names.
        classifiers: List of classifier names.
        n_splits: CV folds.
        max_samples: Max samples per evaluation (subsample if exceeded).

    Returns:
        List of CVResult objects.
    """
    results: list[CVResult] = []

    total = len(channels) * len(feature_subsets) * len(classifiers)
    print(f"\nRunning {total} evaluation combinations...")

    for channel in channels:
        channel_label = channel if channel else "pooled"
        print(f"\n{'='*60}")
        print(f"Channel: {channel_label}")
        print(f"{'='*60}")

        for subset in feature_subsets:
            for clf_name in classifiers:
                result = evaluate_channel(
                    df, channel, subset, clf_name, max_samples=max_samples
                )
                results.append(result)
                if result.mean_auroc > 0:
                    print(f"    -> AUROC: {result.mean_auroc:.4f} "
                          f"+/- {result.std_auroc:.4f} "
                          f"(folds: {result.n_folds})")
                if result.warning:
                    print(f"    -> WARNING: {result.warning}")

    return results


def results_to_dataframe(results: list[CVResult]) -> pd.DataFrame:
    """Convert CVResult list to a DataFrame for easy analysis.

    Args:
        results: List of CVResult objects.

    Returns:
        DataFrame with columns: channel, feature_subset, classifier, fold_0..N,
            mean_auroc, std_auroc, n_positive, n_negative, n_folds, warning.
    """
    rows = []
    max_folds = max(len(r.fold_scores) for r in results) if results else 0

    for r in results:
        row = {
            "channel": r.channel,
            "feature_subset": r.feature_subset,
            "classifier": r.classifier,
        }
        for i in range(max_folds):
            col = f"fold_{i}"
            if i < len(r.fold_scores):
                row[col] = r.fold_scores[i]
            else:
                row[col] = np.nan
        row["mean_auroc"] = r.mean_auroc
        row["std_auroc"] = r.std_auroc
        row["n_positive"] = r.n_positive
        row["n_negative"] = r.n_negative
        row["n_folds"] = r.n_folds
        row["warning"] = r.warning or ""
        rows.append(row)

    return pd.DataFrame(rows)
