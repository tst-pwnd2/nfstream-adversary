"""Feature importance extraction and ranking.

Extracts feature importance from trained classifiers (RF/GBM via
feature_importances_, LogReg via coefficients, GNB via proxy method)
and aggregates across channels.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone

from .classifiers import get_classifier
from .data_prep import prepare_dataset
from .features import get_feature_columns


def extract_feature_importance(
    df: pd.DataFrame,
    channel: str | None,
    classifier_name: str,
    feature_subset: str = "full",
    top_n: int = 20,
    max_samples: int | None = None,
    tgen_type: str | None = None,
) -> pd.DataFrame:
    """Train a classifier on the full dataset and extract feature importance.

    Args:
        df: Full flows DataFrame.
        channel: HCS channel name, or None for pooled.
        classifier_name: One of logreg, random_forest, gradient_boosting, gaussian_nb.
        feature_subset: Feature subset to use (default: full).
        top_n: Number of top features to return.
        max_samples: If set, cap total samples via stratified subsampling.
        tgen_type: TGEN type name for per-TGEN-type analysis.

    Returns:
        DataFrame with columns: feature, importance, rank.
        Sorted by importance descending.
    """
    feature_columns = get_feature_columns(list(df.columns), feature_subset)
    X, y, _ = prepare_dataset(
        df, feature_columns, channel=channel,
        tgen_type=tgen_type, max_samples=max_samples
    )

    if len(X) == 0 or y.nunique() < 2:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    classifier = get_classifier(classifier_name)
    classifier.fit(X, y)

    importances = _get_importances(classifier, classifier_name, X, y)

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    })
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df = importance_df.head(top_n)

    return importance_df


def _get_importances(
    fitted_pipeline,
    classifier_name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> np.ndarray:
    """Extract feature importance from a fitted pipeline.

    - RF/GBM: use feature_importances_ from the final estimator.
    - LogReg: use absolute value of coefficients (averaged across classes for multiclass).
    - GNB: use proxy = mean of (class_means difference) per feature.

    Args:
        fitted_pipeline: Fitted sklearn Pipeline.
        classifier_name: Classifier identifier.
        X: Feature DataFrame (used for shape).
        y: Labels (used for GNB proxy).

    Returns:
        Array of importance values, same length as X.columns.
    """
    estimator = fitted_pipeline.named_steps["clf"]
    X_cols = X.columns

    if classifier_name in ("random_forest", "gradient_boosting"):
        return estimator.feature_importances_

    elif classifier_name == "logreg":
        coefs = estimator.coef_
        # For binary classification, coef_ is shape (1, n_features)
        # Use mean absolute coefficient
        if coefs.ndim > 1:
            coefs = coefs.mean(axis=0)
        return np.abs(coefs)

    elif classifier_name == "gaussian_nb":
        # GNB doesn't have feature_importances_, so use a proxy:
        # the absolute difference in mean feature values between the two classes
        return _gnb_importance_proxy(estimator, X, y, X_cols)

    return np.zeros(len(X_cols))


def _gnb_importance_proxy(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
) -> np.ndarray:
    """Compute a proxy for GNB feature importance.

    Uses the absolute difference of per-class feature means, normalized.
    This approximates which features best separate the classes.

    Args:
        estimator: Fitted GaussianNB.
        X: Feature matrix.
        y: Labels.
        feature_names: Feature column names.

    Returns:
        Array of importance values.
    """
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    mean_diff = np.abs(class_0.mean() - class_1.mean()).values

    # Normalize to [0, 1]
    max_diff = mean_diff.max()
    if max_diff > 0:
        mean_diff = mean_diff / max_diff

    return mean_diff


def compute_all_importance(
    df: pd.DataFrame,
    channels: list[str | None],
    classifiers: list[str],
    feature_subset: str = "full",
    top_n: int = 20,
    output_dir: str | None = None,
    max_samples: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute feature importance for all channel × classifier combinations.

    Args:
        df: Full flows DataFrame.
        channels: List of channel names or None for pooled.
        classifiers: List of classifier names.
        feature_subset: Feature subset name (default: full).
        top_n: Top N features to save per combination.
        output_dir: If provided, save individual CSVs to this directory.
        max_samples: If set, cap total samples via stratified subsampling.

    Returns:
        Dict mapping "{channel}_{classifier}" -> importance DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}

    for channel in channels:
        channel_label = channel if channel else "pooled"

        for clf_name in classifiers:
            key = f"{channel_label}_{clf_name}"
            print(f"  Computing importance: {key}")

            try:
                imp_df = extract_feature_importance(
                    df, channel, clf_name,
                    feature_subset=feature_subset, top_n=top_n,
                    max_samples=max_samples,
                )
                results[key] = imp_df

                if output_dir and not imp_df.empty:
                    import os
                    os.makedirs(output_dir, exist_ok=True)
                    fname = f"{key.replace('/', '_')}.csv"
                    imp_df.to_csv(os.path.join(output_dir, fname), index=False)
            except Exception as e:
                print(f"    Failed: {e}")
                results[key] = pd.DataFrame(columns=["feature", "importance", "rank"])

    return results


def aggregate_importance(
    importance_results: dict[str, pd.DataFrame],
    top_n: int = 20,
) -> pd.DataFrame:
    """Aggregate feature importance across all channel × classifier results.

    Computes the mean rank of each feature across all results, then returns
    the top features. Lower mean rank = more important.

    Args:
        importance_results: Dict from compute_all_importance.
        top_n: Number of top features to return.

    Returns:
        DataFrame with columns: feature, mean_rank, mean_importance, n_present.
    """
    rank_data: list[dict] = []
    imp_data: list[dict] = []

    for key, df in importance_results.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            rank_data.append({
                "feature": row["feature"],
                "rank": row["rank"],
                "source": key,
            })
            imp_data.append({
                "feature": row["feature"],
                "importance": row["importance"],
                "source": key,
            })

    if not rank_data:
        return pd.DataFrame(columns=["feature", "mean_rank", "mean_importance", "n_present"])

    rank_df = pd.DataFrame(rank_data)
    imp_df = pd.DataFrame(imp_data)

    summary = rank_df.groupby("feature").agg(
        mean_rank=("rank", "mean"),
        n_present=("rank", "count"),
    ).reset_index()

    imp_summary = imp_df.groupby("feature").agg(
        mean_importance=("importance", "mean"),
    ).reset_index()

    summary = summary.merge(imp_summary, on="feature")
    summary = summary.sort_values("mean_rank").head(top_n)
    summary = summary.reset_index(drop=True)

    return summary
