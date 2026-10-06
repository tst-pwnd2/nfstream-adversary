"""Data loading and preprocessing for classifier pipeline.

Handles loading extracted flows from parquet, preparing feature matrices
and labels for both pooled and per-channel binary classification, and
handling missing/inf values.
"""

import numpy as np
import pandas as pd

from .features import get_feature_columns


def load_flows(parquet_path: str) -> pd.DataFrame:
    """Load the extracted flows from a parquet file.

    Args:
        parquet_path: Path to the flows.parquet file.

    Returns:
        DataFrame with all flow data including labels.
    """
    return pd.read_parquet(parquet_path)


def prepare_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    channel: str | None = None,
    max_samples: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Prepare features X and binary label y.

    For pooled analysis (channel=None):
        y = True for any HCS flow, False for TGEN flow.

    For per-channel analysis (channel="racetunnel"):
        y = True for HCS flows of that channel, False for all TGEN flows.

    Args:
        df: Full flows DataFrame with labels.
        feature_columns: List of column names to use as features.
        channel: HCS channel name (e.g., "racetunnel") or None for pooled.
        max_samples: If set, cap total samples via stratified subsampling.
            The minority class is preserved; the majority is downsampled.
        random_state: Random seed for subsampling.

    Returns:
        Tuple of (X, y, description) where:
            X: DataFrame of feature values (NaN/inf handled)
            y: Series of binary labels (1 for positive class, 0 for negative)
            description: String describing the dataset (for logging)
    """
    if channel is None:
        # Pooled: any HCS vs all TGEN
        positive_mask = df["is_hcs"] == True  # noqa: E712
        negative_mask = df["is_hcs"] == False  # noqa: E712
        desc = f"pool: HCS={positive_mask.sum()}, TGEN={negative_mask.sum()}"
    else:
        # Per-channel: HCS flows of this channel vs all TGEN
        positive_mask = (df["is_hcs"] == True) & (df["channel_type"] == channel)  # noqa: E712
        negative_mask = df["is_hcs"] == False  # noqa: E712
        desc = (
            f"channel={channel}: "
            f"HCS_{channel}={positive_mask.sum()}, TGEN={negative_mask.sum()}"
        )

    subset = df[positive_mask | negative_mask].copy()

    # Subsample if dataset is too large, preserving class proportions
    if max_samples is not None and len(subset) > max_samples:
        n_pos = int(positive_mask.sum())
        n_neg = int(negative_mask.sum())
        # Scale both classes proportionally
        scale = max_samples / len(subset)
        n_pos_sample = max(10, int(n_pos * scale))
        n_neg_sample = max(10, int(n_neg * scale))
        # Don't upsample the minority class
        n_pos_sample = min(n_pos_sample, n_pos)
        n_neg_sample = min(n_neg_sample, n_neg)

        pos_idx = subset.index[subset["is_hcs"] == True][:n_pos_sample]
        neg_idx = subset.index[subset["is_hcs"] == False][:n_neg_sample]
        subset = subset.loc[list(pos_idx) + list(neg_idx)]
        desc += f" -> subsampled: HCS={n_pos_sample}, TGEN={n_neg_sample}"

    X = subset[feature_columns].copy()
    y = (subset["is_hcs"] == True).astype(int)  # noqa: E712

    X = handle_missing_values(X)

    return X, y, desc


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Handle NaN and inf values in feature DataFrame.

    - NaN values (e.g., TCP flags for UDP flows) filled with 0.
    - inf values (e.g., stddev when all values equal) replaced with 0.

    Args:
        X: Feature DataFrame.

    Returns:
        DataFrame with NaN/inf values handled.
    """
    X = X.copy()

    # Replace inf with NaN first, then fill all with 0
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X


def get_available_channels(df: pd.DataFrame) -> list[str]:
    """Get list of HCS channels available in the data.

    Args:
        df: Flows DataFrame with labels.

    Returns:
        Sorted list of channel names present in HCS flows.
    """
    hcs_channels = df.loc[df["is_hcs"] == True, "channel_type"].dropna().unique()  # noqa: E712
    return sorted(hcs_channels.tolist())
