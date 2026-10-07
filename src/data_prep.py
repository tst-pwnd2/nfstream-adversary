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
    channel: str | list[str] | None = None,
    tgen_type: str | list[str] | None = None,
    max_samples: int | None = None,
    balanced: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Prepare features X and binary label y.

    For pooled analysis (channel=None):
        y = True for any HCS flow, False for TGEN flow.

    For per-channel analysis (channel="racetunnel"):
        y = True for HCS flows of that channel, False for all TGEN flows.
        channel can also be a list to pool multiple channels together.

    For per-TGEN-type analysis (tgen_type="FTP"):
        y = True for TGEN flows of that type, False for all other TGEN flows.
        tgen_type can also be a list to pool multiple TGEN types together.

    For paired analysis (channel="racetunnel", tgen_type="IRC"):
        y = True for HCS flows of that channel, False for TGEN flows of that type.
        Both channel and tgen_type can be lists to pool multiple items.

    Args:
        df: Full flows DataFrame with labels.
        feature_columns: List of column names to use as features.
        channel: HCS channel name(s) or None for pooled.
        tgen_type: TGEN type name(s) for TGEN-specific analysis.
        max_samples: If set, cap total samples via stratified subsampling.
            The minority class is preserved; the majority is downsampled.
        balanced: If True, downsample the majority class to match the minority
            class size, creating a balanced dataset.
        random_state: Random seed for subsampling.

    Returns:
        Tuple of (X, y, description) where:
            X: DataFrame of feature values (NaN/inf handled)
            y: Series of binary labels (1 for positive class, 0 for negative)
            description: String describing the dataset (for logging)
    """
    # Normalize channel and tgen_type to lists
    channel_list = None
    if channel is not None:
        channel_list = [channel] if isinstance(channel, str) else channel

    tgen_type_list = None
    if tgen_type is not None:
        tgen_type_list = [tgen_type] if isinstance(tgen_type, str) else tgen_type

    if channel_list and tgen_type_list:
        # Paired: HCS flows of channel(s) vs TGEN flows of tgen_type(s)
        # Pool all listed HCS channels as positive
        pos_channel_mask = df["is_hcs"] == True  # noqa: E712
        channel_mask = df["channel_type"].isin(channel_list)
        positive_mask = pos_channel_mask & channel_mask

        # Pool all listed TGEN types as negative
        neg_tgen_mask = df["is_hcs"] == False  # noqa: E712
        tgen_mask = df["tgen_type"].isin(tgen_type_list)
        negative_mask = neg_tgen_mask & tgen_mask

        ch_str = channel_list[0] if len(channel_list) == 1 else str(channel_list)
        tt_str = tgen_type_list[0] if len(tgen_type_list) == 1 else str(tgen_type_list)
        desc = f"paired: HCS_[{ch_str}]={positive_mask.sum()}, TGEN_[{tt_str}]={negative_mask.sum()}"

    elif channel_list is None and tgen_type_list is None:
        # Pooled: any HCS vs all TGEN
        positive_mask = df["is_hcs"] == True  # noqa: E712
        negative_mask = df["is_hcs"] == False  # noqa: E712
        desc = f"pool: HCS={positive_mask.sum()}, TGEN={negative_mask.sum()}"

    elif tgen_type_list and channel_list is None:
        # Per-TGEN-type: TGEN flows of this type vs all other TGEN flows
        neg_mask = df["is_hcs"] == False  # noqa: E712
        tgen_mask = df["tgen_type"].isin(tgen_type_list)
        positive_mask = neg_mask & tgen_mask
        negative_mask = neg_mask & ~tgen_mask

        tt_str = tgen_type_list[0] if len(tgen_type_list) == 1 else str(tgen_type_list)
        desc = (
            f"tgen_type={tt_str}: "
            f"TGEN_{tt_str}={positive_mask.sum()}, "
            f"other_TGEN={negative_mask.sum()}"
        )

    elif channel_list and tgen_type_list is None:
        # Per-channel: HCS flows of this channel vs all TGEN
        pos_mask = df["is_hcs"] == True  # noqa: E712
        channel_mask = df["channel_type"].isin(channel_list)
        positive_mask = pos_mask & channel_mask
        negative_mask = df["is_hcs"] == False  # noqa: E712

        ch_str = channel_list[0] if len(channel_list) == 1 else str(channel_list)
        desc = (
            f"channel={ch_str}: "
            f"HCS_{ch_str}={positive_mask.sum()}, TGEN={negative_mask.sum()}"
        )

    else:
        positive_mask = df["is_hcs"] == True  # noqa: E712
        negative_mask = df["is_hcs"] == False  # noqa: E712
        desc = f"pool: HCS={positive_mask.sum()}, TGEN={negative_mask.sum()}"

    subset = df[positive_mask | negative_mask].copy()

    # Track which rows are positive (for subsampling/balancing)
    is_positive = positive_mask.reindex(subset.index, fill_value=False)

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

        pos_idx = subset.index[is_positive][:n_pos_sample]
        neg_idx = subset.index[~is_positive][:n_neg_sample]
        subset = subset.loc[list(pos_idx) + list(neg_idx)]
        is_positive = is_positive.reindex(subset.index, fill_value=False)
        pos_label = "HCS" if channel is None and tgen_type is None else (f"TGEN_{tgen_type}" if tgen_type else f"HCS_{channel}")
        neg_label = "TGEN" if channel is None and tgen_type is None else (f"other_TGEN" if tgen_type else "TGEN")
        desc += f" -> subsampled: {pos_label}={n_pos_sample}, {neg_label}={n_neg_sample}"

    # Balance classes by downsampling the majority class
    if balanced:
        n_pos = int(is_positive.sum())
        n_neg = int((~is_positive).sum())
        n_target = min(n_pos, n_neg)
        if n_pos > n_neg:
            pos_idx = subset.index[is_positive][:n_target]
            neg_idx = subset.index[~is_positive]
        else:
            pos_idx = subset.index[is_positive]
            neg_idx = subset.index[~is_positive][:n_target]
        subset = subset.loc[list(pos_idx) + list(neg_idx)]
        is_positive = is_positive.reindex(subset.index, fill_value=False)
        n_pos_final = int(is_positive.sum())
        n_neg_final = int((~is_positive).sum())
        desc += f" -> balanced: pos={n_pos_final}, neg={n_neg_final}"

    X = subset[feature_columns].copy()
    y = is_positive.astype(int)

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


def get_available_tgen_types(df: pd.DataFrame) -> list[str]:
    """Get list of TGEN types available in the data.

    Args:
        df: Flows DataFrame with labels.

    Returns:
        Sorted list of TGEN type names (uppercase) present in TGEN flows.
    """
    tgen_types = df.loc[df["is_hcs"] == False, "tgen_type"].dropna().unique()  # noqa: E712
    return sorted(tgen_types.tolist())
