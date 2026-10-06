"""Feature subset definitions for classifier analysis.

Defines named feature subsets using an explicit INCLUDE approach.
Each subset specifies exactly which columns to include, making the
feature composition clear and easy to audit.
"""

# Columns that are identifiers, not features
IDENTIFIER_COLUMNS = {
    "id", "expiration_id",
    "src_ip", "src_mac", "src_oui",
    "dst_ip", "dst_mac", "dst_oui",
    "src_port", "dst_port",
}

# Columns that are labels/metadata, not features
LABEL_COLUMNS = {
    "src_node", "dst_node",
    "src_container_type", "dst_container_type",
    "is_hcs", "channel_type", "flow_label",
    "network_name", "source_dir",
}

# nDPI L7 visibility features (excluded per transport-layer-only decision)
L7_COLUMNS = {
    "application_name", "application_category_name",
    "application_is_guessed", "application_confidence",
    "requested_server_name", "client_fingerprint",
    "server_fingerprint", "user_agent", "content_type",
}

# Timestamp columns (not useful as classification features)
TIMESTAMP_COLUMNS = {
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
}

ALL_EXCLUDED_COLUMNS = (
    IDENTIFIER_COLUMNS
    | LABEL_COLUMNS
    | L7_COLUMNS
    | TIMESTAMP_COLUMNS
)

DIRECTIONS = ["bidirectional", "src2dst", "dst2src"]
FLAGS = ["syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin"]
PS_STATS = ["min_ps", "mean_ps", "stddev_ps", "max_ps"]
PIAT_STATS = ["min_piat_ms", "mean_piat_ms", "stddev_piat_ms", "max_piat_ms"]
COUNTS = ["packets", "bytes"]
DURATIONS = ["duration_ms"]


def _tcp_flag_columns() -> list[str]:
    """All TCP flag count columns."""
    return [f"{d}_{f}_packets" for d in DIRECTIONS for f in FLAGS]


def _packet_size_columns() -> list[str]:
    """All packet size stat columns."""
    return [f"{d}_{s}" for d in DIRECTIONS for s in PS_STATS]


def _interarrival_columns() -> list[str]:
    """All inter-arrival time stat columns."""
    return [f"{d}_{s}" for d in DIRECTIONS for s in PIAT_STATS]


def _count_columns() -> list[str]:
    """All packet/byte count columns."""
    return [f"{d}_{c}" for d in DIRECTIONS for c in COUNTS]


def _duration_columns() -> list[str]:
    """All duration columns."""
    return [f"{d}_{c}" for d in DIRECTIONS for c in DURATIONS]


def _directional_columns() -> list[str]:
    """All src2dst and dst2src columns (directional only, excluding bidirectional)."""
    cols = []
    for d in ["src2dst", "dst2src"]:
        cols.extend([f"{d}_{c}" for c in COUNTS + DURATIONS])
        cols.extend([f"{d}_{s}" for s in PS_STATS + PIAT_STATS])
        cols.extend([f"{d}_{f}_packets" for f in FLAGS])
    return cols


# Protocol info columns
PROTOCOL_COLUMNS = ["protocol", "ip_version", "vlan_id", "tunnel_id"]

# All transport-layer feature columns (sorted for deterministic ordering)
ALL_FEATURE_COLUMNS = sorted(
    _tcp_flag_columns()
    + _packet_size_columns()
    + _interarrival_columns()
    + _count_columns()
    + _duration_columns()
    + PROTOCOL_COLUMNS
)


def _remove_items(base: list[str], to_remove: list[str]) -> list[str]:
    """Return base list with items in to_remove excluded."""
    remove_set = set(to_remove)
    return sorted([c for c in base if c not in remove_set])


def _get_full_subset() -> list[str]:
    """Get all transport-layer feature columns."""
    return sorted(ALL_FEATURE_COLUMNS)


def _get_no_tcp_flags_subset() -> list[str]:
    """Full set minus TCP flag count columns."""
    return _remove_items(ALL_FEATURE_COLUMNS, _tcp_flag_columns())


def _get_no_statistical_subset() -> list[str]:
    """Full set minus packet size stats and inter-arrival time stats."""
    return _remove_items(ALL_FEATURE_COLUMNS, _packet_size_columns() + _interarrival_columns())


def _get_no_directional_subset() -> list[str]:
    """Bidirectional features only (no src2dst/dst2src directional)."""
    return sorted(
        [f"bidirectional_{c}" for c in COUNTS + DURATIONS]
        + [f"bidirectional_{s}" for s in PS_STATS + PIAT_STATS]
        + [f"bidirectional_{f}_packets" for f in FLAGS]
        + PROTOCOL_COLUMNS
    )


def _get_no_byte_counts_subset() -> list[str]:
    """Full set minus byte count columns."""
    byte_cols = [f"{d}_bytes" for d in DIRECTIONS]
    return _remove_items(ALL_FEATURE_COLUMNS, byte_cols)


def _get_no_iat_subset() -> list[str]:
    """Full set minus inter-arrival time columns."""
    return _remove_items(ALL_FEATURE_COLUMNS, _interarrival_columns())


def _get_minimal_subset() -> list[str]:
    """Duration + packet/byte counts only (9 core features)."""
    return sorted(
        _duration_columns()
        + _count_columns()
        + PROTOCOL_COLUMNS
    )


def _get_duration_only_subset() -> list[str]:
    """Only duration columns + protocol info."""
    return sorted(_duration_columns() + PROTOCOL_COLUMNS)


def _get_packet_counts_only_subset() -> list[str]:
    """Only packet count columns + protocol info."""
    return sorted(
        [f"{d}_packets" for d in DIRECTIONS]
        + PROTOCOL_COLUMNS
    )


def _get_statistical_only_subset() -> list[str]:
    """Only statistical features (min/max/stddev/mean for ps and piat) + protocol info."""
    return sorted(
        _packet_size_columns()
        + _interarrival_columns()
        + PROTOCOL_COLUMNS
    )


def _get_timing_only_subset() -> list[str]:
    """Only inter-arrival time (piat) features + protocol info."""
    return sorted(_interarrival_columns() + PROTOCOL_COLUMNS)


def _get_sizes_only_subset() -> list[str]:
    """Only packet size (ps) features + protocol info."""
    return sorted(_packet_size_columns() + PROTOCOL_COLUMNS)


# All named subsets available
NAMED_SUBSETS = {
    "full": _get_full_subset,
    "no_tcp_flags": _get_no_tcp_flags_subset,
    "no_statistical": _get_no_statistical_subset,
    "no_directional": _get_no_directional_subset,
    "no_byte_counts": _get_no_byte_counts_subset,
    "no_iat": _get_no_iat_subset,
    "minimal": _get_minimal_subset,
    "duration_only": _get_duration_only_subset,
    "packet_counts_only": _get_packet_counts_only_subset,
    "all-stats": _get_statistical_only_subset,
    "timing": _get_timing_only_subset,
    "sizes": _get_sizes_only_subset,
}


def get_all_feature_columns(df_columns: list[str] | None = None) -> list[str]:
    """Get all transport-layer feature columns.

    Args:
        df_columns: If provided, filter to columns that exist in the DataFrame.

    Returns:
        Sorted list of all transport-layer feature column names.
    """
    if df_columns is None:
        return sorted(ALL_FEATURE_COLUMNS)
    return sorted(c for c in ALL_FEATURE_COLUMNS if c in df_columns)


def get_feature_columns(df_columns: list[str], subset_name: str) -> list[str]:
    """Get the feature columns for a named subset.

    Args:
        df_columns: List of all column names in the DataFrame.
        subset_name: Name of the subset (e.g., "full", "minimal", "duration_only").

    Returns:
        Sorted list of feature column names for the subset,
        filtered to only include columns present in df_columns.

    Raises:
        ValueError: If subset_name is not recognized.
    """
    if subset_name not in NAMED_SUBSETS:
        raise ValueError(
            f"Unknown feature subset '{subset_name}'. "
            f"Valid subsets: {', '.join(sorted(NAMED_SUBSETS.keys()))}"
        )

    all_cols = NAMED_SUBSETS[subset_name]()
    return sorted(c for c in all_cols if c in df_columns)


# List of standard subset names (excluding custom ones)
FEATURE_SUBSET_NAMES = ["full", "no_tcp_flags", "no_statistical", "no_directional",
                        "no_byte_counts", "no_iat", "minimal"]
