"""Orchestrates nfstream extraction, flow labeling, and output.

This module ties together pcap discovery, network map loading, nfstream
flow extraction, and flow labeling to produce labeled DataFrames.
"""

import os
import sys

import pandas as pd

try:
    from nfstream import NFStreamer
except ImportError:
    NFStreamer = None

from .labeler import FlowLabel, is_dns_node_traffic, is_dns_traffic, label_flow
from .nm_loader import NetworkNodeInfo, load_network_map
from .pcap_discovery import PCAPGroup, discover_pcap_groups

# Columns that encode bidirectional direction-agnostic statistics.
# These never need swapping because they aggregate both directions.
_BIDIRECTIONAL_COLUMNS = {
    "bidirectional_first_seen_ms",
    "bidirectional_last_seen_ms",
    "bidirectional_duration_ms",
    "bidirectional_packets",
    "bidirectional_bytes",
    "bidirectional_min_ps",
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_max_ps",
    "bidirectional_min_piat_ms",
    "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms",
    "bidirectional_max_piat_ms",
    "bidirectional_syn_packets",
    "bidirectional_cwr_packets",
    "bidirectional_ece_packets",
    "bidirectional_urg_packets",
    "bidirectional_ack_packets",
    "bidirectional_psh_packets",
    "bidirectional_rst_packets",
    "bidirectional_fin_packets",
    "application_name",
    "application_category_name",
    "application_is_guessed",
    "application_confidence",
    "requested_server_name",
    "client_fingerprint",
    "server_fingerprint",
    "user_agent",
    "content_type",
}

# Columns that represent directional flow statistics. src2dst_* tracks
# traffic from src_ip to dst_ip; dst2src_* tracks traffic from dst_ip to
# src_ip. When a flow's directionality is reversed (e.g., by nfstream's
# active_timeout creating a new flow from a server-side packet), these
# pairs must be swapped to maintain consistent directionality.
_SRC2DST_PREFIX = "src2dst_"
_DST2SRC_PREFIX = "dst2src_"


def _get_directional_stat_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Find pairs of (src2dst_, dst2src_) columns in the DataFrame.

    Args:
        df: DataFrame with nfstream flow columns.

    Returns:
        List of (src2dst_col, dst2src_col) pairs that exist in the DataFrame.
    """
    pairs = []
    src2dst_cols = [
        c for c in df.columns
        if c.startswith(_SRC2DST_PREFIX) and c not in _BIDIRECTIONAL_COLUMNS
    ]
    for col in src2dst_cols:
        suffix = col[len(_SRC2DST_PREFIX):]
        dst2src_col = f"{_DST2SRC_PREFIX}{suffix}"
        if dst2src_col in df.columns:
            pairs.append((col, dst2src_col))
    return pairs


def extract_flows_from_group(
    group: PCAPGroup,
    ip_lookup: dict[str, NetworkNodeInfo],
    active_timeout: int | None = None,
) -> pd.DataFrame:
    """Extract flows from a single pcap group and apply labels.

    Args:
        group: PCAPGroup containing pcap paths and network name.
        ip_lookup: IP-to-node lookup from load_network_map().
        active_timeout: Optional nfstream active_timeout override (seconds).

    Returns:
        DataFrame with nfstream features plus labeling columns.
    """
    if NFStreamer is None:
        raise ImportError(
            "nfstream is not installed. Install requirements: pip install -r requirements.txt"
        )

    streamer_kwargs = {}
    if active_timeout is not None:
        streamer_kwargs["active_timeout"] = active_timeout

    try:
        streamer = NFStreamer(
            source=group.pcap_paths,
            statistical_analysis=True,
            n_dissections=20,
            **streamer_kwargs,
        )
        df = streamer.to_pandas()
    except Exception as e:
        print(
            f"  WARNING: Failed to process {group.source_dir}: {e}",
            file=sys.stderr,
        )
        return pd.DataFrame()

    if df.empty:
        print(
            f"  WARNING: No flows extracted from {group.source_dir}",
            file=sys.stderr,
        )
        return df

    df = _apply_labels(df, ip_lookup)

    df = _filter_flows(df)

    df = _correct_flow_directionality(df, ip_lookup)

    # Add metadata columns
    df["network_name"] = group.network_name
    df["source_dir"] = group.source_dir

    return df


def _apply_labels(
    df: pd.DataFrame,
    ip_lookup: dict[str, NetworkNodeInfo],
) -> pd.DataFrame:
    """Apply flow labeling columns to the DataFrame.

    Adds: src_node, dst_node, src_container_type, dst_container_type,
    is_hcs, channel_type, flow_label
    """
    labels: list[FlowLabel] = []
    for _, row in df.iterrows():
        src_ip = str(row.get("src_ip", ""))
        dst_ip = str(row.get("dst_ip", ""))
        labels.append(label_flow(src_ip, dst_ip, ip_lookup))

    df = df.copy()
    df["src_node"] = [l.src_node for l in labels]
    df["dst_node"] = [l.dst_node for l in labels]
    df["src_container_type"] = [l.src_container_type for l in labels]
    df["dst_container_type"] = [l.dst_container_type for l in labels]
    df["is_hcs"] = [l.is_hcs for l in labels]
    df["channel_type"] = [l.channel_type for l in labels]
    df["tgen_type"] = [l.tgen_type for l in labels]

    # Combined label string: "hcs_racetunnel" or "tgen_mastodon"
    flow_labels = []
    for l in labels:
        if l.is_hcs is True:
            prefix = "hcs"
        elif l.is_hcs is False:
            prefix = "tgen"
        else:
            prefix = "infra"
        channel = l.channel_type if l.channel_type else "unknown"
        flow_labels.append(f"{prefix}_{channel}")
    df["flow_label"] = flow_labels

    return df


def _filter_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out infrastructure-only and DNS traffic.

    1. Drop rows where is_hcs is None (no HCS or TGEN endpoint).
    2. Drop DNS traffic (nDPI identification or port-based heuristics).
    3. Drop flows where both endpoints are DNS infrastructure nodes.
    """
    # Drop infrastructure-only traffic (neither endpoint is HCS or TGEN)
    df = df[df["is_hcs"].notna()].copy()

    # Drop DNS traffic detected by nDPI application_name
    if "application_name" in df.columns:
        dns_mask = df.apply(is_dns_traffic, axis=1)
        df = df[~dns_mask].copy()

    # Drop flows where both endpoints are DNS infrastructure nodes
    dns_node_mask = df.apply(
        lambda row: is_dns_node_traffic(
            row.get("src_node", ""), row.get("dst_node", "")
        ),
        axis=1,
    )
    df = df[~dns_node_mask].copy()

    return df


def _correct_flow_directionality(
    df: pd.DataFrame,
    ip_lookup: dict[str, NetworkNodeInfo],
) -> pd.DataFrame:
    """Correct reversed flow directionality caused by active_timeout.

    When nfstream's active_timeout prematurely expires a flow, the next
    packet may come from the server side. nfstream's direction-agnostic
    flow key means the new flow instance's src_ip/dst_ip are set based
    on that packet's direction — which can be the reverse of the original
    flow. This causes:

    - HCS/TGEN endpoint appearing as dst_ip instead of src_ip
    - Directional statistics (src2dst_*/dst2src_*) being inverted
    - src_node/dst_node being swapped

    This function detects such reversed flows and swaps all directional
    fields so that HCS nodes are always the src (client) and TGEN nodes
    are the source initiator, matching the convention of non-split flows.

    Args:
        df: DataFrame with labeled flows (must have src_ip, dst_ip,
            is_hcs, src_node, dst_node columns).
        ip_lookup: IP-to-node lookup from load_network_map().

    Returns:
        DataFrame with corrected directionality for reversed flows.
    """
    if df.empty:
        return df

    df = df.copy()

    # Precompute the set of HCS IPs for fast lookup
    hcs_ips = {ip for ip, info in ip_lookup.items() if info.is_hcs}
    tgen_ips = {ip for ip, info in ip_lookup.items() if info.container_type == "tgen"}

    # Identify flows where the HCS node is the dst (should be src).
    # HCS nodes are always clients (they initiate traffic toward servers).
    reversed_mask = pd.Series(False, index=df.index)

    src_is_hcs = df["src_ip"].isin(hcs_ips)
    dst_is_hcs = df["dst_ip"].isin(hcs_ips)

    # HCS flows: HCS should be src. If HCS is dst and src is not HCS, reversed.
    reversed_mask |= (dst_is_hcs & ~src_is_hcs) & (df["is_hcs"] == True)

    # TGEN flows: TGEN initiator should be src. If TGEN is dst and src is not
    # TGEN, the flow may be reversed. However, we can only safely correct this
    # when we know the TGEN node should be the source. In this test scenario,
    # TGEN traffic generators always initiate, so we apply the same logic.
    src_is_tgen = df["src_ip"].isin(tgen_ips)
    dst_is_tgen = df["dst_ip"].isin(tgen_ips)
    reversed_mask |= (dst_is_tgen & ~src_is_tgen) & (df["is_hcs"] == False)

    n_reversed = int(reversed_mask.sum())
    if n_reversed == 0:
        return df

    print(f"  Correcting {n_reversed} reversed flow(s) from active_timeout split")

    _swap_columns = _get_directional_stat_columns(df)

    for idx in df.index[reversed_mask]:
        # Swap src_ip <-> dst_ip
        src_ip_val = df.at[idx, "src_ip"]
        dst_ip_val = df.at[idx, "dst_ip"]
        df.at[idx, "src_ip"] = dst_ip_val
        df.at[idx, "dst_ip"] = src_ip_val

        # Swap src_port <-> dst_port
        if "src_port" in df.columns and "dst_port" in df.columns:
            src_port_val = df.at[idx, "src_port"]
            dst_port_val = df.at[idx, "dst_port"]
            df.at[idx, "src_port"] = dst_port_val
            df.at[idx, "dst_port"] = src_port_val

        # Swap src_mac <-> dst_mac
        if "src_mac" in df.columns and "dst_mac" in df.columns:
            src_mac_val = df.at[idx, "src_mac"]
            dst_mac_val = df.at[idx, "dst_mac"]
            df.at[idx, "src_mac"] = dst_mac_val
            df.at[idx, "dst_mac"] = src_mac_val

        # Swap src_oui <-> dst_oui
        if "src_oui" in df.columns and "dst_oui" in df.columns:
            src_oui_val = df.at[idx, "src_oui"]
            dst_oui_val = df.at[idx, "dst_oui"]
            df.at[idx, "src_oui"] = dst_oui_val
            df.at[idx, "dst_oui"] = src_oui_val

        # Swap src2dst_* <-> dst2src_* directional statistics
        for src2dst_col, dst2src_col in _swap_columns:
            src2dst_val = df.at[idx, src2dst_col]
            dst2src_val = df.at[idx, dst2src_col]
            df.at[idx, src2dst_col] = dst2src_val
            df.at[idx, dst2src_col] = src2dst_val

        # Swap label-derived columns
        if "src_node" in df.columns and "dst_node" in df.columns:
            src_node_val = df.at[idx, "src_node"]
            dst_node_val = df.at[idx, "dst_node"]
            df.at[idx, "src_node"] = dst_node_val
            df.at[idx, "dst_node"] = src_node_val

        if "src_container_type" in df.columns and "dst_container_type" in df.columns:
            src_ct_val = df.at[idx, "src_container_type"]
            dst_ct_val = df.at[idx, "dst_container_type"]
            df.at[idx, "src_container_type"] = dst_ct_val
            df.at[idx, "dst_container_type"] = src_ct_val

        # Recompute flow_label since src_node/dst_node changed
        # The is_hcs, channel_type, and tgen_type fields are direction-agnostic
        # and don't need to change, but flow_label is derived from them so it
        # should already be correct. However, src_node and dst_node affect
        # nothing in flow_label, so no recompute needed.

    return df


def process_scenario(
    scenario_dir: str,
    output_dir: str,
    active_timeout: int | None = None,
) -> pd.DataFrame:
    """Process a single scenario: extract flows and save outputs.

    Args:
        scenario_dir: Path to the scenario directory under data/.
        output_dir: Root output directory for processed files.
        active_timeout: Optional nfstream active_timeout override (seconds).

    Returns:
        Concatenated DataFrame of all labeled flows for the scenario.
    """
    scenario_name = os.path.basename(scenario_dir.rstrip(os.sep))
    print(f"\nProcessing scenario: {scenario_name}")

    # Load network map
    ip_lookup = load_network_map(scenario_dir)
    print(f"  Loaded network_map.json: {len(ip_lookup)} IPs")

    # Discover pcap groups
    pcap_groups = discover_pcap_groups(scenario_dir)
    print(f"  Discovered {len(pcap_groups)} pcap groups")

    # Extract flows from each group
    all_dfs: list[pd.DataFrame] = []
    for group in pcap_groups:
        print(
            f"  Extracting flows from {group.source_dir} "
            f"({len(group.pcap_paths)} files)..."
        )
        group_df = extract_flows_from_group(group, ip_lookup, active_timeout=active_timeout)
        if not group_df.empty:
            flow_count = len(group_df)
            print(f"    Extracted {flow_count} labeled flows")
            all_dfs.append(group_df)
        else:
            print("    No flows extracted")

    if not all_dfs:
        print(f"  WARNING: No flows extracted for scenario {scenario_name}")
        return pd.DataFrame()

    # Concatenate all flows
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(combined_df)} flows")

    # Save outputs. Suffix the folder with the timeout so distinct
    # active_timeout runs don't overwrite each other's cached flows.
    output_scenario_name = scenario_name
    if active_timeout is not None:
        output_scenario_name = f"{scenario_name}_timeout{active_timeout}"
    scenario_output_dir = os.path.join(output_dir, output_scenario_name)
    os.makedirs(scenario_output_dir, exist_ok=True)

    parquet_path = os.path.join(scenario_output_dir, "flows.parquet")
    csv_path = os.path.join(scenario_output_dir, "flows.csv")

    combined_df.to_parquet(parquet_path, index=False)
    print(f"  Saved parquet: {parquet_path}")

    combined_df.to_csv(csv_path, index=False)
    print(f"  Saved csv: {csv_path}")

    return combined_df


def process_all_scenarios(
    data_dir: str,
    output_dir: str,
    active_timeout: int | None = None,
) -> pd.DataFrame:
    """Process all scenario directories under data_dir.

    Args:
        data_dir: Path to the data directory containing scenario subdirectories.
        output_dir: Root output directory for processed files.
        active_timeout: Optional nfstream active_timeout override (seconds).

    Returns:
        Concatenated DataFrame of all labeled flows across all scenarios.
    """
    scenarios = sorted(
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    if not scenarios:
        print(f"No scenario directories found in {data_dir}")
        return pd.DataFrame()

    all_scenario_dfs: list[pd.DataFrame] = []
    for scenario in scenarios:
        scenario_path = os.path.join(data_dir, scenario)
        df = process_scenario(scenario_path, output_dir, active_timeout=active_timeout)
        if not df.empty:
            all_scenario_dfs.append(df)

    if not all_scenario_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_scenario_dfs, ignore_index=True)
    print(f"\nAll scenarios complete: {len(combined)} total flows")
    return combined
