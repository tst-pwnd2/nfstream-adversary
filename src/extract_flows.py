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


def extract_flows_from_group(
    group: PCAPGroup,
    ip_lookup: dict[str, NetworkNodeInfo],
) -> pd.DataFrame:
    """Extract flows from a single pcap group and apply labels.

    Args:
        group: PCAPGroup containing pcap paths and network name.
        ip_lookup: IP-to-node lookup from load_network_map().

    Returns:
        DataFrame with nfstream features plus labeling columns.
    """
    if NFStreamer is None:
        raise ImportError(
            "nfstream is not installed. Install requirements: pip install -r requirements.txt"
        )

    try:
        streamer = NFStreamer(
            source=group.pcap_paths,
            statistical_analysis=True,
            n_dissections=20,
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


def process_scenario(scenario_dir: str, output_dir: str) -> pd.DataFrame:
    """Process a single scenario: extract flows and save outputs.

    Args:
        scenario_dir: Path to the scenario directory under data/.
        output_dir: Root output directory for processed files.

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
        group_df = extract_flows_from_group(group, ip_lookup)
        if not group_df.empty:
            flow_count = len(group_df)
            print(f"    Extracted {flow_count} labeled flows")
            all_dfs.append(group_df)
        else:
            print(f"    No flows extracted")

    if not all_dfs:
        print(f"  WARNING: No flows extracted for scenario {scenario_name}")
        return pd.DataFrame()

    # Concatenate all flows
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(combined_df)} flows")

    # Save outputs
    scenario_output_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(scenario_output_dir, exist_ok=True)

    parquet_path = os.path.join(scenario_output_dir, "flows.parquet")
    csv_path = os.path.join(scenario_output_dir, "flows.csv")

    combined_df.to_parquet(parquet_path, index=False)
    print(f"  Saved parquet: {parquet_path}")

    combined_df.to_csv(csv_path, index=False)
    print(f"  Saved csv: {csv_path}")

    return combined_df


def process_all_scenarios(data_dir: str, output_dir: str) -> pd.DataFrame:
    """Process all scenario directories under data_dir.

    Args:
        data_dir: Path to the data directory containing scenario subdirectories.
        output_dir: Root output directory for processed files.

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
        df = process_scenario(scenario_path, output_dir)
        if not df.empty:
            all_scenario_dfs.append(df)

    if not all_scenario_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_scenario_dfs, ignore_index=True)
    print(f"\nAll scenarios complete: {len(combined)} total flows")
    return combined
