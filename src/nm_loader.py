"""Loader and parser for network_map.json files.

Parses the nested JSON structure into a flat IP-to-node lookup, classifying
each node as HCS, TGEN, or infrastructure.
"""

import json
import os
import re
from dataclasses import dataclass


@dataclass
class NetworkNodeInfo:
    """Information about a single network node resolved from network_map.json."""

    ip: str
    name: str
    container_type: str  # "hcs", "tgen", "infra"
    channel_type: str | None  # e.g., "racetunnel", "mastodon", or None for infra
    is_hcs: bool


_HCS_PREFIXES = (
    "alice_node_type_",
    "irc_node_type_",
    "bob_node_type_",
    "reverse_proxy_bob_node_type_",
)
_TGEN_PREFIX = "tgen_tgen_type_"

# Regex patterns to extract channel type from HCS node names.
# Matches alice_node_type_<channel>_<n>, irc_node_type_<channel>_<n>, etc.
_HCS_CHANNEL_RE = re.compile(
    r"^(?:alice|irc|bob|reverse_proxy_bob)_node_type_(.+?)_(\d+)$"
)
# Matches tgen_tgen_type_<type>_<n> or tgen_tgen_type_<type>_monitor_<n>
_TGEN_CHANNEL_RE = re.compile(r"^tgen_tgen_type_(.+?)(?:_(\d+)|_monitor_(\d+))$")


def _classify_node(node_name: str) -> tuple[str, str | None, bool]:
    """Classify a node name into container_type, channel_type, and is_hcs.

    Returns:
        Tuple of (container_type, channel_type, is_hcs).
    """
    if node_name.startswith(_HCS_PREFIXES):
        match = _HCS_CHANNEL_RE.match(node_name)
        if match:
            return "hcs", match.group(1), True
        return "hcs", None, True

    if node_name.startswith(_TGEN_PREFIX):
        match = _TGEN_CHANNEL_RE.match(node_name)
        if match:
            return "tgen", match.group(1), False
        return "tgen", None, False

    return "infra", None, False


def load_network_map(scenario_dir: str) -> dict[str, NetworkNodeInfo]:
    """Load network_map.json and build an IP-to-node lookup.

    Args:
        scenario_dir: Path to the scenario directory containing network_map.json.

    Returns:
        Dict mapping IP address string to NetworkNodeInfo.

    Raises:
        FileNotFoundError: If network_map.json does not exist.
    """
    map_path = os.path.join(scenario_dir, "network_map.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"network_map.json not found in {scenario_dir}")

    with open(map_path) as f:
        network_map = json.load(f)

    ip_lookup: dict[str, NetworkNodeInfo] = {}

    for network_name, network_data in network_map.items():
        container_info = network_data.get("container_info", {})
        for node_name, ip_addr in container_info.items():
            container_type, channel_type, is_hcs = _classify_node(node_name)
            ip_lookup[ip_addr] = NetworkNodeInfo(
                ip=ip_addr,
                name=node_name,
                container_type=container_type,
                channel_type=channel_type,
                is_hcs=is_hcs,
            )

    return ip_lookup
