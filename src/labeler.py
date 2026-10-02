"""Flow labeling logic for resolving IPs to node labels.

Given an IP-to-node lookup from network_map.json, labels each flow row
with HCS/TGEN classification, channel type, and node names.
"""

import re
from dataclasses import dataclass

from .nm_loader import NetworkNodeInfo


@dataclass
class FlowLabel:
    """Label information for a single flow."""

    is_hcs: bool | None
    channel_type: str | None
    src_node: str
    dst_node: str
    src_container_type: str
    dst_container_type: str


def label_flow(
    src_ip: str,
    dst_ip: str,
    ip_lookup: dict[str, NetworkNodeInfo],
) -> FlowLabel:
    """Label a flow based on its source and destination IPs.

    Resolution priority:
    1. If src_ip is HCS -> label as HCS, channel_type from src
    2. If src_ip is TGEN -> label as TGEN, channel_type from src
    3. If dst_ip is HCS -> label as HCS, channel_type from dst (reverse direction)
    4. If dst_ip is TGEN -> label as TGEN, channel_type from dst (reverse direction)
    5. Otherwise -> label as infra (is_hcs=None)

    Args:
        src_ip: Source IP address string.
        dst_ip: Destination IP address string.
        ip_lookup: Dict mapping IP -> NetworkNodeInfo from load_network_map().

    Returns:
        FlowLabel with all label fields populated.
    """
    src_info = ip_lookup.get(src_ip)
    dst_info = ip_lookup.get(dst_ip)

    if src_info and src_info.is_hcs:
        return FlowLabel(
            is_hcs=True,
            channel_type=src_info.channel_type,
            src_node=src_info.name,
            dst_node=dst_info.name if dst_info else "unknown",
            src_container_type="hcs",
            dst_container_type=dst_info.container_type if dst_info else "unknown",
        )

    if src_info and src_info.container_type == "tgen":
        return FlowLabel(
            is_hcs=False,
            channel_type=src_info.channel_type,
            src_node=src_info.name,
            dst_node=dst_info.name if dst_info else "unknown",
            src_container_type="tgen",
            dst_container_type=dst_info.container_type if dst_info else "unknown",
        )

    if dst_info and dst_info.is_hcs:
        return FlowLabel(
            is_hcs=True,
            channel_type=dst_info.channel_type,
            src_node=src_info.name if src_info else "unknown",
            dst_node=dst_info.name,
            src_container_type=src_info.container_type if src_info else "unknown",
            dst_container_type="hcs",
        )

    if dst_info and dst_info.container_type == "tgen":
        return FlowLabel(
            is_hcs=False,
            channel_type=dst_info.channel_type,
            src_node=src_info.name if src_info else "unknown",
            dst_node=dst_info.name,
            src_container_type=src_info.container_type if src_info else "unknown",
            dst_container_type="tgen",
        )

    # Neither endpoint is HCS or TGEN
    return FlowLabel(
        is_hcs=None,
        channel_type=None,
        src_node=src_info.name if src_info else "unknown",
        dst_node=dst_info.name if dst_info else "unknown",
        src_container_type=src_info.container_type if src_info else "unknown",
        dst_container_type=dst_info.container_type if dst_info else "unknown",
    )


def is_dns_traffic(row) -> bool:
    """Check if a flow row represents DNS traffic.

    Detection methods (in priority order):
    1. nDPI application_name starts with "DNS"
    2. UDP protocol with port 53 on either end
    3. DNS-related nDPI application category

    Args:
        row: A pandas DataFrame row (Series).

    Returns:
        True if the flow appears to be DNS traffic.
    """
    # Primary: nDPI application identification
    app_name = row.get("application_name")
    if app_name and isinstance(app_name, str) and app_name.lower().startswith("dns"):
        return True

    # Check nDPI category
    app_category = row.get("application_category_name")
    if app_category and isinstance(app_category, str) and "dns" in app_category.lower():
        return True

    # Fallback: port 53 over UDP
    protocol = row.get("protocol")
    if protocol == 17:  # UDP
        dst_port = row.get("dst_port")
        src_port = row.get("src_port")
        if dst_port == 53 or src_port == 53:
            return True

    return False


# Regex to detect if a node name contains "dns" at the start (DNS infrastructure)
_DNS_NODE_RE = re.compile(r"^dns")


def is_dns_node_traffic(
    src_node: str,
    dst_node: str,
) -> bool:
    """Check if both endpoints are DNS infrastructure nodes.

    Args:
        src_node: Source node name.
        dst_node: Destination node name.

    Returns:
        True if both nodes are DNS infrastructure nodes.
    """
    src_is_dns = bool(_DNS_NODE_RE.match(src_node)) if src_node else False
    dst_is_dns = bool(_DNS_NODE_RE.match(dst_node)) if dst_node else False
    return src_is_dns and dst_is_dns
