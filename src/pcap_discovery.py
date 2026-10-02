"""PCAP file discovery and grouping for split capture files.

PCAP files can be split into multiple parts: ``foo.pcap`` followed by
``foo.pcap1``, ``foo.pcap2``, etc. NFStreamer accepts a list of paths
to process these as a single logical stream.
"""

import os
import re
from dataclasses import dataclass, field


@dataclass
class PCAPGroup:
    """A group of pcap files representing a single logical capture stream."""

    pcap_paths: list[str]
    network_name: str
    source_dir: str


def discover_pcap_groups(scenario_dir: str) -> list[PCAPGroup]:
    """Discover and group split pcap files under a scenario directory.

    Walks all subdirectories of ``scenario_dir`` looking for ``.pcap`` files.
    Files named ``foo.pcap`` followed by ``foo.pcap1``, ``foo.pcap2``, etc.
    are grouped together.

    Args:
        scenario_dir: Path to the scenario directory (e.g., data/scenario1-1).

    Returns:
        List of PCAPGroup objects, each containing the ordered list of
        pcap file paths for a logical capture and the network name.
    """
    pcap_groups: list[PCAPGroup] = []

    # Sort subdirectories for deterministic processing order.
    subdirs = sorted(
        d
        for d in os.listdir(scenario_dir)
        if os.path.isdir(os.path.join(scenario_dir, d))
    )

    for subdir_name in subdirs:
        subdir_path = os.path.join(scenario_dir, subdir_name)

        # Find all pcap files (.pcap, .pcap1, .pcap2, etc.) in this subdirectory.
        files = os.listdir(subdir_path)
        pcap_files = sorted(
            f for f in files if re.match(r".+\.pcap\d*$", f)
        )

        if not pcap_files:
            continue

        # Derive network name from the first pcap filename in this group.
        network_name = _extract_network_name_from_pcap(pcap_files[0])

        # Group split files: base.pcap + base.pcap1 + base.pcap2 + ...
        grouped = _group_split_pcap_files(pcap_files)

        for base_name, file_list in grouped.items():
            full_paths = [
                os.path.join(subdir_path, fname) for fname in _sort_pcap_files(file_list)
            ]
            pcap_groups.append(
                PCAPGroup(
                    pcap_paths=full_paths,
                    network_name=network_name,
                    source_dir=subdir_name,
                )
            )

    return pcap_groups


def _sort_pcap_files(file_list: list[str]) -> list[str]:
    """Sort pcap files so the base .pcap comes first, then .pcap1, .pcap2, etc.

    Args:
        file_list: List of pcap filenames (same base, including split suffixes).

    Returns:
        Sorted list with base file first, then splits in numeric order.
    """
    def sort_key(fname: str) -> tuple[int, str]:
        # Base file ends with .pcap (no digits after)
        match = re.match(r"^(.+\.pcap)(\d+)$", fname)
        if match:
            return (1, match.group(2))
        else:
            return (0, fname)

    return sorted(file_list, key=sort_key)


def _extract_network_name_from_pcap(pcap_filename: str) -> str:
    """Extract the network name from a pcap filename.

    Pcap filenames follow the pattern:
        <date>_<time>_<network_identifier>_ens6.pcap[.N]

    The network identifier is either ``router_<network_name>`` or an
    interface name like ``ixp-router``.

    Examples:
        20260805_051246_router_client_net_racetunnel_ens6.pcap
            -> client_net_racetunnel
        20260805_051240_ixp-router_ens6.pcap
            -> ixp
        20260805_051246_router_server_net_ens6.pcap
            -> server_net

    Args:
        pcap_filename: The pcap filename (or path).

    Returns:
        The extracted network name.
    """
    basename = os.path.basename(pcap_filename)

    # Remove split suffix if present (e.g., .pcap1 -> .pcap)
    basename = re.sub(r"\.pcap(\d+)$", ".pcap", basename)

    # Remove the _ens6.pcap suffix
    basename = re.sub(r"_ens6\.pcap$", "", basename)

    # Split by underscore to remove the date_time prefix
    parts = basename.split("_")

    # Remove the first two parts (date and time), e.g., ["20260805", "051246"]
    if len(parts) >= 3:
        # The remaining parts form the network identifier
        identifier_parts = parts[2:]
        identifier = "_".join(identifier_parts)
    else:
        identifier = basename

    # Strip the "router_" prefix if present
    if identifier.startswith("router_"):
        return identifier[len("router_"):]

    # Handle names like "ixp-router" -> "ixp"
    if "-router" in identifier:
        return identifier.replace("-router", "")

    return identifier


def _group_split_pcap_files(pcap_files: list[str]) -> dict[str, list[str]]:
    """Group split pcap files by their base name.

    A file ``foo.pcap`` is the base, and ``foo.pcap1``, ``foo.pcap2``, etc.
    are its split parts.

    Args:
        pcap_files: List of pcap filename strings (already filtered to pcap files).

    Returns:
        Dict mapping base name to list of filenames (including base + splits).
    """
    groups: dict[str, list[str]] = {}

    # Pattern for split files: <base>.pcap<digits>
    split_re = re.compile(r"^(.+\.pcap)(\d+)$")

    for fname in pcap_files:
        match = split_re.match(fname)
        if match:
            base = match.group(1)
            groups.setdefault(base, []).append(fname)
        else:
            # Base file - ensure it's in its own group
            if fname not in groups:
                groups.setdefault(fname, [fname])

    return groups
