"""Tests for flow directionality correction due to active_timeout.

When nfstream's active_timeout expires a flow, the next packet (possibly
from the server) creates a new flow with reversed src/dst. These tests
verify that _correct_flow_directionality properly detects and fixes
reversed flows.
"""

import pandas as pd
import pytest

from src.extract_flows import (
    _correct_flow_directionality,
    _get_directional_stat_columns,
)
from src.nm_loader import NetworkNodeInfo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ip_lookup():
    """Minimal IP lookup for testing."""
    return {
        "10.1.0.4": NetworkNodeInfo(
            ip="10.1.0.4",
            name="alice_node_type_racetunnel_1",
            container_type="hcs",
            channel_type="racetunnel",
            is_hcs=True,
        ),
        "10.1.0.5": NetworkNodeInfo(
            ip="10.1.0.5",
            name="irc_node_type_racetunnel_1",
            container_type="hcs",
            channel_type="racetunnel",
            is_hcs=True,
        ),
        "10.1.0.2": NetworkNodeInfo(
            ip="10.0.2.2",
            name="router_client_net_racetunnel",
            container_type="infra",
            channel_type=None,
            is_hcs=False,
        ),
        "203.0.2.9": NetworkNodeInfo(
            ip="203.0.2.9",
            name="router_server_net",
            container_type="infra",
            channel_type=None,
            is_hcs=False,
        ),
        "10.1.0.12": NetworkNodeInfo(
            ip="10.1.0.12",
            name="tgen_tgen_type_ftp_1",
            container_type="tgen",
            channel_type="ftp",
            is_hcs=False,
        ),
        "10.2.0.39": NetworkNodeInfo(
            ip="10.2.0.39",
            name="ftp_server",
            container_type="infra",
            channel_type=None,
            is_hcs=False,
        ),
    }


@pytest.fixture
def nfstream_columns():
    """Typical nfstream DataFrame columns we work with."""
    return [
        "id", "src_ip", "src_port", "src_mac", "src_oui",
        "dst_ip", "dst_port", "dst_mac", "dst_oui",
        "protocol", "vlan_id", "tunnel_id",
        "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
        "bidirectional_duration_ms", "bidirectional_packets",
        "bidirectional_bytes",
        "src2dst_first_seen_ms", "src2dst_last_seen_ms",
        "src2dst_duration_ms", "src2dst_packets", "src2dst_bytes",
        "dst2src_first_seen_ms", "dst2src_last_seen_ms",
        "dst2src_duration_ms", "dst2src_packets", "dst2src_bytes",
        "bidirectional_min_ps", "bidirectional_mean_ps",
        "src2dst_min_ps", "src2dst_mean_ps",
        "dst2src_min_ps", "dst2src_mean_ps",
        "src2dst_10_mean_ps", "dst2src_10_mean_ps",  # some stat columns
    ]


@pytest.fixture
def sample_df(nfstream_columns, ip_lookup):
    """Create a sample DataFrame with both correctly-directed and reversed flows."""
    rows = [
        # Flow 1: Correct direction - HCS alice (10.1.0.4) -> server (10.2.0.39)
        {
            "id": 0, "src_ip": "10.1.0.4", "src_port": 50000,
            "src_mac": "aa:bb:cc:dd:ee:01", "src_oui": "aabbcc",
            "dst_ip": "10.2.0.39", "dst_port": 80,
            "dst_mac": "aa:bb:cc:dd:ee:02", "dst_oui": "aabbcc",
            "protocol": 6, "vlan_id": 0, "tunnel_id": 0,
            "bidirectional_first_seen_ms": 1000,
            "bidirectional_last_seen_ms": 2000,
            "bidirectional_duration_ms": 1000,
            "bidirectional_packets": 10,
            "bidirectional_bytes": 5000,
            "src2dst_first_seen_ms": 1000,
            "src2dst_last_seen_ms": 1500,
            "src2dst_duration_ms": 500,
            "src2dst_packets": 5,
            "src2dst_bytes": 2000,
            "dst2src_first_seen_ms": 1500,
            "dst2src_last_seen_ms": 2000,
            "dst2src_duration_ms": 500,
            "dst2src_packets": 5,
            "dst2src_bytes": 3000,
            "bidirectional_min_ps": 100,
            "bidirectional_mean_ps": 500,
            "src2dst_min_ps": 100,
            "src2dst_mean_ps": 400,
            "dst2src_min_ps": 200,
            "dst2src_mean_ps": 600,
            "src2dst_10_mean_ps": 1,
            "dst2src_10_mean_ps": 2,
            # Labels
            "src_node": "alice_node_type_racetunnel_1",
            "dst_node": "ftp_server",
            "src_container_type": "hcs",
            "dst_container_type": "infra",
            "is_hcs": True,
            "channel_type": "racetunnel",
            "tgen_type": None,
            "flow_label": "hcs_racetunnel",
        },
        # Flow 2: Reversed direction - HCS alice is dst, server is src
        # This simulates what happens when active_timeout splits a flow
        # and the next packet arrives from the server side.
        {
            "id": 1, "src_ip": "10.2.0.39", "src_port": 80,
            "src_mac": "aa:bb:cc:dd:ee:02", "src_oui": "aabbcc",
            "dst_ip": "10.1.0.4", "dst_port": 50000,
            "dst_mac": "aa:bb:cc:dd:ee:01", "dst_oui": "aabbcc",
            "protocol": 6, "vlan_id": 0, "tunnel_id": 0,
            "bidirectional_first_seen_ms": 2000,
            "bidirectional_last_seen_ms": 3000,
            "bidirectional_duration_ms": 1000,
            "bidirectional_packets": 8,
            "bidirectional_bytes": 4000,
            "src2dst_first_seen_ms": 2000,  # This is actually server->client
            "src2dst_last_seen_ms": 2500,
            "src2dst_duration_ms": 500,
            "src2dst_packets": 3,  # Only 3 server packets
            "src2dst_bytes": 1500,
            "dst2src_first_seen_ms": 2500,  # This is actually client->server
            "dst2src_last_seen_ms": 3000,
            "dst2src_duration_ms": 500,
            "dst2src_packets": 5,  # 5 client packets
            "dst2src_bytes": 2500,
            "bidirectional_min_ps": 100,
            "bidirectional_mean_ps": 500,
            "src2dst_min_ps": 200,  # server side
            "src2dst_mean_ps": 500,
            "dst2src_min_ps": 100,  # client side
            "dst2src_mean_ps": 500,
            "src2dst_10_mean_ps": 2,
            "dst2src_10_mean_ps": 1,
            # Labels (assigned by label_flow with reversed direction)
            "src_node": "ftp_server",
            "dst_node": "alice_node_type_racetunnel_1",
            "src_container_type": "infra",
            "dst_container_type": "hcs",
            "is_hcs": True,
            "channel_type": "racetunnel",
            "tgen_type": None,
            "flow_label": "hcs_racetunnel",
        },
        # Flow 3: TGEN flow, reversed direction
        # TGEN node should be src (initiator), but ended up as dst
        {
            "id": 2, "src_ip": "10.2.0.39", "src_port": 21,
            "src_mac": "aa:bb:cc:dd:ee:03", "src_oui": "aabb01",
            "dst_ip": "10.1.0.12", "dst_port": 40000,
            "dst_mac": "aa:bb:cc:dd:ee:04", "dst_oui": "aabb02",
            "protocol": 6, "vlan_id": 0, "tunnel_id": 0,
            "bidirectional_first_seen_ms": 5000,
            "bidirectional_last_seen_ms": 6000,
            "bidirectional_duration_ms": 1000,
            "bidirectional_packets": 6,
            "bidirectional_bytes": 3000,
            "src2dst_first_seen_ms": 5000,
            "src2dst_last_seen_ms": 5500,
            "src2dst_duration_ms": 500,
            "src2dst_packets": 2,
            "src2dst_bytes": 1000,
            "dst2src_first_seen_ms": 5500,
            "dst2src_last_seen_ms": 6000,
            "dst2src_duration_ms": 500,
            "dst2src_packets": 4,
            "dst2src_bytes": 2000,
            "bidirectional_min_ps": 100,
            "bidirectional_mean_ps": 500,
            "src2dst_min_ps": 200,
            "src2dst_mean_ps": 500,
            "dst2src_min_ps": 100,
            "dst2src_mean_ps": 500,
            "src2dst_10_mean_ps": 2,
            "dst2src_10_mean_ps": 1,
            # Labels - reversed
            "src_node": "ftp_server",
            "dst_node": "tgen_tgen_type_ftp_1",
            "src_container_type": "infra",
            "dst_container_type": "tgen",
            "is_hcs": False,
            "channel_type": "ftp",
            "tgen_type": "FTP",
            "flow_label": "tgen_ftp",
        },
    ]
    df = pd.DataFrame(rows)
    # Ensure all expected columns exist (some may be NaN in our test)
    for col in nfstream_columns:
        if col not in df.columns:
            df[col] = None
    return df


# ---------------------------------------------------------------------------
# Tests for _get_directional_stat_columns
# ---------------------------------------------------------------------------

class TestGetDirectionalStatColumns:
    def test_finds_src2dst_dst2src_pairs(self, sample_df):
        pairs = _get_directional_stat_columns(sample_df)
        pair_names = [(s, d) for s, d in pairs]
        assert ("src2dst_packets", "dst2src_packets") in pair_names
        assert ("src2dst_bytes", "dst2src_bytes") in pair_names
        assert ("src2dst_duration_ms", "dst2src_duration_ms") in pair_names
        assert ("src2dst_first_seen_ms", "dst2src_first_seen_ms") in pair_names
        assert ("src2dst_last_seen_ms", "dst2src_last_seen_ms") in pair_names
        assert ("src2dst_min_ps", "dst2src_min_ps") in pair_names
        assert ("src2dst_mean_ps", "dst2src_mean_ps") in pair_names
        assert ("src2dst_10_mean_ps", "dst2src_10_mean_ps") in pair_names

    def test_excludes_bidirectional_columns(self, sample_df):
        pairs = _get_directional_stat_columns(sample_df)
        all_cols = set()
        for s, d in pairs:
            all_cols.add(s)
            all_cols.add(d)
        # bidirectional columns should not appear in directional pairs
        assert "bidirectional_packets" not in all_cols
        assert "bidirectional_bytes" not in all_cols

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        pairs = _get_directional_stat_columns(df)
        assert pairs == []


# ---------------------------------------------------------------------------
# Tests for _correct_flow_directionality
# ---------------------------------------------------------------------------

class TestCorrectFlowDirectionality:
    def test_empty_dataframe(self, ip_lookup):
        df = pd.DataFrame()
        result = _correct_flow_directionality(df, ip_lookup)
        assert result.empty

    def test_no_reversed_flows(self, sample_df, ip_lookup):
        """Flow 0 has correct direction; should not be modified."""
        # Isolate only the correctly-directed flow
        df = sample_df[sample_df["id"] == 0].copy()
        result = _correct_flow_directionality(df, ip_lookup)

        # src_ip should still be the HCS node
        assert result.iloc[0]["src_ip"] == "10.1.0.4"
        assert result.iloc[0]["dst_ip"] == "10.2.0.39"
        # src2dst_packets should still be the client->server count
        assert result.iloc[0]["src2dst_packets"] == 5
        assert result.iloc[0]["dst2src_packets"] == 5

    def test_corrects_reversed_hcs_flow(self, sample_df, ip_lookup):
        """Flow 1 is reversed (server is src, HCS is dst). Should be corrected."""
        df = sample_df[sample_df["id"] == 1].copy()
        result = _correct_flow_directionality(df, ip_lookup)

        # After correction, HCS node should be src
        assert result.iloc[0]["src_ip"] == "10.1.0.4"  # HCS node
        assert result.iloc[0]["dst_ip"] == "10.2.0.39"  # server

        # Ports should be swapped
        assert result.iloc[0]["src_port"] == 50000  # client port
        assert result.iloc[0]["dst_port"] == 80  # server port

        # MACs should be swapped
        assert result.iloc[0]["src_mac"] == "aa:bb:cc:dd:ee:01"  # client mac
        assert result.iloc[0]["dst_mac"] == "aa:bb:cc:dd:ee:02"  # server mac

        # Directional stats should be swapped
        # In the original (reversed) flow, src2dst was server->client (3 packets)
        # and dst2src was client->server (5 packets).
        # After swap, src2dst should be client->server (5 packets)
        assert result.iloc[0]["src2dst_packets"] == 5
        assert result.iloc[0]["dst2src_packets"] == 3
        assert result.iloc[0]["src2dst_bytes"] == 2500
        assert result.iloc[0]["dst2src_bytes"] == 1500

    def test_corrects_reversed_hcs_labels(self, sample_df, ip_lookup):
        """src_node and dst_node should be swapped for reversed HCS flows."""
        df = sample_df[sample_df["id"] == 1].copy()
        result = _correct_flow_directionality(df, ip_lookup)

        # After correction, HCS node should be the src_node
        assert result.iloc[0]["src_node"] == "alice_node_type_racetunnel_1"
        assert result.iloc[0]["dst_node"] == "ftp_server"
        assert result.iloc[0]["src_container_type"] == "hcs"
        assert result.iloc[0]["dst_container_type"] == "infra"

    def test_corrects_reversed_tgen_flow(self, sample_df, ip_lookup):
        """Flow 2 is reversed TGEN flow (infra is src, TGEN is dst). Should be corrected."""
        df = sample_df[sample_df["id"] == 2].copy()
        result = _correct_flow_directionality(df, ip_lookup)

        # After correction, TGEN node should be src
        assert result.iloc[0]["src_ip"] == "10.1.0.12"  # TGEN node
        assert result.iloc[0]["dst_ip"] == "10.2.0.39"  # server

        # Directional stats should be swapped
        assert result.iloc[0]["src2dst_packets"] == 4  # was dst2src
        assert result.iloc[0]["dst2src_packets"] == 2  # was src2dst

        # Labels should be swapped
        assert result.iloc[0]["src_node"] == "tgen_tgen_type_ftp_1"
        assert result.iloc[0]["dst_node"] == "ftp_server"
        assert result.iloc[0]["src_container_type"] == "tgen"
        assert result.iloc[0]["dst_container_type"] == "infra"

    def test_corrects_multiple_reversed_flows(self, sample_df, ip_lookup):
        """All reversed flows in a multi-flow DataFrame should be corrected."""
        result = _correct_flow_directionality(sample_df, ip_lookup)

        # Flow 0: was correct, should remain unchanged
        row0 = result[result["id"] == 0].iloc[0]
        assert row0["src_ip"] == "10.1.0.4"
        assert row0["dst_ip"] == "10.2.0.39"
        assert row0["src2dst_packets"] == 5
        assert row0["dst2src_packets"] == 5

        # Flow 1: was reversed (HCS), should be corrected
        row1 = result[result["id"] == 1].iloc[0]
        assert row1["src_ip"] == "10.1.0.4"  # HCS node now src
        assert row1["dst_ip"] == "10.2.0.39"
        assert row1["src2dst_packets"] == 5  # client->server count
        assert row1["dst2src_packets"] == 3  # server->client count
        assert row1["src_node"] == "alice_node_type_racetunnel_1"
        assert row1["dst_node"] == "ftp_server"

        # Flow 2: was reversed (TGEN), should be corrected
        row2 = result[result["id"] == 2].iloc[0]
        assert row2["src_ip"] == "10.1.0.12"  # TGEN node now src
        assert row2["dst_ip"] == "10.2.0.39"
        assert row2["src2dst_packets"] == 4  # was dst2src (client side)
        assert row2["dst2src_packets"] == 2  # was src2dst (server side)
        assert row2["src_node"] == "tgen_tgen_type_ftp_1"
        assert row2["dst_node"] == "ftp_server"

    def test_bidirectional_stats_not_swapped(self, sample_df, ip_lookup):
        """Bidirectional statistics should NOT be swapped (they're direction-agnostic)."""
        df = sample_df[sample_df["id"] == 1].copy()
        result = _correct_flow_directionality(df, ip_lookup)

        # Bidirectional stats should remain unchanged
        assert result.iloc[0]["bidirectional_packets"] == 8
        assert result.iloc[0]["bidirectional_bytes"] == 4000
        assert result.iloc[0]["bidirectional_duration_ms"] == 1000
        assert result.iloc[0]["bidirectional_first_seen_ms"] == 2000
        assert result.iloc[0]["bidirectional_last_seen_ms"] == 3000
        assert result.iloc[0]["bidirectional_min_ps"] == 100
        assert result.iloc[0]["bidirectional_mean_ps"] == 500

    def test_does_not_flip_correctly_directed_hcs_flow(self, ip_lookup):
        """When HCS is already src, no correction should happen."""
        df = pd.DataFrame([{
            "id": 0, "src_ip": "10.1.0.4", "src_port": 50000,
            "src_mac": "aa:bb:cc:dd:ee:01", "src_oui": "aabbcc",
            "dst_ip": "10.2.0.39", "dst_port": 80,
            "dst_mac": "aa:bb:cc:dd:ee:02", "dst_oui": "aabbcc",
            "protocol": 6, "vlan_id": 0, "tunnel_id": 0,
            "src2dst_packets": 5, "src2dst_bytes": 2000,
            "dst2src_packets": 5, "dst2src_bytes": 3000,
            "is_hcs": True, "channel_type": "racetunnel",
            "src_node": "alice_node_type_racetunnel_1",
            "dst_node": "ftp_server",
            "src_container_type": "hcs", "dst_container_type": "infra",
            "flow_label": "hcs_racetunnel",
        }])
        result = _correct_flow_directionality(df, ip_lookup)
        assert result.iloc[0]["src_ip"] == "10.1.0.4"
        assert result.iloc[0]["dst_ip"] == "10.2.0.39"
        assert result.iloc[0]["src2dst_packets"] == 5
        assert result.iloc[0]["dst2src_packets"] == 5

    def test_does_not_correct_infra_only_flows(self, ip_lookup):
        """Flows with neither HCS nor TGEN endpoints should not be corrected."""
        df = pd.DataFrame([{
            "id": 0,
            "src_ip": "203.0.2.9", "src_port": 53,
            "dst_ip": "10.2.0.39", "dst_port": 80,
            "is_hcs": None,
            "flow_label": "infra_unknown",
        }])
        result = _correct_flow_directionality(df, ip_lookup)
        # Should not flip since neither endpoint is HCS or TGEN
        assert result.iloc[0]["src_ip"] == "203.0.2.9"
        assert result.iloc[0]["dst_ip"] == "10.2.0.39"

    def test_handles_missing_columns_gracefully(self, ip_lookup):
        """Should not crash if optional columns are missing."""
        df = pd.DataFrame([{
            "id": 0, "src_ip": "10.1.0.4", "dst_ip": "10.2.0.39",
            "is_hcs": True, "flow_label": "hcs_racetunnel",
        }])
        result = _correct_flow_directionality(df, ip_lookup)
        # HCS is src, so no correction needed
        assert result.iloc[0]["src_ip"] == "10.1.0.4"
        assert result.iloc[0]["dst_ip"] == "10.2.0.39"

    def test_preserves_row_count(self, sample_df, ip_lookup):
        """Directionality correction should not add/remove rows."""
        result = _correct_flow_directionality(sample_df, ip_lookup)
        assert len(result) == len(sample_df)

    def test_preserves_other_columns(self, sample_df, ip_lookup):
        """Non-directional columns should be untouched."""
        result = _correct_flow_directionality(sample_df, ip_lookup)
        # application_name and similar should still exist
        # protocol should not change
        for idx in result.index:
            assert result.at[idx, "protocol"] == sample_df.at[idx, "protocol"]
