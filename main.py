#!/usr/bin/env python3
"""CLI entry point for nfstream feature extraction.

Usage:
    python main.py                         # Process all scenarios in data/
    python main.py --scenario scenario1-1  # Process a single scenario
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Extract and label network flows from PCAP files using nfstream."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing scenario subdirectories (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        default="processed",
        help="Output directory for processed flows (default: processed/)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Process a single scenario by name (default: all scenarios)",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(repo_root, args.data_dir)
    output_dir = os.path.join(repo_root, args.output_dir)

    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    if args.scenario:
        scenario_path = os.path.join(data_dir, args.scenario)
        if not os.path.isdir(scenario_path):
            print(f"Error: scenario directory not found: {scenario_path}", file=sys.stderr)
            sys.exit(1)
        from src.extract_flows import process_scenario

        df = process_scenario(scenario_path, output_dir)
    else:
        from src.extract_flows import process_all_scenarios

        df = process_all_scenarios(data_dir, output_dir)

    if df.empty:
        print("\nNo flows extracted.")
        sys.exit(0)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"Total flows: {len(df)}")

    if "is_hcs" in df.columns:
        print("\nis_hcs value counts:")
        print(df["is_hcs"].value_counts(dropna=False).to_string())

    if "channel_type" in df.columns:
        print("\nchannel_type value counts:")
        print(df["channel_type"].value_counts(dropna=False).to_string())

    if "flow_label" in df.columns:
        print("\nflow_label value counts:")
        print(df["flow_label"].value_counts(dropna=False).to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()
