#!/usr/bin/env python3
"""CLI entry point for nfstream-adversary.

Usage:
    python main.py                         # Process all scenarios in data/
    python main.py --scenario scenario1-1  # Extract flows for a single scenario

    python main.py analyze                         # Full classifier evaluation
    python main.py analyze --channel mastodon       # Single channel only

    python main.py experiment                       # Run experiment with plan
    python main.py experiment --plan experiments/default.json
    python main.py experiment --subsets full,minimal --classifiers logreg,random_forest
"""

import argparse
import json
import os
import sys


def cmd_extract(args):
    """Extract flows from PCAP files."""
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


def cmd_analyze(args):
    """Run classifier evaluation on extracted flows (legacy interface)."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(repo_root, args.output_dir)

    scenario_name = args.scenario or "scenario1-1"
    parquet_path = os.path.join(output_dir, scenario_name, "flows.parquet")

    if not os.path.exists(parquet_path):
        print(f"Error: flows parquet not found: {parquet_path}", file=sys.stderr)
        print("Run 'python main.py --scenario " + scenario_name + "' first.", file=sys.stderr)
        sys.exit(1)

    results_dir = os.path.join(output_dir, scenario_name, "results")
    os.makedirs(results_dir, exist_ok=True)

    from src.pipeline import run_full_evaluation, results_to_dataframe
    from src.data_prep import load_flows, get_available_channels
    from src.classifiers import CLASSIFIER_NAMES
    from src.features import FEATURE_SUBSET_NAMES

    print(f"Loading flows from {parquet_path}...")
    df = load_flows(parquet_path)
    print(f"  Loaded {len(df)} flows")

    if args.channel:
        channels = [args.channel]
    elif args.pooled_only:
        channels = [None]
    else:
        available = get_available_channels(df)
        channels = [None] + available
    print(f"  Channels to evaluate: {[c or 'pooled' for c in channels]}")

    if args.subset:
        subsets = [args.subset]
    else:
        subsets = FEATURE_SUBSET_NAMES

    if args.classifier:
        classifiers = [args.classifier]
    else:
        classifiers = CLASSIFIER_NAMES

    print(f"  Feature subsets: {subsets}")
    print(f"  Classifiers: {classifiers}")

    max_samples = 50000 if args.max_samples is None else args.max_samples
    print(f"  Max samples per run: {max_samples}")

    results = run_full_evaluation(
        df, channels, subsets, classifiers, n_splits=5, max_samples=max_samples
    )

    results_df = results_to_dataframe(results)

    auroc_csv = os.path.join(results_dir, "auroc_results.csv")
    results_df.to_csv(auroc_csv, index=False)
    print(f"\nSaved AUROC results: {auroc_csv}")

    summary_cols = [
        "channel", "feature_subset", "classifier",
        "mean_auroc", "std_auroc", "n_positive", "n_negative",
        "n_folds", "warning",
    ]
    summary_df = results_df[summary_cols]
    summary_csv = os.path.join(results_dir, "auroc_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved AUROC summary: {summary_csv}")

    print("\n" + "=" * 80)
    print("AUROC Summary (mean +/- std)")
    print("=" * 80)
    print(f"{'Channel':<15} {'Subset':<18} {'Classifier':<18} {'AUROC':>12} {'Samples':>12} {'Folds':>5}")
    print("-" * 80)
    for r in results:
        if r.n_folds > 0:
            print(f"{r.channel:<15} {r.feature_subset:<18} {r.classifier:<18} "
                  f"{r.mean_auroc:.4f}+/-{r.std_auroc:.4f}  "
                  f"{r.n_positive+r.n_negative:>12} {r.n_folds:>5}")
        else:
            print(f"{r.channel:<15} {r.feature_subset:<18} {r.classifier:<18} "
                  f"{'N/A':>12} {'':>12} {r.n_folds:>5}  {r.warning or ''}")

    if not args.no_importance and any(r.n_folds > 0 for r in results):
        print("\n" + "=" * 60)
        print("Feature Importance Analysis")
        print("=" * 60)

        from src.feature_importance import compute_all_importance, aggregate_importance

        imp_dir = os.path.join(results_dir, "feature_importance")
        os.makedirs(imp_dir, exist_ok=True)

        imp_results = compute_all_importance(
            df, channels, classifiers,
            feature_subset="full", top_n=20,
            output_dir=imp_dir,
            max_samples=max_samples,
        )

        agg = aggregate_importance(imp_results, top_n=20)
        agg_path = os.path.join(results_dir, "feature_importance_summary.csv")
        agg.to_csv(agg_path, index=False)
        print(f"Saved aggregated importance: {agg_path}")

        if not agg.empty:
            print("\nTop 20 Features (aggregated mean rank):")
            print(agg.to_string(index=False))

    print("\nDone!")


def cmd_experiment(args):
    """Run experiment plan with comprehensive outputs and visualizations."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(repo_root, args.output_dir)

    # Load experiment plan
    if args.plan:
        from src.experiment import load_experiment_plan
        plan = load_experiment_plan(args.plan)
        print(f"Loaded experiment plan from {args.plan}")
    else:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        default_plan_path = os.path.join(repo_root, "experiments", "default.json")
        if os.path.exists(default_plan_path):
            from src.experiment import load_experiment_plan
            plan = load_experiment_plan(default_plan_path)
            print(f"Loaded default plan from {default_plan_path}")
        else:
            from src.experiment import default_experiment_plan
            plan = default_experiment_plan()
            print("Using inline default experiment plan")

    # Override with CLI args if provided
    if args.subsets:
        plan["feature_subsets"] = args.subsets.split(",")
    if args.classifiers:
        plan["classifiers"] = args.classifiers.split(",")
    if args.channel:
        plan["channels"] = [args.channel]
        plan["pooled"] = False
    if args.pooled_only:
        plan["channels"] = []
        plan["pooled"] = True
    if args.max_samples is not None:
        plan["max_samples"] = args.max_samples
    if args.no_importance:
        plan["skip_importance"] = True
    if args.no_plots:
        plan["skip_plots"] = True
    if args.balanced:
        plan["balanced"] = True
    if args.tgen_types:
        plan["tgen_types"] = args.tgen_types.split(",")
    if args.paired:
        pairs = []
        for pair_str in args.paired.split(","):
            parts = pair_str.split(":")
            if len(parts) == 2:
                hcs_val = parts[0].split("+") if "+" in parts[0] else parts[0]
                tgen_val = parts[1].split("+") if "+" in parts[1] else parts[1]
                pairs.append({"hcs": hcs_val, "tgen": tgen_val})
        plan["paired_evaluations"] = pairs

    scenario_name = plan.get("scenario", args.scenario or "scenario1-1")
    parquet_path = os.path.join(output_dir, scenario_name, "flows.parquet")

    if not os.path.exists(parquet_path):
        print(f"Error: flows parquet not found: {parquet_path}", file=sys.stderr)
        print("Run 'python main.py --scenario " + scenario_name + "' first.", file=sys.stderr)
        sys.exit(1)

    # Set up output directory
    exp_name = plan.get("name", "experiment")
    results_dir = os.path.join(output_dir, scenario_name, "results", "experiments", exp_name)
    os.makedirs(results_dir, exist_ok=True)

    from src.data_prep import load_flows

    print(f"\nExperiment: {exp_name}")
    print(f"Description: {plan.get('description', '')}")
    print(f"Channels: {plan.get('channels', [])} (pooled: {plan.get('pooled', False)})")
    print(f"Feature subsets: {plan.get('feature_subsets', [])}")
    print(f"Classifiers: {plan.get('classifiers', [])}")
    print(f"CV folds: {plan.get('n_splits', 5)}")
    print(f"Max samples: {plan.get('max_samples', 50000)}")

    # Load data
    print(f"\nLoading flows from {parquet_path}...")
    df = load_flows(parquet_path)
    print(f"  Loaded {len(df)} flows")

    # Run experiment
    from src.experiment import run_experiment
    results_df = run_experiment(df, plan, results_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("Experiment Results Summary")
    print("=" * 80)
    print(f"Total evaluations: {len(results_df)}")
    print(f"Successful (with CV): {(results_df['n_folds'] > 0).sum()}")
    print(f"Failed/skipped: {(results_df['n_folds'] == 0).sum()}")

    # Print per-channel best results
    print("\n" + "-" * 80)
    print("Best AUROC per channel (any subset/classifier):")
    print("-" * 80)
    for channel in results_df["channel"].unique():
        ch_df = results_df[
            (results_df["channel"] == channel) & (results_df["n_folds"] > 0)
        ]
        if not ch_df.empty:
            best = ch_df.loc[ch_df["mean_auroc"].idxmax()]
            print(
                f"  {channel:<15} {best['classifier']:<18} "
                f"{best['feature_subset']:<18} "
                f"AUROC={best['mean_auroc']:.4f} +/- {best['std_auroc']:.4f}  "
                f"train: {best['train_n_positive']}/{best['train_n_negative']}, "
                f"test: {best['test_n_positive']}/{best['test_n_negative']}"
            )

    print(f"\nResults saved to: {results_dir}")
    print("  - results.csv (full CV results with per-fold scores)")
    print("  - results.json (JSON format)")
    print("  - sample_counts.csv (train/test sample counts)")
    print("  - feature_importance/*.csv (per channel/classifier)")
    print("  - plots/ (PNG visualizations)")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="nfstream-adversary: extract flows from PCAP and run classifier analysis."
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

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # analyze subcommand (legacy)
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run classifier evaluation on extracted flows (legacy)",
    )
    analyze_parser.add_argument("--scenario", default="scenario1-1")
    analyze_parser.add_argument("--channel", default=None)
    analyze_parser.add_argument("--subset", default=None)
    analyze_parser.add_argument("--classifier", default=None)
    analyze_parser.add_argument("--pooled-only", action="store_true")
    analyze_parser.add_argument("--no-importance", action="store_true")
    analyze_parser.add_argument("--max-samples", type=int, default=None)

    # experiment subcommand
    exp_parser = subparsers.add_parser(
        "experiment",
        help="Run experiment plan with visualizations",
    )
    exp_parser.add_argument(
        "--plan",
        default=None,
        help="Path to experiment JSON plan (default: experiments/default.json)",
    )
    exp_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    exp_parser.add_argument(
        "--scenario",
        default="scenario1-1",
        help="Scenario to analyze (default: scenario1-1)",
    )
    exp_parser.add_argument(
        "--channel",
        default=None,
        help="Single HCS channel to evaluate (default: all from plan)",
    )
    exp_parser.add_argument(
        "--subsets",
        default=None,
        help="Comma-separated feature subsets (overrides plan)",
    )
    exp_parser.add_argument(
        "--classifiers",
        default=None,
        help="Comma-separated classifiers (overrides plan)",
    )
    exp_parser.add_argument(
        "--pooled-only",
        action="store_true",
        help="Only run pooled HCS-vs-TGEN (skip per-channel)",
    )
    exp_parser.add_argument(
        "--no-importance",
        action="store_true",
        help="Skip feature importance computation",
    )
    exp_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max samples (default: 50000)",
    )
    exp_parser.add_argument(
        "--balanced",
        action="store_true",
        help="Balance classes by downsampling majority to match minority",
    )
    exp_parser.add_argument(
        "--tgen-types",
        default=None,
        help="Comma-separated TGEN types to evaluate (e.g., FTP,IRC)",
    )
    exp_parser.add_argument(
        "--paired",
        default=None,
        help="Comma-separated HCS:TGEN pairs (e.g., sky:MINIO,racetunnel:IRC). Use + for lists (e.g., sky+mastodon:MINIO+IRC)",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "experiment":
        cmd_experiment(args)
    else:
        cmd_extract(args)


if __name__ == "__main__":
    main()
