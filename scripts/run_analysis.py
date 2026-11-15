"""
run_analysis.py
===============
Analysis script: loads flight logs from both experiments, computes metrics,
generates all graphs, and prints a formatted comparison table.

Run from the project root:
    python scripts/run_analysis.py

Optional arguments:
    --log-dir   : Override default results directory
    --graphs-dir: Override default graphs directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader
from analysis.metrics import load_log, compute_metrics, format_comparison_table, metrics_from_csv
from analysis.plotter import generate_all_plots, plot_comparison


def main() -> None:
    """Run the complete flight-log analysis pipeline.

    Command-line options select alternate log and graph locations.  For each
    baseline or tuned experiment this function loads the CSV, computes metrics,
    writes JSON results, and generates diagnostic plots.  When both
    experiments are available it also writes a formatted comparison table and
    grouped comparison chart.
    """
    # Parse optional output overrides while retaining the repository defaults.
    parser = argparse.ArgumentParser(description="Flight log analysis and plotting")
    parser.add_argument('--log-dir',    default=None)
    parser.add_argument('--graphs-dir', default=None)
    args = parser.parse_args()

    # Load mission paths, controller labels, and settling criteria from the
    # shared configuration used by the rest of the project.
    cfg = ConfigLoader(_PROJECT_ROOT / "config" / "mission.yaml")

    # Resolve all output roots relative to the project unless the user supplied
    # a graph-directory override.
    results_root = _PROJECT_ROOT / "results"
    graphs_root  = Path(args.graphs_dir) if args.graphs_dir else (_PROJECT_ROOT / cfg.graphs_dir)
    metrics_root = _PROJECT_ROOT / cfg.results_dir

    # Ensure output directories exist before writing JSON, text, or PNG files.
    metrics_root.mkdir(parents=True, exist_ok=True)
    graphs_root.mkdir(parents=True, exist_ok=True)

    # Accumulate successful experiment results for the final comparison stage.
    all_metrics = []
    experiments = ['baseline', 'tuned']

    # Process each expected controller experiment independently so one missing
    # or invalid log does not prevent analysis of the other experiment.
    for exp in experiments:
        log_path = results_root / f"experiment_{exp}" / f"flight_log_{exp}.csv"
        # Skip absent logs with an actionable message rather than failing before
        # any available experiment can be analyzed.
        if not log_path.exists():
            print(f"[analysis] WARNING: Log not found for '{exp}': {log_path}")
            print(f"           Run 'python simulation/simulate_mission.py' first.\n")
            continue

        print(f"\n[analysis] Processing experiment: {exp}")
        print(f"           Log: {log_path}")

        try:
            # Validate and clean the CSV before any metrics or plots consume it.
            df = load_log(log_path)
            print(f"           Rows: {len(df)}")

            # Use configured controller gains to obtain the human-readable
            # experiment label stored in reports and plot legends.
            gains = cfg.controller_gains(exp)
            controller_name = gains.get('name', exp.capitalize())

            # Apply the configured settling criteria and retain the controller
            # name so later comparisons identify each metric series.
            m = compute_metrics(
                df,
                settling_tolerance_m=cfg.settling_tolerance_m,
                settling_duration_s=cfg.settling_duration_s,
                controller_name=controller_name,
            )
            all_metrics.append(m)

            # Save a portable per-experiment JSON report.
            m_path = metrics_root / f"metrics_{exp}.json"
            with open(m_path, 'w') as f:
                # JSON has no standard infinity representation; convert
                # infinity and NaN to strings while preserving normal values.
                safe_m = {
                    k: (str(v) if isinstance(v, float) and (v == float('inf') or v != v)
                        else v)
                    for k, v in m.items()
                }
                json.dump(safe_m, f, indent=2)
            print(f"           Metrics saved: {m_path.name}")

            # Produce all per-experiment tracking and error visualizations.
            exp_graphs = graphs_root / exp
            generate_all_plots(df, exp_graphs, label=controller_name, prefix=exp)

        except Exception as e:
            # Report the full traceback for diagnosis, then continue with the
            # next experiment so independent logs remain usable.
            print(f"[analysis] ERROR processing {exp}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compare only the experiments that completed successfully.
    if len(all_metrics) >= 2:
        print("\n" + "=" * 70)
        print("  CONTROLLER COMPARISON")
        print("=" * 70)
        print(format_comparison_table(all_metrics))
        print("=" * 70 + "\n")

        # Persist the same comparison shown in the console for later review.
        table_path = metrics_root / "comparison_table.txt"
        with open(table_path, 'w') as f:
            f.write("AISR Quadrotor Drone – Controller Comparison\n")
            f.write("=" * 70 + "\n")
            f.write(format_comparison_table(all_metrics) + "\n")
        print(f"[analysis] Comparison table saved: {table_path}")

        # Save a grouped chart for quick visual comparison of key metrics.
        chart_path = graphs_root / "comparison_chart.png"
        plot_comparison(all_metrics, chart_path)
        print(f"[analysis] Comparison chart saved: {chart_path}")

    elif len(all_metrics) == 1:
        print("\n[analysis] Only one experiment found – no comparison generated.")
        print("           Run the simulator for both 'baseline' and 'tuned' first.")

    else:
        print("\n[analysis] No experiment logs found. Run the simulator first:")
        print("           python simulation/simulate_mission.py\n")

    print("[analysis] Done.")


if __name__ == '__main__':
    main()
