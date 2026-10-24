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
    parser = argparse.ArgumentParser(description="Flight log analysis and plotting")
    parser.add_argument('--log-dir',    default=None)
    parser.add_argument('--graphs-dir', default=None)
    args = parser.parse_args()

    cfg = ConfigLoader(_PROJECT_ROOT / "config" / "mission.yaml")

    results_root = _PROJECT_ROOT / "results"
    graphs_root  = Path(args.graphs_dir) if args.graphs_dir else (_PROJECT_ROOT / cfg.graphs_dir)
    metrics_root = _PROJECT_ROOT / cfg.results_dir

    metrics_root.mkdir(parents=True, exist_ok=True)
    graphs_root.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    experiments = ['baseline', 'tuned']

    for exp in experiments:
        log_path = results_root / f"experiment_{exp}" / f"flight_log_{exp}.csv"
        if not log_path.exists():
            print(f"[analysis] WARNING: Log not found for '{exp}': {log_path}")
            print(f"           Run 'python simulation/simulate_mission.py' first.\n")
            continue

        print(f"\n[analysis] Processing experiment: {exp}")
        print(f"           Log: {log_path}")

        try:
            df = load_log(log_path)
            print(f"           Rows: {len(df)}")

            gains = cfg.controller_gains(exp)
            controller_name = gains.get('name', exp.capitalize())

            m = compute_metrics(
                df,
                settling_tolerance_m=cfg.settling_tolerance_m,
                settling_duration_s=cfg.settling_duration_s,
                controller_name=controller_name,
            )
            all_metrics.append(m)

            # Save metrics JSON
            m_path = metrics_root / f"metrics_{exp}.json"
            with open(m_path, 'w') as f:
                # Convert inf to string for JSON serialisation
                safe_m = {
                    k: (str(v) if isinstance(v, float) and (v == float('inf') or v != v)
                        else v)
                    for k, v in m.items()
                }
                json.dump(safe_m, f, indent=2)
            print(f"           Metrics saved: {m_path.name}")

            # Generate graphs
            exp_graphs = graphs_root / exp
            generate_all_plots(df, exp_graphs, label=controller_name, prefix=exp)

        except Exception as e:
            print(f"[analysis] ERROR processing {exp}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ---- Comparison table and chart ----------------------------------------
    if len(all_metrics) >= 2:
        print("\n" + "=" * 70)
        print("  CONTROLLER COMPARISON")
        print("=" * 70)
        print(format_comparison_table(all_metrics))
        print("=" * 70 + "\n")

        # Save comparison table
        table_path = metrics_root / "comparison_table.txt"
        with open(table_path, 'w') as f:
            f.write("AISR Quadrotor Drone – Controller Comparison\n")
            f.write("=" * 70 + "\n")
            f.write(format_comparison_table(all_metrics) + "\n")
        print(f"[analysis] Comparison table saved: {table_path}")

        # Comparison bar chart
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
