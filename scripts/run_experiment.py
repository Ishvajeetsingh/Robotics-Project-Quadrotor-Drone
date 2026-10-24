"""
run_experiment.py
=================
Convenience wrapper script that:
    1. Runs the standalone simulator for one or both experiments
    2. Immediately runs the analysis (metrics + graphs)

Usage:
    python scripts/run_experiment.py                    # runs both experiments
    python scripts/run_experiment.py --experiment tuned # runs only 'tuned'
    python scripts/run_experiment.py --no-analysis      # skip analysis step

This script does NOT require Webots to be installed.
For real Webots execution, open simulation/mavic_survey.wbt in Webots.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYTHON       = sys.executable


def run(cmd: list, cwd: Path) -> bool:
    """Run a subprocess command and return True on success."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=str(cwd))
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drone experiments end-to-end")
    parser.add_argument(
        '--experiment',
        choices=['baseline', 'tuned', 'both'],
        default='both',
        help="Which experiment to run (default: both)",
    )
    parser.add_argument(
        '--no-analysis', action='store_true',
        help="Skip the analysis/plotting step",
    )
    args = parser.parse_args()

    sim_script = _PROJECT_ROOT / "simulation" / "simulate_mission.py"
    ana_script = _PROJECT_ROOT / "scripts"    / "run_analysis.py"

    # ---- Run simulator(s) ------------------------------------------------
    ok = run(
        [_PYTHON, str(sim_script), '--experiment', args.experiment],
        cwd=_PROJECT_ROOT,
    )
    if not ok:
        print("[run_experiment] Simulator exited with errors.")
        sys.exit(1)

    # ---- Run analysis ----------------------------------------------------
    if not args.no_analysis:
        ok = run([_PYTHON, str(ana_script)], cwd=_PROJECT_ROOT)
        if not ok:
            print("[run_experiment] Analysis exited with errors.")
            sys.exit(1)

    print("\n[run_experiment] All done.")
    print(f"  Logs    : {_PROJECT_ROOT / 'results'}")
    print(f"  Graphs  : {_PROJECT_ROOT / 'results' / 'graphs'}")
    print(f"  Metrics : {_PROJECT_ROOT / 'results' / 'metrics'}")


if __name__ == '__main__':
    main()
