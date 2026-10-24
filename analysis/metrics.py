"""
metrics.py
==========
Calculates performance metrics from a flight log CSV file.

Metrics computed:
    - Mean / Max / RMSE for X, Z (horizontal), Y (altitude), and 3-D error
    - Altitude settling time (time until error stays within tolerance)
    - Altitude overshoot
    - Waypoint success rate
    - Mission completion time
    - Per-waypoint statistics

All metrics are returned as a plain dict so they can be serialised to JSON or
formatted into a comparison table without further processing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CSV loading with robust error handling
# ---------------------------------------------------------------------------

REQUIRED_COLS = {
    'time_s', 'desired_alt', 'actual_alt',
    'desired_x', 'actual_x', 'desired_z', 'actual_z',
    'error_x', 'error_y', 'error_z', 'error_3d', 'waypoint_id',
}


def load_log(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a flight log CSV into a cleaned DataFrame.

    Raises
    ------
    FileNotFoundError : If the file does not exist.
    ValueError        : If required columns are missing or the file is empty.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Log file is empty: {path}")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Log file missing required columns: {missing}\n"
            f"  File: {path}"
        )

    # Coerce all numeric columns; NaN non-numeric rows will be dropped
    numeric_cols = list(REQUIRED_COLS)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    initial_rows = len(df)
    df = df.dropna(subset=numeric_cols)

    if df.empty:
        raise ValueError(f"Log file has no valid numeric rows after cleaning: {path}")

    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"[metrics] Warning: dropped {dropped} invalid rows from {path.name}")

    df = df.sort_values('time_s').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Individual metric calculators
# ---------------------------------------------------------------------------

def rmse(errors: np.ndarray) -> float:
    """Root Mean Square Error of an array of signed errors."""
    if len(errors) == 0:
        return float('nan')
    return float(np.sqrt(np.mean(errors ** 2)))


def altitude_settling_time(
    df: pd.DataFrame,
    tolerance_m: float,
    duration_s: float,
    desired_col: str = 'desired_alt',
    actual_col: str = 'actual_alt',
) -> Optional[float]:
    """
    Calculate altitude settling time.

    Settling is defined as: the first time T such that |error| ≤ tolerance
    continuously for ``duration_s`` seconds.

    Returns None if the system never settles.
    """
    alt_err = (df[actual_col] - df[desired_col]).abs().values
    times = df['time_s'].values

    if len(times) < 2:
        return None

    _UNSET = object()          # sentinel that is never equal to any float
    settle_start = _UNSET

    for t, err in zip(times, alt_err):
        if err <= tolerance_m:
            if settle_start is _UNSET:
                settle_start = t
            elif t - settle_start >= duration_s:  # type: ignore[operator]
                return float(settle_start)  # type: ignore[arg-type]
        else:
            settle_start = _UNSET   # left tolerance band – restart counter

    return None   # never settled


def altitude_overshoot(
    df: pd.DataFrame,
    desired_col: str = 'desired_alt',
    actual_col: str = 'actual_alt',
) -> float:
    """
    Maximum altitude overshoot as a percentage of the desired altitude.

    Only counts positive overshoot (altitude exceeding setpoint).
    """
    target = df[desired_col].iloc[-1]  # use steady-state setpoint
    if target == 0:
        return float('nan')
    peak = df[actual_col].max()
    overshoot = max(0.0, (peak - target) / target * 100.0)
    return float(overshoot)


def waypoint_success_rate(df: pd.DataFrame) -> Tuple[int, int, float]:
    """
    Return (reached, total, success_rate_pct) based on waypoint_id column.

    A waypoint is counted as reached if at least one row with that ID appears
    after the takeoff phase (waypoint_id > 0).
    """
    nav_df = df[df['waypoint_id'] > 0]
    if nav_df.empty:
        return 0, 0, 0.0
    unique_wps = nav_df['waypoint_id'].nunique()
    # Assume max waypoint ID = total waypoints
    total = int(df['waypoint_id'].max())
    rate = (unique_wps / total * 100.0) if total > 0 else 0.0
    return int(unique_wps), total, float(rate)


def mission_completion_time(df: pd.DataFrame) -> float:
    """Total elapsed simulation time (seconds)."""
    return float(df['time_s'].max() - df['time_s'].min())


# ---------------------------------------------------------------------------
# Master metrics function
# ---------------------------------------------------------------------------

def compute_metrics(
    df: pd.DataFrame,
    settling_tolerance_m: float = 0.3,
    settling_duration_s: float = 1.5,
    controller_name: str = "Unknown",
) -> Dict[str, Any]:
    """
    Compute all performance metrics from a loaded DataFrame.

    Parameters
    ----------
    df                   : Flight log DataFrame (from ``load_log``).
    settling_tolerance_m : Altitude settling tolerance (m).
    settling_duration_s  : Duration altitude must stay within tolerance (s).
    controller_name      : Label for this experiment's controller.

    Returns
    -------
    dict : Flat dictionary of metric name → value pairs.
    """
    err_x = df['error_x'].values
    err_z = df['error_z'].values
    err_y = df['error_y'].values   # altitude error
    err_3d = df['error_3d'].values

    # Settling time
    st = altitude_settling_time(
        df, settling_tolerance_m, settling_duration_s
    )
    settling_time = st if st is not None else float('inf')

    # Waypoint success
    wp_reached, wp_total, wp_rate = waypoint_success_rate(df)

    metrics = {
        'controller': controller_name,
        # Altitude
        'alt_rmse_m': rmse(err_y),
        'alt_mean_err_m': float(np.mean(np.abs(err_y))),
        'alt_max_err_m': float(np.max(np.abs(err_y))),
        'alt_overshoot_pct': altitude_overshoot(df),
        'alt_settling_time_s': settling_time,
        # X position
        'x_rmse_m': rmse(err_x),
        'x_mean_err_m': float(np.mean(np.abs(err_x))),
        'x_max_err_m': float(np.max(np.abs(err_x))),
        # Z position
        'z_rmse_m': rmse(err_z),
        'z_mean_err_m': float(np.mean(np.abs(err_z))),
        'z_max_err_m': float(np.max(np.abs(err_z))),
        # 3-D
        '3d_rmse_m': rmse(err_3d),
        '3d_mean_err_m': float(np.mean(err_3d)),
        '3d_max_err_m': float(np.max(err_3d)),
        # Mission
        'waypoints_reached': wp_reached,
        'waypoints_total': wp_total,
        'waypoint_success_rate_pct': wp_rate,
        'mission_time_s': mission_completion_time(df),
        'total_rows': len(df),
        'settling_tolerance_m': settling_tolerance_m,
        'settling_duration_s': settling_duration_s,
    }
    return metrics


def metrics_from_csv(
    csv_path: str | Path,
    settling_tolerance_m: float = 0.3,
    settling_duration_s: float = 1.5,
    controller_name: str = "Unknown",
) -> Dict[str, Any]:
    """
    Convenience function: load a CSV and compute all metrics in one call.
    """
    df = load_log(csv_path)
    return compute_metrics(
        df,
        settling_tolerance_m=settling_tolerance_m,
        settling_duration_s=settling_duration_s,
        controller_name=controller_name,
    )


def format_comparison_table(metrics_list: List[Dict[str, Any]]) -> str:
    """
    Return a human-readable comparison table of two or more experiments.
    """
    if not metrics_list:
        return "No metrics to compare."

    rows = [
        ("Metric", *[m['controller'] for m in metrics_list]),
        ("-" * 35, *["-" * 20] * len(metrics_list)),
        ("Altitude RMSE (m)",
         *[f"{m.get('alt_rmse_m', float('nan')):.4f}" for m in metrics_list]),
        ("Altitude Mean Error (m)",
         *[f"{m.get('alt_mean_err_m', float('nan')):.4f}" for m in metrics_list]),
        ("Altitude Max Error (m)",
         *[f"{m.get('alt_max_err_m', float('nan')):.4f}" for m in metrics_list]),
        ("Altitude Overshoot (%)",
         *[f"{m.get('alt_overshoot_pct', float('nan')):.2f}" for m in metrics_list]),
        ("Altitude Settling Time (s)",
         *[f"{m.get('alt_settling_time_s', float('inf')):.2f}"
           if m.get('alt_settling_time_s', float('inf')) != float('inf')
           else "Never" for m in metrics_list]),
        ("X RMSE (m)",
         *[f"{m.get('x_rmse_m', float('nan')):.4f}" for m in metrics_list]),
        ("Z RMSE (m)",
         *[f"{m.get('z_rmse_m', float('nan')):.4f}" for m in metrics_list]),
        ("3-D RMSE (m)",
         *[f"{m.get('3d_rmse_m', float('nan')):.4f}" for m in metrics_list]),
        ("3-D Max Error (m)",
         *[f"{m.get('3d_max_err_m', float('nan')):.4f}" for m in metrics_list]),
        ("WP Success Rate (%)",
         *[f"{m.get('waypoint_success_rate_pct', 0.0):.1f}" for m in metrics_list]),
        ("Mission Time (s)",
         *[f"{m.get('mission_time_s', 0.0):.1f}" for m in metrics_list]),
    ]

    col_w = [max(len(str(r[c])) for r in rows) + 2 for c in range(len(rows[0]))]
    lines = []
    for row in rows:
        lines.append("  ".join(str(row[c]).ljust(col_w[c]) for c in range(len(row))))
    return "\n".join(lines)
