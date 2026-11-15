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
    Load and validate a flight log CSV for metric calculation.

    The CSV is read, checked for required columns, converted to numeric
    values, cleaned of invalid rows, and sorted chronologically.  The returned
    frame is the normalized input consumed by the metric calculators.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the flight log CSV.

    Returns
    -------
    pandas.DataFrame
        A non-empty, chronologically sorted flight log with valid numeric
        values in every required column.

    Raises
    ------
    FileNotFoundError : If the file does not exist.
    ValueError        : If required columns are missing or the file is empty.
    """
    # Normalize string and Path inputs for consistent filesystem operations.
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    # Read the complete mission history because metrics use all samples.
    df = pd.read_csv(path)

    # An empty file cannot provide tracking errors or mission timing.
    if df.empty:
        raise ValueError(f"Log file is empty: {path}")

    # Validate the schema before selecting columns so malformed logs fail with
    # a clear explanation of which analysis inputs are unavailable.
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Log file missing required columns: {missing}\n"
            f"  File: {path}"
        )

    # Convert required fields to numbers; malformed values become NaN and are
    # removed below as invalid samples.
    numeric_cols = list(REQUIRED_COLS)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    initial_rows = len(df)
    # Keep only rows that contain usable values for every metric input.
    df = df.dropna(subset=numeric_cols)

    if df.empty:
        raise ValueError(f"Log file has no valid numeric rows after cleaning: {path}")

    # Report partial data loss while allowing the remaining valid mission data
    # to continue through the analysis pipeline.
    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"[metrics] Warning: dropped {dropped} invalid rows from {path.name}")

    # Chronological ordering is required by settling-time traversal and keeps
    # plots and elapsed-time calculations aligned with mission time.
    df = df.sort_values('time_s').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Individual metric calculators
# ---------------------------------------------------------------------------

def rmse(errors: np.ndarray) -> float:
    """
    Return the root mean square error of signed tracking errors.

    RMSE is ``sqrt(mean(error**2))``: squaring removes sign, averaging
    summarizes all samples, and the square root restores the error units.

    Parameters
    ----------
    errors : numpy.ndarray
        Signed error samples for an axis or the combined 3-D position.

    Returns
    -------
    float
        RMSE in the input units, or NaN for an empty array.
    """
    # No samples means there is no meaningful mean error to calculate.
    if len(errors) == 0:
        return float('nan')
    # Apply the RMSE formula to penalize larger deviations more strongly.
    return float(np.sqrt(np.mean(errors ** 2)))


def altitude_settling_time(
    df: pd.DataFrame,
    tolerance_m: float,
    duration_s: float,
    desired_col: str = 'desired_alt',
    actual_col: str = 'actual_alt',
) -> Optional[float]:
    """
    Calculate the first altitude settling time.

    Settling is defined as: the first time T such that |error| ≤ tolerance
    continuously for ``duration_s`` seconds.

    Parameters
    ----------
    df : pandas.DataFrame
        Chronologically ordered flight log containing altitude columns.
    tolerance_m : float
        Maximum absolute altitude error allowed inside the settling band.
    duration_s : float
        Required continuous time spent inside the band.
    desired_col, actual_col : str
        Columns containing desired and measured altitude.

    Returns
    -------
    float or None
        The beginning of the qualifying in-band interval, or ``None`` if the
        log is too short or never remains settled.
    """
    # Absolute error measures distance from the setpoint; timestamps allow
    # duration to be measured in seconds rather than sample count.
    alt_err = (df[actual_col] - df[desired_col]).abs().values
    times = df['time_s'].values

    # One timestamp cannot prove a non-zero continuous settling interval.
    if len(times) < 2:
        return None

    # A unique marker prevents a numeric timestamp from being mistaken for the
    # "not currently settling" state.
    _UNSET = object()          # sentinel that is never equal to any float
    settle_start = _UNSET

    # Track each continuous in-band run; an excursion restarts the interval.
    for t, err in zip(times, alt_err):
        if err <= tolerance_m:
            if settle_start is _UNSET:
                # This sample begins a candidate settling interval.
                settle_start = t
            elif t - settle_start >= duration_s:  # type: ignore[operator]
                # The candidate has remained within tolerance long enough.
                return float(settle_start)  # type: ignore[arg-type]
        else:
            # Leaving the band invalidates the candidate and restarts timing.
            settle_start = _UNSET   # left tolerance band – restart counter

    # No continuous in-band interval satisfied the required duration.
    return None   # never settled


def altitude_overshoot(
    df: pd.DataFrame,
    desired_col: str = 'desired_alt',
    actual_col: str = 'actual_alt',
) -> float:
    """
    Calculate maximum positive altitude overshoot as a percentage.

    The formula is ``max(0, (peak - target) / target * 100)``.  It compares
    the highest measured altitude with the final desired setpoint and ignores
    undershoot because this metric measures exceeding the target.

    Parameters
    ----------
    df : pandas.DataFrame
        Flight log containing desired and actual altitude columns.
    desired_col, actual_col : str
        Columns used to identify the target and measured peak.

    Returns
    -------
    float
        Overshoot percentage, or NaN when the target altitude is zero.
    """
    # Use the final desired value as the steady-state target and percentage
    # denominator.
    target = df[desired_col].iloc[-1]  # use steady-state setpoint
    if target == 0:
        return float('nan')
    # Find the highest observed altitude and clamp undershoot to zero percent.
    peak = df[actual_col].max()
    overshoot = max(0.0, (peak - target) / target * 100.0)
    return float(overshoot)


def waypoint_success_rate(df: pd.DataFrame) -> Tuple[int, int, float]:
    """
    Calculate waypoint completion as ``(reached / total) * 100``.

    A waypoint is counted as reached if at least one row with that ID appears
    after the takeoff phase (waypoint_id > 0).

    Parameters
    ----------
    df : pandas.DataFrame
        Flight log containing the ``waypoint_id`` column.

    Returns
    -------
    tuple[int, int, float]
        Distinct positive IDs reached, maximum logged ID as total, and the
        resulting percentage.
    """
    # Exclude takeoff samples, whose zero ID is not a navigation waypoint.
    nav_df = df[df['waypoint_id'] > 0]
    if nav_df.empty:
        return 0, 0, 0.0
    # Distinct positive IDs represent waypoints reached at least once.
    unique_wps = nav_df['waypoint_id'].nunique()
    # Assume max waypoint ID = total waypoints
    total = int(df['waypoint_id'].max())
    # Avoid division by zero for logs without a valid positive total.
    rate = (unique_wps / total * 100.0) if total > 0 else 0.0
    return int(unique_wps), total, float(rate)


def mission_completion_time(df: pd.DataFrame) -> float:
    """
    Return total elapsed simulation time in seconds.

    Parameters
    ----------
    df : pandas.DataFrame
        Flight log containing the ``time_s`` column.

    Returns
    -------
    float
        Latest timestamp minus earliest timestamp.
    """
    # Endpoint difference gives duration independently of sample interval.
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
    dict : Flat dictionary of metric name → value pairs, including tracking,
        mission, waypoint, and analysis-configuration values.
    """
    # Extract each signed error series once so all derived metrics use the same
    # cleaned samples from the input DataFrame.
    err_x = df['error_x'].values
    err_z = df['error_z'].values
    err_y = df['error_y'].values   # altitude error
    err_3d = df['error_3d'].values

    # Convert an unsuccessful settling search to infinity so the result stays
    # numeric and can be compared or serialized with the other metrics.
    st = altitude_settling_time(
        df, settling_tolerance_m, settling_duration_s
    )
    settling_time = st if st is not None else float('inf')

    # Count reached IDs and calculate the logged waypoint completion rate.
    wp_reached, wp_total, wp_rate = waypoint_success_rate(df)

    # Assemble stable field names consumed by JSON output, tables, plots, and
    # tests.  Each value describes a different controller-performance aspect.
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
    Load a flight-log CSV and compute its complete metric dictionary.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Flight-log file to validate and analyze.
    settling_tolerance_m, settling_duration_s : float
        Altitude settling criteria passed to :func:`compute_metrics`.
    controller_name : str
        Label stored with the resulting metrics.

    Returns
    -------
    dict
        The flat metric dictionary returned by :func:`compute_metrics`.
    """
    # Reuse the canonical loader so direct CSV callers get the same validation
    # and cleaning behavior as the command-line analysis pipeline.
    df = load_log(csv_path)
    return compute_metrics(
        df,
        settling_tolerance_m=settling_tolerance_m,
        settling_duration_s=settling_duration_s,
        controller_name=controller_name,
    )


def format_comparison_table(metrics_list: List[Dict[str, Any]]) -> str:
    """
    Format experiment metrics as aligned, human-readable text columns.

    Parameters
    ----------
    metrics_list : list[dict]
        Dictionaries produced by :func:`compute_metrics`, each containing a
        controller label and values for the displayed comparison fields.

    Returns
    -------
    str
        A formatted comparison table, or a status message when empty.
    """
    # Give callers useful output instead of indexing an empty metrics list.
    if not metrics_list:
        return "No metrics to compare."

    # Use controller labels as headings and provide a positional fallback for
    # incomplete dictionaries.
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

    # Measure every cell before padding so all values align in their columns.
    col_w = [max(len(str(r[c])) for r in rows) + 2 for c in range(len(rows[0]))]
    lines = []
    # Pad and join each row into the final printable table.
    for row in rows:
        lines.append("  ".join(str(row[c]).ljust(col_w[c]) for c in range(len(row))))
    return "\n".join(lines)
