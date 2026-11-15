"""
plotter.py
==========
Generates all required performance graphs from flight log DataFrames.

Graphs produced:
    1. Desired vs actual altitude over time
    2. Desired vs actual X position over time
    3. Desired vs actual Z position over time
    4. 3-D trajectory (desired vs actual)
    5. Position error (X, Z, 3-D) over time
    6. Altitude error over time
    7. Controller comparison bar chart (two experiments)

All graphs are saved as PNG files with 150 dpi.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')   # non-interactive backend – safe in all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Consistent colour palette
C_DESIRED = '#2E86AB'
C_ACTUAL  = '#E84855'
C_ERR_X   = '#F4A261'
C_ERR_Z   = '#2A9D8F'
C_ERR_3D  = '#8338EC'

FIGSIZE_WIDE   = (12, 5)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_TALL   = (12, 8)
DPI            = 150


def _save(fig: plt.Figure, path: Path, title: str = "") -> None:
    """Finalize, create the destination directory, save, and close a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure assembled by one of the plotting functions.
    path : pathlib.Path
        PNG destination for the rendered figure.
    title : str
        Optional figure-level title added before layout calculation.
    """
    # Add an optional figure-level heading before tightening layout so it is
    # included without overlapping axes content.
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    # Tight layout improves readability, while mkdir makes nested experiment
    # output paths usable on a fresh checkout.
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use the project-wide resolution and crop excess whitespace in the PNG.
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    # Release the figure so batch analysis does not accumulate open figures.
    plt.close(fig)
    print(f"  [plotter] Saved: {path.name}")


# ---------------------------------------------------------------------------
# 1. Altitude plot
# ---------------------------------------------------------------------------

def plot_altitude(df: pd.DataFrame, out_path: Path, label: str = "") -> None:
    """Plot desired and actual altitude to evaluate vertical tracking.

    Parameters
    ----------
    df : pandas.DataFrame
        Flight log containing time, desired altitude, and actual altitude.
    out_path : pathlib.Path
        PNG path passed to the shared save helper.
    label : str
        Optional controller name shown in the plot title.
    """
    # The dashed setpoint and measured trace reveal response, overshoot, and
    # steady-state altitude error across the mission.
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(df['time_s'], df['desired_alt'], color=C_DESIRED,
            lw=2, ls='--', label='Desired altitude')
    ax.plot(df['time_s'], df['actual_alt'],  color=C_ACTUAL,
            lw=1.5, alpha=0.9, label='Actual altitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Altitude Tracking{" – " + label if label else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. X-position plot
# ---------------------------------------------------------------------------

def plot_x_position(df: pd.DataFrame, out_path: Path, label: str = "") -> None:
    """Plot desired versus actual X position over time.

    Parameters
    ----------
    df : pandas.DataFrame
        Flight log containing time and X setpoint/measurement columns.
    out_path : pathlib.Path
        PNG destination.
    label : str
        Optional controller label for the title.
    """
    # Comparing the horizontal X traces shows lateral tracking accuracy and
    # transient deviations along that mission axis.
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(df['time_s'], df['desired_x'], color=C_DESIRED,
            lw=2, ls='--', label='Desired X')
    ax.plot(df['time_s'], df['actual_x'],  color=C_ACTUAL,
            lw=1.5, alpha=0.9, label='Actual X')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Position (m)')
    ax.set_title(f'X-Position Tracking{" – " + label if label else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Z-position plot
# ---------------------------------------------------------------------------

def plot_z_position(df: pd.DataFrame, out_path: Path, label: str = "") -> None:
    """Plot desired versus actual Z position over time.

    Parameters
    ----------
    df : pandas.DataFrame
        Flight log containing time and Z setpoint/measurement columns.
    out_path : pathlib.Path
        PNG destination.
    label : str
        Optional controller label for the title.
    """
    # The separation between traces identifies horizontal Z tracking error and
    # helps distinguish axis-specific controller behavior.
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(df['time_s'], df['desired_z'], color=C_DESIRED,
            lw=2, ls='--', label='Desired Z')
    ax.plot(df['time_s'], df['actual_z'],  color=C_ACTUAL,
            lw=1.5, alpha=0.9, label='Actual Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z Position (m)')
    ax.set_title(f'Z-Position Tracking{" – " + label if label else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. 3-D trajectory
# ---------------------------------------------------------------------------

def plot_3d_trajectory(df: pd.DataFrame, out_path: Path, label: str = "") -> None:
    """Plot desired and actual three-dimensional flight trajectories.

    Parameters
    ----------
    df : pandas.DataFrame
        Flight log containing X, Z, altitude, and waypoint columns.
    out_path : pathlib.Path
        PNG destination.
    label : str
        Optional controller label for the title.
    """
    # A 3-D comparison shows the accumulated spatial path error rather than
    # viewing each position axis independently.
    fig = plt.figure(figsize=FIGSIZE_SQUARE)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(df['desired_x'], df['desired_z'], df['desired_alt'],
            color=C_DESIRED, lw=2, ls='--', label='Desired path')
    ax.plot(df['actual_x'], df['actual_z'], df['actual_alt'],
            color=C_ACTUAL, lw=1.5, alpha=0.85, label='Actual path')

    # Mark samples where the logged waypoint ID changes so mission targets are
    # visible on the desired path.
    wp_changes = df[df['waypoint_id'].diff() != 0]
    ax.scatter(
        wp_changes['desired_x'], wp_changes['desired_z'], wp_changes['desired_alt'],
        color='gold', s=60, zorder=5, label='Waypoints'
    )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(f'3-D Trajectory{" – " + label if label else ""}')
    ax.legend()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Position error over time
# ---------------------------------------------------------------------------

def plot_position_errors(df: pd.DataFrame, out_path: Path, label: str = "") -> None:
    """Plot absolute X/Z errors and combined 3-D error over time.

    Parameters
    ----------
    df : pandas.DataFrame
        Flight log containing axis and 3-D error columns.
    out_path : pathlib.Path
        PNG destination.
    label : str
        Optional controller label for the title.
    """
    # Absolute axis errors make magnitude comparable, while error_3d shows
    # total spatial deviation used by the mission-level evaluation.
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(df['time_s'], df['error_x'].abs(), color=C_ERR_X,
            lw=1.5, alpha=0.85, label='|X error|')
    ax.plot(df['time_s'], df['error_z'].abs(), color=C_ERR_Z,
            lw=1.5, alpha=0.85, label='|Z error|')
    ax.plot(df['time_s'], df['error_3d'],       color=C_ERR_3D,
            lw=2, alpha=0.9, label='3-D error')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title(f'Position Error Over Time{" – " + label if label else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Altitude error over time
# ---------------------------------------------------------------------------

def plot_altitude_error(df: pd.DataFrame, out_path: Path, label: str = "") -> None:
    """Plot altitude tracking and absolute altitude error in two shared panels.

    Parameters
    ----------
    df : pandas.DataFrame
        Flight log containing altitude setpoint and measurement columns.
    out_path : pathlib.Path
        PNG destination.
    label : str
        Optional controller label for the title.
    """
    # Compute the magnitude of vertical error for the lower diagnostic panel.
    alt_err = (df['actual_alt'] - df['desired_alt']).abs()
    # Sharing the time axis directly relates altitude response to its error.
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_TALL, sharex=True)

    axes[0].plot(df['time_s'], df['actual_alt'], color=C_ACTUAL,
                 lw=1.5, label='Actual altitude')
    axes[0].plot(df['time_s'], df['desired_alt'], color=C_DESIRED,
                 lw=2, ls='--', label='Desired altitude')
    axes[0].set_ylabel('Altitude (m)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Altitude & Error{" – " + label if label else ""}')

    # The filled error trace emphasizes the size and persistence of vertical
    # deviations used when judging settling performance.
    axes[1].fill_between(df['time_s'], alt_err, alpha=0.4, color=C_ACTUAL)
    axes[1].plot(df['time_s'], alt_err, color=C_ACTUAL, lw=1.5, label='|Alt error|')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('|Error| (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 7. Controller comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison(
    metrics_list: List[Dict],
    out_path: Path,
) -> None:
    """
    Plot selected controller metrics as grouped bars.

    Parameters
    ----------
    metrics_list : list[dict]
        Metric dictionaries with controller names and comparison values.
    out_path : pathlib.Path
        PNG destination.

    The grouped chart provides a direct baseline-versus-tuned view of RMSE,
    mean altitude error, and altitude overshoot; lower bars generally indicate
    smaller tracking deviations.
    """
    # Nothing can be plotted when no experiment produced metrics.
    if not metrics_list:
        return

    # Controller labels become legend entries; positional names keep the chart
    # usable even when a caller omits a label.
    names = [m.get('controller', f'Exp {i}') for i, m in enumerate(metrics_list)]

    # These keys are the common performance measures shown side by side.
    metric_keys = [
        ('alt_rmse_m',          'Altitude RMSE (m)'),
        ('alt_mean_err_m',      'Mean Alt Error (m)'),
        ('x_rmse_m',            'X RMSE (m)'),
        ('z_rmse_m',            'Z RMSE (m)'),
        ('3d_rmse_m',           '3-D RMSE (m)'),
        ('alt_overshoot_pct',   'Alt Overshoot (%)'),
    ]

    # Allocate equal category positions and divide each category into grouped
    # bars for the supplied experiments.
    x_pos = np.arange(len(metric_keys))
    width = 0.8 / len(metrics_list)
    colors = ['#2E86AB', '#E84855', '#F4A261', '#2A9D8F']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw one series per controller so the same metric can be compared across
    # experiments at one x-axis category.
    for i, (m, name) in enumerate(zip(metrics_list, names)):
        vals = [m.get(k, float('nan')) for k, _ in metric_keys]
        offset = (i - len(metrics_list) / 2 + 0.5) * width
        bars = ax.bar(x_pos + offset, vals, width, label=name,
                      color=colors[i % len(colors)], alpha=0.85)
        # Add readable values while skipping NaN and infinity placeholders.
        for bar, v in zip(bars, vals):
            if not math.isnan(v) and not math.isinf(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8,
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([label for _, label in metric_keys], rotation=20, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Controller Performance Comparison')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Convenience: generate ALL plots for one experiment
# ---------------------------------------------------------------------------

def generate_all_plots(
    df: pd.DataFrame,
    graphs_dir: Path,
    label: str = "",
    prefix: str = "",
) -> None:
    """Generate and save every diagnostic graph for one experiment.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned flight log shared by all individual plot functions.
    graphs_dir : pathlib.Path
        Directory where the experiment's PNG files are written.
    label : str
        Controller name included in plot titles.
    prefix : str
        Optional filename prefix used to distinguish experiment outputs.
    """
    # Keep filenames grouped by experiment while allowing callers to omit a
    # prefix when they already provide a dedicated directory.
    pfx = f"{prefix}_" if prefix else ""
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Generate altitude, axis-position, 3-D path, and error diagnostics using
    # the same cleaned DataFrame so all views describe one experiment.
    plot_altitude(df,         graphs_dir / f"{pfx}altitude.png",         label)
    plot_x_position(df,       graphs_dir / f"{pfx}x_position.png",       label)
    plot_z_position(df,       graphs_dir / f"{pfx}z_position.png",       label)
    plot_3d_trajectory(df,    graphs_dir / f"{pfx}trajectory_3d.png",    label)
    plot_position_errors(df,  graphs_dir / f"{pfx}position_errors.png",  label)
    plot_altitude_error(df,   graphs_dir / f"{pfx}altitude_error.png",   label)
    print(f"[plotter] All plots saved for '{label}' in {graphs_dir}")
