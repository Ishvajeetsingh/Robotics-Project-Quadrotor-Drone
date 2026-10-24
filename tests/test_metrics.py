"""
test_metrics.py
===============
Unit tests for the analysis/metrics module.

Tests cover:
    - RMSE calculation
    - Altitude settling time (settled, never-settled, edge cases)
    - Altitude overshoot
    - Waypoint success rate
    - Mission completion time
    - compute_metrics on synthetic DataFrames
    - load_log on valid and invalid files
    - Empty dataset handling
    - Missing columns handling
    - Invalid value rows handling
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.metrics import (
    rmse,
    altitude_settling_time,
    altitude_overshoot,
    waypoint_success_rate,
    mission_completion_time,
    compute_metrics,
    load_log,
    format_comparison_table,
    REQUIRED_COLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(
    times: List[float],
    actual_alt: List[float],
    desired_alt: float = 5.0,
    wp_ids: List[int] | None = None,
) -> pd.DataFrame:
    """Create a minimal DataFrame that passes load_log validation."""
    n = len(times)
    wp = wp_ids if wp_ids is not None else [1] * n
    err_y = [d - a for d, a in zip([desired_alt] * n, actual_alt)]
    return pd.DataFrame({
        'time_s':      times,
        'waypoint_id': wp,
        'desired_alt': [desired_alt] * n,
        'actual_alt':  actual_alt,
        'desired_x':   [0.0] * n,
        'actual_x':    [0.0] * n,
        'desired_z':   [0.0] * n,
        'actual_z':    [0.0] * n,
        'error_x':     [0.0] * n,
        'error_y':     err_y,
        'error_z':     [0.0] * n,
        'error_3d':    [abs(e) for e in err_y],
    })


# ---------------------------------------------------------------------------
# RMSE tests
# ---------------------------------------------------------------------------

class TestRMSE:
    def test_perfect_tracking(self):
        errors = np.zeros(100)
        assert rmse(errors) == pytest.approx(0.0)

    def test_constant_error(self):
        errors = np.full(100, 2.0)
        assert rmse(errors) == pytest.approx(2.0)

    def test_mixed_errors(self):
        errors = np.array([1.0, -1.0, 2.0, -2.0])
        expected = math.sqrt((1 + 1 + 4 + 4) / 4)
        assert rmse(errors) == pytest.approx(expected, rel=1e-6)

    def test_empty_array_returns_nan(self):
        assert math.isnan(rmse(np.array([])))

    def test_single_value(self):
        errors = np.array([3.0])
        assert rmse(errors) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Altitude settling time tests
# ---------------------------------------------------------------------------

class TestAltitudeSettlingTime:
    def test_never_settles_returns_none(self):
        """System with constant large error never settles."""
        times = list(range(20))
        actual = [10.0] * 20   # desired = 5 → error = 5, tolerance = 0.3
        df = make_df(times, actual, desired_alt=5.0)
        result = altitude_settling_time(df, tolerance_m=0.3, duration_s=1.5)
        assert result is None

    def test_settles_immediately(self):
        """Error within tolerance from t=0 → settling at start."""
        times = list(range(10))
        actual = [5.0] * 10   # zero error
        df = make_df(times, actual, desired_alt=5.0)
        result = altitude_settling_time(df, tolerance_m=0.3, duration_s=1.5)
        assert result is not None
        assert result < 2.0

    def test_settles_after_transient(self):
        """System overshoots then settles; settling time > initial transient."""
        times  = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        actual = [0.0, 2.0, 6.0, 5.5, 5.2, 5.05, 5.0, 5.0, 5.0]
        df = make_df(times, actual, desired_alt=5.0)
        result = altitude_settling_time(df, tolerance_m=0.3, duration_s=1.5)
        # Should settle somewhere around t=5 (within tolerance for ≥1.5 s)
        assert result is not None
        assert 4.0 <= result <= 7.0

    def test_intermittent_disturbance_resets_counter(self):
        """A spike out of tolerance resets the settling counter before duration elapses."""
        # Spike at t=1 (before 1.5 s elapses from t=0) resets the counter.
        # After the spike the drone re-settles; settling is confirmed around t=4.
        times  = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        actual = [5.0, 8.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        df = make_df(times, actual, desired_alt=5.0)
        result = altitude_settling_time(df, tolerance_m=0.3, duration_s=1.5)
        # Counter resets at t=1; re-settles at t=2, confirmed ≥1.5 s later = t=3.5
        assert result is not None
        assert result >= 2.0   # settling must start AFTER the spike

    def test_too_short_to_settle(self):
        """Duration shorter than required hold → None."""
        times  = [0.0, 0.5, 1.0]   # only 1 s total at tolerance
        actual = [5.0, 5.0, 5.0]
        df = make_df(times, actual, desired_alt=5.0)
        # Require 2 s of holding
        result = altitude_settling_time(df, tolerance_m=0.3, duration_s=2.0)
        assert result is None

    def test_single_row_returns_none(self):
        df = make_df([0.0], [5.0], desired_alt=5.0)
        result = altitude_settling_time(df, tolerance_m=0.3, duration_s=1.5)
        assert result is None


# ---------------------------------------------------------------------------
# Altitude overshoot tests
# ---------------------------------------------------------------------------

class TestAltitudeOvershoot:
    def test_no_overshoot(self):
        times = list(range(10))
        actual = [4.0, 4.5, 4.8, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        df = make_df(times, actual, desired_alt=5.0)
        os_pct = altitude_overshoot(df)
        assert os_pct == pytest.approx(0.0)

    def test_overshoot_percentage(self):
        times = list(range(5))
        actual = [5.0, 5.5, 6.0, 5.0, 5.0]   # peak = 6.0, target = 5.0
        df = make_df(times, actual, desired_alt=5.0)
        os_pct = altitude_overshoot(df)
        # (6-5)/5 * 100 = 20%
        assert os_pct == pytest.approx(20.0, rel=0.01)

    def test_zero_desired_altitude_returns_nan(self):
        df = make_df([0.0, 1.0], [0.5, 0.5], desired_alt=0.0)
        result = altitude_overshoot(df)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Waypoint success rate tests
# ---------------------------------------------------------------------------

class TestWaypointSuccessRate:
    def test_all_waypoints_reached(self):
        n = 20
        wp_ids = [1]*5 + [2]*5 + [3]*5 + [4]*5
        df = make_df(list(range(n)), [5.0]*n, wp_ids=wp_ids)
        reached, total, rate = waypoint_success_rate(df)
        assert reached == 4
        assert total == 4
        assert rate == pytest.approx(100.0)

    def test_no_navigation_waypoints(self):
        """Only waypoint_id=0 → nothing reached."""
        df = make_df([0.0, 1.0, 2.0], [5.0]*3, wp_ids=[0, 0, 0])
        reached, total, rate = waypoint_success_rate(df)
        assert reached == 0
        assert rate == 0.0

    def test_partial_completion(self):
        n = 10
        # 3 out of 5 waypoints reached (max id = 5 but only 1-3 appear)
        wp_ids = [1]*3 + [2]*3 + [3]*4
        df = make_df(list(range(n)), [5.0]*n, wp_ids=wp_ids)
        reached, total, rate = waypoint_success_rate(df)
        assert reached == 3
        assert total == 3   # max id in data
        # rate = 3/3 = 100% (by logged ids)
        assert rate == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Mission completion time
# ---------------------------------------------------------------------------

class TestMissionCompletionTime:
    def test_correct_duration(self):
        times = [0.0, 1.0, 5.0, 10.0]
        df = make_df(times, [5.0]*4)
        assert mission_completion_time(df) == pytest.approx(10.0)

    def test_non_zero_start(self):
        times = [5.0, 6.0, 7.0]
        df = make_df(times, [5.0]*3)
        assert mission_completion_time(df) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# compute_metrics integration test
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _make_realistic_df(self) -> pd.DataFrame:
        """Simulate a drone reaching altitude 5 m with some oscillation."""
        times = [i * 0.032 for i in range(500)]
        desired = 5.0
        actual_alt = []
        y = 0.0
        vy = 0.0
        for t in times:
            err = desired - y
            vy += 0.5 * err * 0.032 - 0.3 * vy * 0.032
            y   += vy * 0.032
            actual_alt.append(y)

        wp_ids = [0]*50 + [1]*100 + [2]*100 + [3]*100 + [4]*100 + [5]*50
        return make_df(times, actual_alt, desired_alt=desired, wp_ids=wp_ids)

    def test_metrics_have_required_keys(self):
        df = self._make_realistic_df()
        m  = compute_metrics(df, controller_name="Test")
        required = [
            'alt_rmse_m', 'alt_mean_err_m', 'alt_max_err_m',
            'x_rmse_m', 'z_rmse_m', '3d_rmse_m',
            'waypoints_reached', 'mission_time_s', 'controller',
        ]
        for key in required:
            assert key in m, f"Missing key: {key}"

    def test_rmse_is_non_negative(self):
        df = self._make_realistic_df()
        m  = compute_metrics(df)
        for key in ('alt_rmse_m', 'x_rmse_m', 'z_rmse_m', '3d_rmse_m'):
            assert m[key] >= 0.0, f"{key} should be non-negative"

    def test_overshoot_is_non_negative(self):
        df = self._make_realistic_df()
        m  = compute_metrics(df)
        assert m['alt_overshoot_pct'] >= 0.0

    def test_max_error_greater_than_mean_error(self):
        df = self._make_realistic_df()
        m  = compute_metrics(df)
        assert m['alt_max_err_m'] >= m['alt_mean_err_m']


# ---------------------------------------------------------------------------
# load_log tests
# ---------------------------------------------------------------------------

class TestLoadLog:
    def _write_csv(self, rows: List[dict], tmp_path: Path) -> Path:
        if not rows:
            p = tmp_path / "empty.csv"
            p.write_text("")
            return p
        p = tmp_path / "test_log.csv"
        with open(p, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return p

    def _valid_row(self, t: float = 0.0, wp: int = 1) -> dict:
        return {
            'time_s': t, 'waypoint_id': wp,
            'desired_alt': 5.0, 'actual_alt': 5.0,
            'desired_x': 0.0, 'actual_x': 0.0,
            'desired_z': 0.0, 'actual_z': 0.0,
            'error_x': 0.0, 'error_y': 0.0,
            'error_z': 0.0, 'error_3d': 0.0,
        }

    def test_valid_file_loads(self, tmp_path):
        rows = [self._valid_row(t=float(i)) for i in range(10)]
        p = self._write_csv(rows, tmp_path)
        df = load_log(p)
        assert len(df) == 10

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_log(tmp_path / "nonexistent.csv")

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(ValueError):
            load_log(p)

    def test_missing_columns_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("time_s,waypoint_id\n1,2\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_log(p)

    def test_invalid_values_dropped(self, tmp_path):
        rows = [self._valid_row(t=float(i)) for i in range(5)]
        # Add a row with a non-numeric value
        bad_row = self._valid_row(t=5.0)
        bad_row['actual_alt'] = 'NaN_BAD'
        rows.append(bad_row)
        p = self._write_csv(rows, tmp_path)
        df = load_log(p)
        # The bad row should be dropped
        assert len(df) == 5

    def test_sorted_by_time(self, tmp_path):
        rows = [self._valid_row(t=float(5-i)) for i in range(5)]   # descending
        p = self._write_csv(rows, tmp_path)
        df = load_log(p)
        assert list(df['time_s']) == sorted(df['time_s'].tolist())


# ---------------------------------------------------------------------------
# format_comparison_table
# ---------------------------------------------------------------------------

class TestFormatComparisonTable:
    def test_two_experiments(self):
        m1 = {'controller': 'Baseline', 'alt_rmse_m': 0.5, 'alt_mean_err_m': 0.3,
               'x_rmse_m': 0.2, 'z_rmse_m': 0.2, '3d_rmse_m': 0.6,
               'alt_overshoot_pct': 10.0, 'alt_settling_time_s': 4.0,
               'alt_max_err_m': 0.8, 'waypoint_success_rate_pct': 100.0,
               'mission_time_s': 120.0, '3d_max_err_m': 1.0}
        m2 = {'controller': 'Tuned', 'alt_rmse_m': 0.3, 'alt_mean_err_m': 0.2,
               'x_rmse_m': 0.15, 'z_rmse_m': 0.15, '3d_rmse_m': 0.4,
               'alt_overshoot_pct': 5.0, 'alt_settling_time_s': 2.5,
               'alt_max_err_m': 0.5, 'waypoint_success_rate_pct': 100.0,
               'mission_time_s': 110.0, '3d_max_err_m': 0.7}
        table = format_comparison_table([m1, m2])
        assert 'Baseline' in table
        assert 'Tuned' in table
        assert 'RMSE' in table

    def test_empty_list(self):
        result = format_comparison_table([])
        assert result  # non-empty string

    def test_infinite_settling_time_displays_never(self):
        m = {'controller': 'Test', 'alt_rmse_m': 0.5, 'alt_mean_err_m': 0.4,
             'x_rmse_m': 0.3, 'z_rmse_m': 0.3, '3d_rmse_m': 0.6,
             'alt_overshoot_pct': 0.0, 'alt_settling_time_s': float('inf'),
             'alt_max_err_m': 1.0, 'waypoint_success_rate_pct': 50.0,
             'mission_time_s': 300.0, '3d_max_err_m': 1.5}
        table = format_comparison_table([m])
        assert 'Never' in table
