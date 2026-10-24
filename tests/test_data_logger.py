"""
test_data_logger.py
===================
Unit tests for the DataLogger class.

Tests cover:
    - CSV file creation
    - Header row written correctly
    - log_step appends rows
    - Computed error columns (error_x, error_y, error_z, error_3d)
    - Context-manager interface
    - File closed on close()
    - Output file accessible after close
    - Multiple rows accumulate correctly
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.data_logger import DataLogger


def _default_step_kwargs(
    time_s=0.0, waypoint_id=1,
    desired_x=0.0, desired_y=5.0, desired_z=0.0,
    actual_x=0.0,  actual_y=4.5,  actual_z=0.0,
):
    return dict(
        time_s=time_s, waypoint_id=waypoint_id,
        desired_x=desired_x, desired_y=desired_y, desired_z=desired_z,
        actual_x=actual_x,   actual_y=actual_y,   actual_z=actual_z,
        roll=0.0, pitch=0.0, yaw=0.0,
        motor_fl=68.5, motor_fr=68.5, motor_rl=68.5, motor_rr=68.5,
    )


class TestDataLoggerCreation:
    def test_creates_csv_file(self, tmp_path):
        log = DataLogger(tmp_path / "test.csv")
        log.close()
        assert (tmp_path / "test.csv").exists()

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "log.csv"
        log = DataLogger(nested)
        log.close()
        assert nested.exists()

    def test_header_row_present(self, tmp_path):
        log = DataLogger(tmp_path / "h.csv")
        log.close()
        with open(tmp_path / "h.csv", newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
        assert 'time_s' in header
        assert 'actual_alt' in header


class TestDataLoggerLogging:
    def test_single_row_written(self, tmp_path):
        log = DataLogger(tmp_path / "r.csv")
        log.log_step(**_default_step_kwargs())
        log.close()
        with open(tmp_path / "r.csv", newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_multiple_rows(self, tmp_path):
        log = DataLogger(tmp_path / "m.csv")
        for i in range(10):
            log.log_step(**_default_step_kwargs(time_s=float(i) * 0.032))
        log.close()
        with open(tmp_path / "m.csv", newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10

    def test_row_count_property(self, tmp_path):
        log = DataLogger(tmp_path / "rc.csv")
        for i in range(7):
            log.log_step(**_default_step_kwargs(time_s=float(i)))
        assert log.row_count == 7
        log.close()

    def test_error_columns_computed_correctly(self, tmp_path):
        log = DataLogger(tmp_path / "e.csv")
        log.log_step(**_default_step_kwargs(
            desired_x=3.0, desired_y=5.0, desired_z=2.0,
            actual_x=1.0,  actual_y=4.0,  actual_z=0.0,
        ))
        log.close()
        with open(tmp_path / "e.csv", newline='') as f:
            row = list(csv.DictReader(f))[0]
        # error_x = desired_x - actual_x = 3 - 1 = 2
        assert float(row['error_x']) == pytest.approx(2.0, abs=1e-3)
        # error_y = desired_y - actual_y = 5 - 4 = 1
        assert float(row['error_y']) == pytest.approx(1.0, abs=1e-3)
        # error_z = desired_z - actual_z = 2 - 0 = 2
        assert float(row['error_z']) == pytest.approx(2.0, abs=1e-3)
        # error_3d = sqrt(4 + 1 + 4) = 3
        assert float(row['error_3d']) == pytest.approx(3.0, abs=1e-2)

    def test_zero_error_produces_zero_3d_error(self, tmp_path):
        log = DataLogger(tmp_path / "z.csv")
        log.log_step(**_default_step_kwargs(
            desired_x=0.0, desired_y=5.0, desired_z=0.0,
            actual_x=0.0,  actual_y=5.0,  actual_z=0.0,
        ))
        log.close()
        with open(tmp_path / "z.csv", newline='') as f:
            row = list(csv.DictReader(f))[0]
        assert float(row['error_3d']) == pytest.approx(0.0, abs=1e-6)


class TestDataLoggerContextManager:
    def test_context_manager_closes_file(self, tmp_path):
        path = tmp_path / "ctx.csv"
        with DataLogger(path) as log:
            log.log_step(**_default_step_kwargs())
        # File should exist and be readable
        assert path.exists()
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_file_closed_after_close(self, tmp_path):
        log = DataLogger(tmp_path / "cl.csv")
        log.close()
        assert log._file.closed

    def test_output_path_property(self, tmp_path):
        path = tmp_path / "op.csv"
        log = DataLogger(path)
        log.close()
        assert log.output_path == path
