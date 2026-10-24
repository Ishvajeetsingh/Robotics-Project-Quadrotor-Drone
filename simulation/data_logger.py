"""
data_logger.py
==============
Streams simulation state to a CSV file in real time.
Each call to ``log_step()`` appends one row; the file is opened on first write
and closed by calling ``close()`` or using the context-manager interface.

All column names come from the mission configuration so the logged fields can
be changed without editing this module.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class DataLogger:
    """
    Real-time CSV data logger for the drone flight simulation.

    Parameters
    ----------
    output_path : Full path to the CSV output file.
    fields      : List of column names (from config).
    """

    def __init__(
        self,
        output_path: str | Path,
        fields: Optional[List[str]] = None,
    ) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._fields: List[str] = fields or [
            'time_s', 'waypoint_id',
            'desired_x', 'desired_y', 'desired_z',
            'actual_x', 'actual_y', 'actual_z',
            'desired_alt', 'actual_alt',
            'roll', 'pitch', 'yaw',
            'motor_fl', 'motor_fr', 'motor_rl', 'motor_rr',
            'error_x', 'error_y', 'error_z', 'error_3d',
        ]

        self._file = open(self._path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fields,
            extrasaction='ignore',
        )
        self._writer.writeheader()
        self._file.flush()
        self._row_count: int = 0

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "DataLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        time_s: float,
        waypoint_id: int,
        desired_x: float,
        desired_y: float,
        desired_z: float,
        actual_x: float,
        actual_y: float,
        actual_z: float,
        roll: float,
        pitch: float,
        yaw: float,
        motor_fl: float,
        motor_fr: float,
        motor_rl: float,
        motor_rr: float,
    ) -> None:
        """Append one timestep row to the CSV file."""
        err_x = desired_x - actual_x
        err_y = desired_y - actual_y
        err_z = desired_z - actual_z
        err_3d = math.sqrt(err_x ** 2 + err_y ** 2 + err_z ** 2)

        row: Dict[str, Any] = {
            'time_s': f"{time_s:.4f}",
            'waypoint_id': waypoint_id,
            'desired_x': f"{desired_x:.4f}",
            'desired_y': f"{desired_y:.4f}",
            'desired_z': f"{desired_z:.4f}",
            'actual_x': f"{actual_x:.4f}",
            'actual_y': f"{actual_y:.4f}",
            'actual_z': f"{actual_z:.4f}",
            'desired_alt': f"{desired_y:.4f}",
            'actual_alt': f"{actual_y:.4f}",
            'roll': f"{roll:.6f}",
            'pitch': f"{pitch:.6f}",
            'yaw': f"{yaw:.6f}",
            'motor_fl': f"{motor_fl:.4f}",
            'motor_fr': f"{motor_fr:.4f}",
            'motor_rl': f"{motor_rl:.4f}",
            'motor_rr': f"{motor_rr:.4f}",
            'error_x': f"{err_x:.4f}",
            'error_y': f"{err_y:.4f}",
            'error_z': f"{err_z:.4f}",
            'error_3d': f"{err_3d:.4f}",
        }
        self._writer.writerow(row)
        self._row_count += 1

        # Flush periodically (every 50 rows) to avoid data loss on crash
        if self._row_count % 50 == 0:
            self._file.flush()

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the log file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def output_path(self) -> Path:
        return self._path
