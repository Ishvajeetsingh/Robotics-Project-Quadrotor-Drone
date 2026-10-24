"""
config_loader.py
================
Loads and validates the mission YAML configuration.
Provides typed access to all simulation, controller, waypoint, logging,
and analysis parameters.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Optional YAML dependency – fall back to a pure-Python TOML-like parser when
# PyYAML is not available (unit-test environments without YAML installed).
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Minimal built-in YAML parser (subset sufficient for this config file)
# ---------------------------------------------------------------------------
def _parse_minimal_yaml(text: str) -> Dict[str, Any]:
    """
    Very small YAML parser that handles the specific structure in mission.yaml.
    Supports: scalars, nested mappings (indented), and inline lists.
    """
    import re

    def cast(value: str) -> Any:
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            inner = value[1:-1].split(',')
            return [cast(v) for v in inner]
        if value.lower() in ('true', 'yes'):
            return True
        if value.lower() in ('false', 'no'):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value.strip('"').strip("'")

    root: Dict[str, Any] = {}
    stack: List[tuple] = [(root, -1)]
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            i += 1
            continue
        indent = len(line) - len(stripped)
        # pop stack until indent matches
        while len(stack) > 1 and indent <= stack[-1][1]:
            stack.pop()
        parent = stack[-1][0]

        if stripped.startswith('- '):
            # sequence item
            content = stripped[2:].strip()
            if ':' in content:
                # mapping item
                item: Dict[str, Any] = {}
                # process first key on this line
                first_k, first_v = content.split(':', 1)
                if first_v.strip():
                    item[first_k.strip()] = cast(first_v)
                else:
                    item[first_k.strip()] = {}
                # look-ahead for subsequent keys at same indent+2
                j = i + 1
                sub_indent = indent + 4
                while j < len(lines):
                    sub_line = lines[j]
                    sub_stripped = sub_line.lstrip()
                    if not sub_stripped or sub_stripped.startswith('#'):
                        j += 1
                        continue
                    cur_indent = len(sub_line) - len(sub_stripped)
                    if cur_indent < sub_indent:
                        break
                    if ':' in sub_stripped:
                        k, v = sub_stripped.split(':', 1)
                        item[k.strip()] = cast(v.strip()) if v.strip() else {}
                    j += 1
                i = j
                if isinstance(parent, list):
                    parent.append(item)
                continue
            else:
                if isinstance(parent, list):
                    parent.append(cast(content))
                i += 1
                continue
        elif ':' in stripped:
            key, sep, val = stripped.partition(':')
            key = key.strip()
            val = val.strip()
            if val == '' or val.startswith('#'):
                # mapping value
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((child, indent))
            else:
                # remove inline comment
                val = val.split('#')[0].strip()
                if val == '':
                    child = {}
                    parent[key] = child
                    stack.append((child, indent))
                else:
                    parent[key] = cast(val)
        i += 1
    return root


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ConfigLoader:
    """Loads mission.yaml and exposes strongly-typed accessors."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            # Resolve relative to the project root (two levels up from this file)
            here = Path(__file__).resolve().parent
            config_path = here / "mission.yaml"
        self._path = Path(config_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")
        self._data: Dict[str, Any] = self._load()
        self._validate()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        text = self._path.read_text(encoding="utf-8")
        if _HAS_YAML:
            return yaml.safe_load(text)
        return _parse_minimal_yaml(text)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        required_keys = ['simulation', 'drone', 'waypoints', 'controller_baseline',
                         'controller_tuned', 'logging', 'analysis']
        for key in required_keys:
            if key not in self._data:
                raise ValueError(f"Config missing required section: '{key}'")
        # Validate waypoints
        wps = self._data['waypoints'].get('points', [])
        if len(wps) < 2:
            raise ValueError("At least 2 waypoints are required.")
        arr = self._data['waypoints'].get('arrival_radius_m', 0)
        if not isinstance(arr, (int, float)) or arr <= 0:
            raise ValueError("arrival_radius_m must be a positive number.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

    @property
    def timestep_ms(self) -> int:
        return int(self._data['simulation']['timestep_ms'])

    @property
    def max_mission_time_s(self) -> float:
        return float(self._data['simulation']['max_mission_time_s'])

    @property
    def target_altitude_m(self) -> float:
        return float(self._data['drone']['target_altitude_m'])

    @property
    def arrival_radius_m(self) -> float:
        return float(self._data['waypoints']['arrival_radius_m'])

    @property
    def dwell_time_s(self) -> float:
        return float(self._data['waypoints']['dwell_time_s'])

    @property
    def waypoints(self) -> List[Dict[str, Any]]:
        return self._data['waypoints']['points']

    def controller_gains(self, variant: str = 'baseline') -> Dict[str, Any]:
        """Return gains dict for 'baseline' or 'tuned'."""
        key = f'controller_{variant}'
        if key not in self._data:
            raise KeyError(f"Unknown controller variant: '{variant}'")
        return self._data[key]

    @property
    def log_dir(self) -> Path:
        return Path(self._data['logging']['csv_dir'])

    @property
    def log_fields(self) -> List[str]:
        return self._data['logging']['fields']

    @property
    def log_filename_prefix(self) -> str:
        return self._data['logging']['filename_prefix']

    @property
    def settling_tolerance_m(self) -> float:
        return float(self._data['analysis']['settling_tolerance_m'])

    @property
    def settling_duration_s(self) -> float:
        return float(self._data['analysis']['settling_duration_s'])

    @property
    def results_dir(self) -> Path:
        return Path(self._data['analysis']['results_dir'])

    @property
    def graphs_dir(self) -> Path:
        return Path(self._data['analysis']['graphs_dir'])
