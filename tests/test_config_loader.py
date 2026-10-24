"""
test_config_loader.py
=====================
Unit tests for the ConfigLoader class.

Tests cover:
    - Successful load from file
    - Required section validation
    - Waypoint count validation
    - arrival_radius_m validation
    - Accessor properties (timestep, altitude, waypoints, gains, paths)
    - controller_gains accessor for both variants
    - Invalid variant raises KeyError
    - Missing file raises FileNotFoundError
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_loader import ConfigLoader

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "mission.yaml"


class TestConfigLoaderLoad:
    def test_loads_mission_yaml(self):
        """ConfigLoader should load the project mission.yaml without errors."""
        cfg = ConfigLoader(_CONFIG_PATH)
        assert cfg is not None

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ConfigLoader(tmp_path / "nonexistent.yaml")

    def test_missing_section_raises(self, tmp_path):
        """A config missing a required section should raise ValueError."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("simulation:\n  timestep_ms: 32\n")
        with pytest.raises(ValueError, match="missing required section"):
            ConfigLoader(bad)

    def test_too_few_waypoints_raises(self, tmp_path):
        """Fewer than 2 waypoints should raise ValueError."""
        content = """
simulation:
  timestep_ms: 32
  max_mission_time_s: 300
  world_file: "x.wbt"
drone:
  start_position: [0.0, 0.0, 0.0]
  target_altitude_m: 5.0
  takeoff_speed_ms: 1.5
waypoints:
  arrival_radius_m: 0.8
  dwell_time_s: 2.0
  points:
    - id: 0
      label: Home
      x: 0.0
      z: 0.0
controller_baseline:
  name: B
  altitude: {kp: 1, ki: 0, kd: 0, integrator_limit: 5, output_limit: 100}
  roll:     {kp: 1, ki: 0, kd: 0, integrator_limit: 5, output_limit: 100}
  pitch:    {kp: 1, ki: 0, kd: 0, integrator_limit: 5, output_limit: 100}
  yaw:      {kp: 1, ki: 0, kd: 0, integrator_limit: 5, output_limit: 100}
  position: {kp: 1, ki: 0, kd: 0, integrator_limit: 5, output_limit: 0.4}
controller_tuned:
  name: T
  altitude: {kp: 1, ki: 0, kd: 0, integrator_limit: 5, output_limit: 100}
  roll:     {kp: 1, ki: 0, kd: 0, integrator_limit: 5, output_limit: 100}
  pitch:    {kp: 1, ki: 0, kd: 0, integrator_limit: 5, output_limit: 100}
  yaw:      {kp: 1, ki: 0, kd: 0, integrator_limit: 5, output_limit: 100}
  position: {kp: 1, ki: 0, kd: 0, integrator_limit: 5, output_limit: 0.4}
logging:
  csv_dir: results
  filename_prefix: flight_log
  fields: [time_s]
analysis:
  settling_tolerance_m: 0.3
  settling_duration_s: 1.5
  results_dir: results/metrics
  graphs_dir: results/graphs
"""
        bad = tmp_path / "fw.yaml"
        bad.write_text(content)
        with pytest.raises(ValueError):
            ConfigLoader(bad)


class TestConfigLoaderAccessors:
    @pytest.fixture(scope="class")
    def cfg(self):
        return ConfigLoader(_CONFIG_PATH)

    def test_timestep_ms(self, cfg):
        assert cfg.timestep_ms == 32

    def test_target_altitude_positive(self, cfg):
        assert cfg.target_altitude_m > 0

    def test_arrival_radius_positive(self, cfg):
        assert cfg.arrival_radius_m > 0

    def test_dwell_time_non_negative(self, cfg):
        assert cfg.dwell_time_s >= 0

    def test_waypoints_list_not_empty(self, cfg):
        assert len(cfg.waypoints) >= 2

    def test_waypoint_has_x_z(self, cfg):
        wp = cfg.waypoints[0]
        assert 'x' in wp and 'z' in wp

    def test_baseline_gains_keys(self, cfg):
        gains = cfg.controller_gains('baseline')
        for section in ('altitude', 'roll', 'pitch', 'yaw', 'position'):
            assert section in gains

    def test_tuned_gains_keys(self, cfg):
        gains = cfg.controller_gains('tuned')
        for section in ('altitude', 'roll', 'pitch', 'yaw', 'position'):
            assert section in gains

    def test_unknown_variant_raises(self, cfg):
        with pytest.raises(KeyError):
            cfg.controller_gains('unknown_variant')

    def test_log_fields_is_list(self, cfg):
        assert isinstance(cfg.log_fields, list)
        assert len(cfg.log_fields) > 0

    def test_settling_tolerance_positive(self, cfg):
        assert cfg.settling_tolerance_m > 0

    def test_settling_duration_positive(self, cfg):
        assert cfg.settling_duration_s > 0

    def test_max_mission_time_positive(self, cfg):
        assert cfg.max_mission_time_s > 0
