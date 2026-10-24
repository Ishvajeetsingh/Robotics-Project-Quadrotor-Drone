"""
test_waypoint_manager.py
========================
Unit tests for the WaypointManager class.

Tests cover:
    - Takeoff phase transition
    - Waypoint arrival detection (within/outside radius)
    - Dwell timer logic
    - Mission completion sequence
    - Abort functionality
    - Invalid configuration rejection
    - Target coordinates during each phase
    - Waypoint progress counters
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from navigation.waypoint_manager import WaypointManager, MissionPhase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WAYPOINTS_SIMPLE = [
    {'id': 0, 'label': 'Home',  'x': 0.0, 'z': 0.0},
    {'id': 1, 'label': 'WP-1', 'x': 5.0, 'z': 0.0},
    {'id': 2, 'label': 'WP-2', 'x': 5.0, 'z': -5.0},
    {'id': 3, 'label': 'Home', 'x': 0.0, 'z': 0.0},
]

WAYPOINTS_SURVEY = [
    {'id': 0, 'label': 'Start', 'x':  0.0, 'z':  0.0},
    {'id': 1, 'label': 'A',     'x':  5.0, 'z':  0.0},
    {'id': 2, 'label': 'B',     'x':  5.0, 'z': -5.0},
    {'id': 3, 'label': 'C',     'x':  0.0, 'z': -5.0},
    {'id': 4, 'label': 'D',     'x': -5.0, 'z': -5.0},
    {'id': 5, 'label': 'E',     'x': -5.0, 'z':  0.0},
    {'id': 6, 'label': 'Home',  'x':  0.0, 'z':  0.0},
]


def make_manager(wps=None, alt=5.0, radius=0.8, dwell=2.0, takeoff_alt=4.5):
    wps = wps or WAYPOINTS_SIMPLE
    return WaypointManager(
        waypoints=wps,
        target_altitude_m=alt,
        arrival_radius_m=radius,
        dwell_time_s=dwell,
        takeoff_altitude_m=takeoff_alt,
    )


# ---------------------------------------------------------------------------
# Initialisation / validation
# ---------------------------------------------------------------------------

class TestWaypointManagerInit:
    def test_requires_at_least_two_waypoints(self):
        with pytest.raises(ValueError):
            WaypointManager(
                waypoints=[{'id': 0, 'label': 'Only', 'x': 0.0, 'z': 0.0}],
                target_altitude_m=5.0,
            )

    def test_rejects_non_positive_arrival_radius(self):
        with pytest.raises(ValueError):
            WaypointManager(
                waypoints=WAYPOINTS_SIMPLE,
                target_altitude_m=5.0,
                arrival_radius_m=0.0,
            )

    def test_rejects_negative_dwell(self):
        with pytest.raises(ValueError):
            WaypointManager(
                waypoints=WAYPOINTS_SIMPLE,
                target_altitude_m=5.0,
                arrival_radius_m=1.0,
                dwell_time_s=-1.0,
            )

    def test_initial_phase_is_takeoff(self):
        mgr = make_manager()
        assert mgr.phase == MissionPhase.TAKEOFF


# ---------------------------------------------------------------------------
# Takeoff phase
# ---------------------------------------------------------------------------

class TestTakeoffPhase:
    def test_stays_in_takeoff_below_altitude(self):
        mgr = make_manager(takeoff_alt=4.5)
        phase = mgr.update(0.0, 0.0, 2.0, 0.0)   # altitude 2 m < 4.5
        assert phase == MissionPhase.TAKEOFF

    def test_transitions_to_navigating_at_altitude(self):
        mgr = make_manager(takeoff_alt=4.5)
        phase = mgr.update(0.0, 0.0, 4.6, 0.0)   # altitude 4.6 ≥ 4.5
        assert phase == MissionPhase.NAVIGATING

    def test_target_y_always_returns_cruise_altitude(self):
        mgr = make_manager(alt=5.0)
        assert mgr.target_y == 5.0


# ---------------------------------------------------------------------------
# Navigation phase
# ---------------------------------------------------------------------------

class TestNavigationPhase:
    def _navigate_to_cruise(self, mgr, t=0.0):
        """Helper: quickly push manager to NAVIGATING state."""
        mgr.update(0.0, 0.0, 5.0, t)
        return mgr

    def test_outside_radius_stays_navigating(self):
        mgr = make_manager(radius=0.8)
        self._navigate_to_cruise(mgr)
        # WP-1 is at (5, 0); we're at (0, 0) → distance = 5
        phase = mgr.update(0.0, 0.0, 5.0, 0.1)
        assert phase == MissionPhase.NAVIGATING

    def test_inside_radius_transitions_to_dwelling(self):
        mgr = make_manager(radius=0.8, dwell=2.0)
        self._navigate_to_cruise(mgr)
        # Move to WP-1 position (5.0, 0.0) with 0 error
        phase = mgr.update(5.0, 0.0, 5.0, 0.5)
        assert phase == MissionPhase.DWELLING

    def test_arrival_counts_waypoint(self):
        mgr = make_manager(radius=0.8, dwell=0.0)  # no dwell → instant advance
        self._navigate_to_cruise(mgr)
        # Arrive at WP-1
        mgr.update(5.0, 0.0, 5.0, 0.5)
        assert mgr.waypoints_reached >= 1

    def test_target_coordinates_match_current_waypoint(self):
        mgr = make_manager(radius=0.8)
        self._navigate_to_cruise(mgr)
        # After takeoff, navigating to WP-1 at (5, 0)
        assert mgr.target_x == pytest.approx(5.0)
        assert mgr.target_z == pytest.approx(0.0)

    def test_advances_through_all_waypoints(self):
        """Full mission: navigate through all waypoints, reach COMPLETE."""
        mgr = make_manager(
            wps=WAYPOINTS_SIMPLE, radius=1.0, dwell=0.0, takeoff_alt=4.5
        )
        # Takeoff
        mgr.update(0.0, 0.0, 5.0, 0.0)
        assert mgr.phase == MissionPhase.NAVIGATING

        t = 1.0
        # Drive through all 3 navigation waypoints
        for wp in WAYPOINTS_SIMPLE[1:]:
            mgr.update(wp['x'], wp['z'], 5.0, t)
            t += 0.5

        # Mission should be complete
        assert mgr.phase == MissionPhase.COMPLETE


# ---------------------------------------------------------------------------
# Dwell phase
# ---------------------------------------------------------------------------

class TestDwellPhase:
    def _arrive_at_wp1(self, mgr, arrive_time=0.5):
        mgr.update(0.0, 0.0, 5.0, 0.0)         # takeoff
        mgr.update(5.0, 0.0, 5.0, arrive_time)  # arrive at WP-1
        return arrive_time

    def test_dwell_holds_for_configured_time(self):
        mgr = make_manager(radius=0.8, dwell=2.0)
        arrive_t = self._arrive_at_wp1(mgr)
        phase = mgr.update(5.0, 0.0, 5.0, arrive_t + 1.0)  # 1 s into dwell
        assert phase == MissionPhase.DWELLING

    def test_dwell_transitions_after_duration(self):
        mgr = make_manager(radius=0.8, dwell=2.0)
        arrive_t = self._arrive_at_wp1(mgr)
        phase = mgr.update(5.0, 0.0, 5.0, arrive_t + 2.5)  # 2.5 s > 2.0 s
        assert phase in (MissionPhase.NAVIGATING, MissionPhase.COMPLETE)

    def test_zero_dwell_skips_dwelling_phase(self):
        mgr = make_manager(radius=0.8, dwell=0.0)
        mgr.update(0.0, 0.0, 5.0, 0.0)         # takeoff
        phase = mgr.update(5.0, 0.0, 5.0, 0.5)  # arrive with zero dwell
        # Should skip straight to NAVIGATING (next WP) or COMPLETE
        assert phase in (MissionPhase.NAVIGATING, MissionPhase.COMPLETE)


# ---------------------------------------------------------------------------
# Mission completion
# ---------------------------------------------------------------------------

class TestMissionCompletion:
    def test_complete_phase_stays_complete(self):
        mgr = make_manager(wps=WAYPOINTS_SIMPLE, radius=1.0, dwell=0.0)
        mgr.update(0.0, 0.0, 5.0, 0.0)
        for wp in WAYPOINTS_SIMPLE[1:]:
            mgr.update(wp['x'], wp['z'], 5.0, 1.0)
        assert mgr.phase == MissionPhase.COMPLETE
        # Further updates should not change phase
        phase = mgr.update(0.0, 0.0, 5.0, 999.0)
        assert phase == MissionPhase.COMPLETE

    def test_total_waypoints_count(self):
        mgr = make_manager(wps=WAYPOINTS_SURVEY)
        # 7 waypoints – 1 home = 6 navigation targets
        assert mgr.total_waypoints == 6


# ---------------------------------------------------------------------------
# Abort
# ---------------------------------------------------------------------------

class TestAbort:
    def test_abort_sets_aborted_phase(self):
        mgr = make_manager()
        mgr.update(0.0, 0.0, 5.0, 0.0)
        mgr.abort()
        assert mgr.phase == MissionPhase.ABORTED

    def test_aborted_stays_aborted(self):
        mgr = make_manager()
        mgr.abort()
        phase = mgr.update(5.0, 0.0, 5.0, 1.0)
        assert phase == MissionPhase.ABORTED


# ---------------------------------------------------------------------------
# Horizontal arrival distance
# ---------------------------------------------------------------------------

class TestHorizontalDistance:
    def test_exact_position_zero_distance(self):
        from navigation.waypoint_manager import Waypoint
        wp = Waypoint(id=1, label='Test', x=3.0, z=4.0)
        assert wp.horizontal_distance_to(3.0, 4.0) == pytest.approx(0.0)

    def test_known_distance(self):
        from navigation.waypoint_manager import Waypoint
        wp = Waypoint(id=1, label='Test', x=0.0, z=0.0)
        assert wp.horizontal_distance_to(3.0, 4.0) == pytest.approx(5.0)
