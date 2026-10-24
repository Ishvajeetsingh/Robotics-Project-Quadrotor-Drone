"""
waypoint_manager.py
===================
Manages the ordered survey waypoint sequence, detects arrival, handles dwell
timers, and tracks mission state.

Coordinate convention:
    Positions are in the Webots world frame: X = East, Y = Up, Z = South.
    The waypoint manager only checks horizontal distance (XZ-plane) for
    arrival detection; altitude is controlled separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


class MissionPhase(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    NAVIGATING = auto()
    DWELLING = auto()
    COMPLETE = auto()
    ABORTED = auto()


@dataclass
class Waypoint:
    """A single survey waypoint in the XZ horizontal plane."""
    id: int
    label: str
    x: float   # East offset (m)
    z: float   # South offset (m)

    def horizontal_distance_to(self, x: float, z: float) -> float:
        """Euclidean distance in the horizontal plane only."""
        return math.sqrt((self.x - x) ** 2 + (self.z - z) ** 2)


@dataclass
class MissionStatus:
    """Snapshot of mission progress at each step."""
    phase: MissionPhase
    current_wp_index: int
    current_wp: Optional[Waypoint]
    next_wp: Optional[Waypoint]
    waypoints_reached: int
    waypoints_total: int
    elapsed_s: float
    dwell_remaining_s: float = 0.0


class WaypointManager:
    """
    Manages the ordered waypoint list for a survey mission.

    Usage
    -----
    1. Instantiate with waypoint list and configuration.
    2. Call ``update(drone_x, drone_z, drone_y, current_time_s)`` every step.
    3. Read ``target_x``, ``target_z``, ``phase``, and ``status`` properties.
    """

    def __init__(
        self,
        waypoints: List[dict],
        target_altitude_m: float,
        arrival_radius_m: float = 0.8,
        dwell_time_s: float = 2.0,
        takeoff_altitude_m: float = 4.5,     # altitude at which nav begins
    ) -> None:
        """
        Parameters
        ----------
        waypoints        : List of dicts with keys id, label, x, z.
        target_altitude_m: Cruise altitude (passed through to controller).
        arrival_radius_m : Horizontal radius for waypoint arrival detection.
        dwell_time_s     : Time to hover at each waypoint after arrival.
        takeoff_altitude_m: Altitude above which we consider takeoff complete.
        """
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints are required.")
        if arrival_radius_m <= 0:
            raise ValueError("arrival_radius_m must be positive.")
        if dwell_time_s < 0:
            raise ValueError("dwell_time_s must be non-negative.")

        self._wps: List[Waypoint] = [
            Waypoint(
                id=int(wp['id']),
                label=str(wp.get('label', f'WP{wp["id"]}')),
                x=float(wp['x']),
                z=float(wp['z']),
            )
            for wp in waypoints
        ]

        self.target_altitude_m = target_altitude_m
        self.arrival_radius_m = arrival_radius_m
        self.dwell_time_s = dwell_time_s
        self.takeoff_altitude_m = takeoff_altitude_m

        # Skip waypoint 0 (takeoff point) – drone starts there
        self._current_idx: int = 0
        self._phase: MissionPhase = MissionPhase.TAKEOFF
        self._dwell_start: float = 0.0
        self._reached: List[int] = []   # IDs of reached waypoints
        self._start_time: float = 0.0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Update (call every timestep)
    # ------------------------------------------------------------------

    def update(
        self,
        drone_x: float,
        drone_z: float,
        drone_y: float,       # actual altitude
        current_time_s: float,
    ) -> MissionPhase:
        """
        Advance mission state machine.

        Returns the current MissionPhase after update.
        """
        if not self._initialized:
            self._start_time = current_time_s
            self._initialized = True

        if self._phase == MissionPhase.COMPLETE:
            return self._phase
        if self._phase == MissionPhase.ABORTED:
            return self._phase

        # ---- TAKEOFF: wait until near cruise altitude --------------------
        if self._phase == MissionPhase.TAKEOFF:
            if drone_y >= self.takeoff_altitude_m:
                self._phase = MissionPhase.NAVIGATING
                # First real target is waypoint index 1 (skip home)
                self._current_idx = 1
            return self._phase

        # ---- NAVIGATING: check horizontal arrival at current target ------
        if self._phase == MissionPhase.NAVIGATING:
            wp = self._current_waypoint()
            if wp is None:
                self._phase = MissionPhase.COMPLETE
                return self._phase

            dist = wp.horizontal_distance_to(drone_x, drone_z)
            if dist <= self.arrival_radius_m:
                self._reached.append(wp.id)
                if self.dwell_time_s > 0:
                    self._dwell_start = current_time_s
                    self._phase = MissionPhase.DWELLING
                else:
                    self._advance_waypoint()
            return self._phase

        # ---- DWELLING: hover for dwell_time_s seconds --------------------
        if self._phase == MissionPhase.DWELLING:
            elapsed_dwell = current_time_s - self._dwell_start
            if elapsed_dwell >= self.dwell_time_s:
                self._advance_waypoint()
            return self._phase

        return self._phase

    # ------------------------------------------------------------------
    # Target accessor (always valid; returns last waypoint when complete)
    # ------------------------------------------------------------------

    @property
    def target_x(self) -> float:
        wp = self._current_waypoint()
        return wp.x if wp else self._wps[-1].x

    @property
    def target_z(self) -> float:
        wp = self._current_waypoint()
        return wp.z if wp else self._wps[-1].z

    @property
    def target_y(self) -> float:
        """Always the configured cruise altitude."""
        return self.target_altitude_m

    @property
    def phase(self) -> MissionPhase:
        return self._phase

    @property
    def current_waypoint_index(self) -> int:
        return self._current_idx

    # ------------------------------------------------------------------
    # Status snapshot
    # ------------------------------------------------------------------

    def status(self, elapsed_s: float, dwell_remaining_s: float = 0.0) -> MissionStatus:
        wp = self._current_waypoint()
        next_wp = self._next_waypoint()
        return MissionStatus(
            phase=self._phase,
            current_wp_index=self._current_idx,
            current_wp=wp,
            next_wp=next_wp,
            waypoints_reached=len(self._reached),
            waypoints_total=len(self._wps) - 1,   # exclude home/start
            elapsed_s=elapsed_s,
            dwell_remaining_s=dwell_remaining_s,
        )

    @property
    def waypoints_reached(self) -> int:
        return len(self._reached)

    @property
    def total_waypoints(self) -> int:
        return len(self._wps) - 1  # exclude start

    # ------------------------------------------------------------------
    # Abort
    # ------------------------------------------------------------------

    def abort(self) -> None:
        self._phase = MissionPhase.ABORTED

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_waypoint(self) -> Optional[Waypoint]:
        if self._current_idx < len(self._wps):
            return self._wps[self._current_idx]
        return None

    def _next_waypoint(self) -> Optional[Waypoint]:
        idx = self._current_idx + 1
        if idx < len(self._wps):
            return self._wps[idx]
        return None

    def _advance_waypoint(self) -> None:
        self._current_idx += 1
        if self._current_idx >= len(self._wps):
            self._phase = MissionPhase.COMPLETE
        else:
            self._phase = MissionPhase.NAVIGATING
