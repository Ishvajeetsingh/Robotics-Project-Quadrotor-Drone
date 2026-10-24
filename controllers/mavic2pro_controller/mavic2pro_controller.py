"""
mavic2pro_controller.py
=======================
Webots Python controller for the DJI Mavic 2 Pro quadrotor drone.

This controller implements:
    - Sensor initialization (GPS, IMU, cameras disabled)
    - Closed-loop altitude, attitude, and position PID control
    - Ordered waypoint navigation with arrival detection and dwell
    - Real-time CSV data logging
    - Graceful mission completion and simulation termination

Device names match the official Webots Mavic 2 Pro model:
    Motors  : "front left propeller", "front right propeller",
              "rear left propeller",  "rear right propeller"
    GPS     : "gps"
    IMU     : "inertial unit"
    Gyro    : "gyro"
    Compass : "compass"   (optional, for absolute yaw)

Coordinate system (Webots):
    X = East,  Y = Up,  Z = South   (right-handed)
    GPS returns [X, Y, Z] in world coordinates.
    InertialUnit returns [roll, pitch, yaw] in radians.

Usage:
    Open simulation/mavic_survey.wbt in Webots.
    Webots will automatically launch this script as the robot controller.

IMPORTANT: Do NOT run this file directly with python.exe outside Webots.
    The ``controller`` package is only available inside the Webots runtime.
    For stand-alone testing, run the unit tests in tests/.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Webots runtime import guard
# ---------------------------------------------------------------------------
try:
    from controller import Robot, Motor, GPS, InertialUnit, Gyro, Compass
    _IN_WEBOTS = True
except ImportError:
    _IN_WEBOTS = False
    # Allow import without Webots for static analysis / documentation
    class _Stub:  # type: ignore
        def __getattr__(self, _): return self
        def __call__(self, *a, **kw): return self
    Robot = Motor = GPS = InertialUnit = Gyro = Compass = _Stub()

# ---------------------------------------------------------------------------
# Project imports  (resolve paths relative to the controller file location)
# ---------------------------------------------------------------------------
_CONTROLLER_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT    = _CONTROLLER_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import ConfigLoader
from control.flight_controller import FlightController, DroneState
from navigation.waypoint_manager import WaypointManager, MissionPhase
from simulation.data_logger import DataLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMESTEP_MS: int = 32   # Webots requires timestep set in world; matches config


def _safe_float(val, default: float = 0.0) -> float:
    """Cast a Webots sensor reading to float, returning default on error."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Main controller entry point
# ---------------------------------------------------------------------------

class MavicController:
    """
    Encapsulates the full Webots robot controller lifecycle:
        init → run loop → teardown
    """

    def __init__(
        self,
        config_path: Path | None = None,
        experiment: str = "baseline",
        stage: str = "full",
        max_time_override: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        config_path       : Path to mission.yaml (auto-resolved if None).
        experiment        : 'baseline' or 'tuned' – selects controller gains.
        stage             : 'hover', 'single_waypoint', or 'full'.
        max_time_override : Optional override for max mission time (seconds).
        """
        if not _IN_WEBOTS:
            raise RuntimeError(
                "MavicController requires the Webots Python runtime.\n"
                "Run this file only through Webots, not directly."
            )

        self._experiment = experiment
        self._stage = stage
        self._max_time_override = max_time_override

        # ---- Load configuration ----------------------------------------
        cfg_path = config_path or (_PROJECT_ROOT / "config" / "mission.yaml")
        self._cfg = ConfigLoader(cfg_path)

        # ---- Webots Robot object ----------------------------------------
        self._robot = Robot()
        self._timestep = int(self._robot.getBasicTimeStep())

        # ---- Enable sensors --------------------------------------------
        self._gps      = self._robot.getDevice("gps")
        self._imu      = self._robot.getDevice("inertial unit")
        self._gyro     = self._robot.getDevice("gyro")
        self._compass  = self._robot.getDevice("compass")

        for dev in (self._gps, self._imu, self._gyro, self._compass):
            try:
                dev.enable(self._timestep)
            except Exception as e:
                print(f"[controller] Warning: could not enable device: {e}")

        # ---- Motors ------------------------------------------------------
        # Official Webots Mavic 2 Pro device names:
        motor_names = [
            "front left propeller",
            "front right propeller",
            "rear left propeller",
            "rear right propeller",
        ]
        self._motors = []
        for name in motor_names:
            m = self._robot.getDevice(name)
            m.setPosition(float('inf'))   # velocity control mode
            m.setVelocity(0.0)
            self._motors.append(m)
        # Unpack for readability
        self._m_fl, self._m_fr, self._m_rl, self._m_rr = self._motors

        # ---- Flight controller -----------------------------------------
        gains = self._cfg.controller_gains(experiment)
        dt_s  = self._timestep / 1000.0
        self._controller = FlightController(gains=gains, dt_s=dt_s)

        # ---- Waypoint manager (configured for stage) -------------------
        if stage == "hover":
            wps = [
                {'id': 0, 'label': 'Takeoff / Hover', 'x': 0.0, 'z': 0.0},
                {'id': 1, 'label': 'Hold Hover', 'x': 0.0, 'z': 0.0},
            ]
            dwell = 999.0
        elif stage == "single_waypoint":
            wps = self._cfg.waypoints[:2]
            dwell = self._cfg.dwell_time_s
        else:
            wps = self._cfg.waypoints
            dwell = self._cfg.dwell_time_s

        self._wp_manager = WaypointManager(
            waypoints=wps,
            target_altitude_m=self._cfg.target_altitude_m,
            arrival_radius_m=self._cfg.arrival_radius_m,
            dwell_time_s=dwell,
        )

        # ---- Data logger -----------------------------------------------
        log_dir  = _PROJECT_ROOT / self._cfg.log_dir / f"experiment_{experiment}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"flight_log_{experiment}.csv"
        self._logger = DataLogger(
            output_path=log_dir / log_name,
            fields=self._cfg.log_fields,
        )
        effective_max = self._max_time_override or self._cfg.max_mission_time_s
        print(f"[controller] Logging to: {log_dir / log_name}")
        print(f"[controller] Experiment : {experiment}")
        print(f"[controller] Stage      : {stage}")
        print(f"[controller] Controller : {gains.get('name', experiment)}")
        print(f"[controller] Waypoints  : {len(wps)}")
        print(f"[controller] Max time   : {effective_max:.1f} s")

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the survey mission until completion or timeout."""
        max_time = self._max_time_override or self._cfg.max_mission_time_s
        step_count = 0
        prev_alt = None
        start_alt = None
        dt_s = self._timestep / 1000.0

        try:
            while self._robot.step(self._timestep) != -1:
                t = self._robot.getTime()

                # ---- Timeout guard ------------------------------------
                if t > max_time:
                    print(f"[controller] Mission timeout at {t:.1f} s")
                    break

                # ---- Read sensors (Webots ENU frame) ------------------
                # In Webots ENU:
                #   gps_vals[0] = East (X)
                #   gps_vals[1] = North (Y) -> mapped to internal horizontal coordinate 'z'
                #   gps_vals[2] = Up (Z)    -> mapped to internal altitude coordinate 'y'
                gps_vals  = self._gps.getValues()
                imu_vals  = self._imu.getRollPitchYaw()
                gyro_vals = self._gyro.getValues()

                raw_x     = _safe_float(gps_vals[0])
                raw_north = _safe_float(gps_vals[1])
                raw_alt   = _safe_float(gps_vals[2])

                # Guard against non-finite readings
                if not (math.isfinite(raw_x) and math.isfinite(raw_north) and math.isfinite(raw_alt)):
                    continue

                if prev_alt is None:
                    vy = 0.0
                else:
                    vy = (raw_alt - prev_alt) / dt_s
                prev_alt = raw_alt

                state = DroneState(
                    time_s=t,
                    x=raw_x,
                    y=raw_alt,    # Altitude in ENU is the 3rd GPS component
                    z=raw_north,  # North horizontal coordinate in ENU
                    roll=_safe_float(imu_vals[0]),
                    pitch=_safe_float(imu_vals[1]),
                    yaw=_safe_float(imu_vals[2]),
                    vy=_safe_float(vy),
                    roll_rate=_safe_float(gyro_vals[0]),
                    pitch_rate=_safe_float(gyro_vals[1]),
                    yaw_rate=_safe_float(gyro_vals[2]),
                )

                # ---- Safety Divergence Watchdog -----------------------
                if (
                    abs(state.x) > 30.0
                    or abs(state.z) > 30.0
                    or state.y > 15.0
                    or state.y < -1.0
                    or abs(state.roll) > 1.2
                    or abs(state.pitch) > 1.2
                ):
                    print(
                        f"[controller] SAFETY WATCHDOG TRIGGERED at {t:.2f} s: State divergence detected!\n"
                        f"  Pos: (X={state.x:.2f}, Alt={state.y:.2f}, Z={state.z:.2f})\n"
                        f"  Attitude: (roll={math.degrees(state.roll):.1f} deg, pitch={math.degrees(state.pitch):.1f} deg)\n"
                        f"  Aborting mission and cutting motor power."
                    )
                    for m in self._motors:
                        m.setVelocity(0.0)
                    self._wp_manager.abort()
                    break

                # ---- Mission state machine ----------------------------
                phase = self._wp_manager.update(
                    drone_x=state.x,
                    drone_z=state.z,
                    drone_y=state.y,
                    current_time_s=t,
                )

                if start_alt is None:
                    start_alt = state.y

                if phase == MissionPhase.TAKEOFF:
                    if state.y < 0.20:
                        self._controller.reset()
                    target_x = state.x
                    target_z = state.z
                else:
                    target_x = self._wp_manager.target_x
                    target_z = self._wp_manager.target_z

                # Smooth altitude trajectory during takeoff to avoid aggressive liftoff
                if phase == MissionPhase.TAKEOFF:
                    climb_speed = 1.2
                    target_y = min(self._wp_manager.target_y, start_alt + climb_speed * t)
                else:
                    target_y = self._wp_manager.target_y

                wp_id = self._wp_manager.current_waypoint_index

                # ---- Flight controller --------------------------------
                commands, _diag = self._controller.compute(
                    state=state,
                    target_x=target_x,
                    target_y=target_y,
                    target_z=target_z,
                )

                # ---- Set motor velocities (clamped to physical limits) --
                # FL and RR spin CCW (+velocity), FR and RL spin CW (-velocity)
                MAX_VELOCITY = 576.0
                v_fl = min(MAX_VELOCITY, max(0.0, commands.fl))
                v_fr = min(MAX_VELOCITY, max(0.0, commands.fr))
                v_rl = min(MAX_VELOCITY, max(0.0, commands.rl))
                v_rr = min(MAX_VELOCITY, max(0.0, commands.rr))

                self._m_fl.setVelocity( v_fl)
                self._m_fr.setVelocity(-v_fr)
                self._m_rl.setVelocity(-v_rl)
                self._m_rr.setVelocity( v_rr)

                # ---- Log data ----------------------------------------
                self._logger.log_step(
                    time_s=t,
                    waypoint_id=wp_id,
                    desired_x=target_x,
                    desired_y=target_y,
                    desired_z=target_z,
                    actual_x=state.x,
                    actual_y=state.y,
                    actual_z=state.z,
                    roll=state.roll,
                    pitch=state.pitch,
                    yaw=state.yaw,
                    motor_fl=v_fl,
                    motor_fr=v_fr,
                    motor_rl=v_rl,
                    motor_rr=v_rr,
                )

                step_count += 1

                # ---- Progress reporting (every 3 seconds) -----------
                if step_count % (3000 // self._timestep) == 0:
                    dist_to_wp = math.sqrt((target_x - state.x)**2 + (target_z - state.z)**2)
                    print(
                        f"[{t:7.1f}s] Phase={phase.name:12s} "
                        f"WP={wp_id}  "
                        f"Pos=({state.x:.2f},{state.z:.2f}) Alt={state.y:.2f}m  "
                        f"Target=({target_x:.2f},{target_z:.2f}) Alt={target_y:.2f}m  "
                        f"Dist={dist_to_wp:.2f}m  "
                        f"WPs reached={self._wp_manager.waypoints_reached}"
                    )

                # ---- Mission complete ---------------------------------
                if phase == MissionPhase.COMPLETE:
                    print(
                        f"[controller] Mission COMPLETE at {t:.1f} s  "
                        f"({self._wp_manager.waypoints_reached}/"
                        f"{self._wp_manager.total_waypoints} waypoints)"
                    )
                    # Coast to a stop
                    for motor in self._motors:
                        motor.setVelocity(0.0)
                    self._robot.step(self._timestep * 50)  # ~1.6 s coast
                    break

        finally:
            self._logger.close()
            print(f"[controller] Log rows written: {self._logger.row_count}")
            print(f"[controller] Log saved: {self._logger.output_path}")


# ---------------------------------------------------------------------------
# Entry point (launched by Webots runtime)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import shlex

    parser = argparse.ArgumentParser(description="Mavic 2 Pro survey controller")
    parser.add_argument(
        '--experiment',
        choices=['baseline', 'tuned'],
        default='baseline',
        help='Controller gain set to use (default: baseline)',
    )
    parser.add_argument(
        '--stage',
        choices=['hover', 'single_waypoint', 'full'],
        default='full',
        help='Verification stage to run (default: full)',
    )
    parser.add_argument(
        '--max-time',
        type=float,
        default=None,
        help='Override max mission duration in seconds',
    )

    # Handle Webots passing controllerArgs as single string in argv[1]
    raw_argv = sys.argv[1:]
    if len(raw_argv) == 1 and ('--' in raw_argv[0] or ' ' in raw_argv[0]):
        raw_argv = shlex.split(raw_argv[0])

    args, _ = parser.parse_known_args(raw_argv)

    ctrl = MavicController(
        experiment=args.experiment,
        stage=args.stage,
        max_time_override=args.max_time,
    )
    ctrl.run()
