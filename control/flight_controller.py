"""
flight_controller.py
====================
Assembles the altitude, attitude, and position PID loops into a single
FlightController that returns motor thrust commands for the Webots
DJI Mavic 2 Pro.

Motor layout (viewed from above):
    Front-Left  (FL) – index 0   spins CCW
    Front-Right (FR) – index 1   spins CW
    Rear-Left   (RL) – index 2   spins CW
    Rear-Right  (RR) – index 3   spins CCW

Webots Mavic 2 Pro rotor velocity convention:
    Positive velocity = CCW when viewed from above.
    FL and RR rotate CCW (+), FR and RL rotate CW (-).
    The built-in physics model derives thrust from ω² so the *speed* value
    (always ≥ 0) is fed as the magnitude; direction is fixed by motor index.

References:
    Webots Mavic 2 Pro example controller (worlds/mavic_2_pro.wbt)
    Webots documentation – Rotational Motor device
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from control.pid_controller import PIDController


# ---------------------------------------------------------------------------
# Constants derived from the DJI Mavic 2 Pro Webots model physics
# ---------------------------------------------------------------------------
# Hover thrust coefficient:  motors must produce ~68.5 units each to hover
# (derived from Webots example controller k_vertical_thrust ≈ 68.5 N/unit)
K_VERTICAL_THRUST: float = 68.5
# Thrust-to-weight ratio headroom for attitude commands
K_VERTICAL_OFFSET: float = 0.6   # radians-per-m/s² coupling offset
# Maximum tilt allowed (radians) – prevents instability
MAX_TILT_RAD: float = 0.5


@dataclass
class DroneState:
    """Current sensor-fused drone state (all in Webots world frame)."""
    time_s: float = 0.0
    # Position
    x: float = 0.0
    y: float = 0.0   # altitude
    z: float = 0.0
    # Attitude (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    # Velocities
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclass
class MotorCommands:
    """Rotor speed setpoints for Webots RotationalMotor devices."""
    fl: float = 0.0   # front-left  (CCW, +)
    fr: float = 0.0   # front-right (CW, -)
    rl: float = 0.0   # rear-left   (CW, -)
    rr: float = 0.0   # rear-right  (CCW, +)

    def as_list(self) -> List[float]:
        return [self.fl, self.fr, self.rl, self.rr]


class FlightController:
    """
    Closed-loop flight controller for the Webots DJI Mavic 2 Pro.

    Implements four PID loops:
        1. Altitude (vertical position → vertical thrust adjustment)
        2. Roll     (lateral attitude stabilisation)
        3. Pitch    (longitudinal attitude stabilisation)
        4. Yaw      (heading control)

    Plus two horizontal position controllers that generate roll and pitch
    setpoints for the attitude loops:
        5. X-axis position → pitch demand
        6. Z-axis position → roll demand
    """

    def __init__(self, gains: Dict, dt_s: float = 0.032) -> None:
        """
        Parameters
        ----------
        gains  : Dictionary loaded from config ('controller_baseline' or
                 'controller_tuned' section of mission.yaml).
        dt_s   : Simulation timestep in seconds.
        """
        self.dt = dt_s
        self.name: str = gains.get('name', 'PID Controller')

        # ---- Altitude loop -----------------------------------------------
        self.alt_pid = PIDController.from_config(
            gains['altitude'], name='Altitude'
        )

        # ---- Attitude loops -----------------------------------------------
        self.roll_pid = PIDController.from_config(
            gains['roll'], name='Roll'
        )
        self.pitch_pid = PIDController.from_config(
            gains['pitch'], name='Pitch'
        )
        self.yaw_pid = PIDController.from_config(
            gains['yaw'], name='Yaw'
        )

        # ---- Position loops (output = desired attitude angle) ------------
        self.x_pos_pid = PIDController.from_config(
            gains['position'], name='X-Position'
        )
        self.z_pos_pid = PIDController.from_config(
            gains['position'], name='Z-Position'
        )

        self._pos_output_limit: float = float(
            gains['position'].get('output_limit', 0.4)
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all PID integrators (e.g. between experiments)."""
        for ctrl in (self.alt_pid, self.roll_pid, self.pitch_pid,
                     self.yaw_pid, self.x_pos_pid, self.z_pos_pid):
            ctrl.reset()

    def compute(
        self,
        state: DroneState,
        target_x: float,
        target_y: float,   # desired altitude
        target_z: float,
        target_yaw: float = 0.0,
    ) -> Tuple[MotorCommands, Dict]:
        """
        Run one control cycle and return motor commands.

        Parameters
        ----------
        state      : Current measured drone state.
        target_x   : Desired X position (m).
        target_y   : Desired altitude   (m).
        target_z   : Desired Z position (m).
        target_yaw : Desired yaw angle  (rad).

        Returns
        -------
        MotorCommands, diagnostics_dict
        """
        t = state.time_s

        # ---- 1. Altitude loop --------------------------------------------
        # Vertical error drives extra thrust above hover baseline
        vertical_input = self.alt_pid.compute(
            setpoint=target_y,
            measurement=state.y,
            current_time=t,
        )

        # ---- 2. Horizontal position loops → desired attitude angles ------
        # Error in world-X drives pitch (nose forward/back)
        desired_pitch = self.x_pos_pid.compute(
            setpoint=target_x,
            measurement=state.x,
            current_time=t,
        )
        desired_pitch = self._clamp(desired_pitch, self._pos_output_limit)

        # Error in world-Z drives roll (strafe left/right)
        # Negative because positive Z is southward; positive roll → drift West
        desired_roll = -self.z_pos_pid.compute(
            setpoint=target_z,
            measurement=state.z,
            current_time=t,
        )
        desired_roll = self._clamp(desired_roll, self._pos_output_limit)

        # ---- 3. Attitude loops -------------------------------------------
        roll_input = self.roll_pid.compute(
            setpoint=desired_roll,
            measurement=state.roll,
            current_time=t,
        )
        pitch_input = self.pitch_pid.compute(
            setpoint=desired_pitch,
            measurement=state.pitch,
            current_time=t,
        )
        yaw_input = self.yaw_pid.compute(
            setpoint=target_yaw,
            measurement=state.yaw,
            current_time=t,
        )

        # ---- 4. Mixer: combine into per-rotor speeds ---------------------
        # Base thrust = hover constant + altitude correction + body-tilt offset
        tilt_comp = K_VERTICAL_OFFSET * math.cos(state.roll) * math.cos(state.pitch)
        base_thrust = K_VERTICAL_THRUST + vertical_input + tilt_comp

        # Motor mixing equations (standard quadrotor X-configuration):
        #   FL = base + roll + pitch - yaw   (CCW)
        #   FR = base - roll + pitch + yaw   (CW  → negative sign convention)
        #   RL = base + roll - pitch + yaw   (CW  → negative)
        #   RR = base - roll - pitch - yaw   (CCW)
        fl = base_thrust + roll_input + pitch_input - yaw_input
        fr = base_thrust - roll_input + pitch_input + yaw_input
        rl = base_thrust + roll_input - pitch_input + yaw_input
        rr = base_thrust - roll_input - pitch_input - yaw_input

        # Clamp to non-negative (rotor cannot reverse)
        fl = max(0.0, fl)
        fr = max(0.0, fr)
        rl = max(0.0, rl)
        rr = max(0.0, rr)

        commands = MotorCommands(fl=fl, fr=fr, rl=rl, rr=rr)

        diagnostics = {
            'vertical_input': vertical_input,
            'desired_roll': desired_roll,
            'desired_pitch': desired_pitch,
            'roll_input': roll_input,
            'pitch_input': pitch_input,
            'yaw_input': yaw_input,
            'base_thrust': base_thrust,
        }

        return commands, diagnostics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))
