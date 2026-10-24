"""
flight_controller_test.py
=========================
Unit tests for the FlightController class.

Tests cover:
    - Correct construction from YAML-style gains dict
    - compute() returns MotorCommands with all four motors
    - Motor commands are non-negative (rotors can't reverse)
    - Altitude error drives thrust higher/lower correctly
    - Reset clears all PID states
    - MotorCommands.as_list returns exactly 4 elements
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from control.flight_controller import FlightController, DroneState, MotorCommands

# Minimal gains dict compatible with FlightController
GAINS = {
    'name': 'Test Controller',
    'altitude': {'kp': 10.0, 'ki': 2.0, 'kd': 5.0, 'integrator_limit': 5.0, 'output_limit': 100.0},
    'roll':     {'kp': 50.0, 'ki': 0.0, 'kd': 5.0, 'integrator_limit': 2.0, 'output_limit': 100.0},
    'pitch':    {'kp': 30.0, 'ki': 0.0, 'kd': 5.0, 'integrator_limit': 2.0, 'output_limit': 100.0},
    'yaw':      {'kp': 2.0,  'ki': 0.5, 'kd': 0.1, 'integrator_limit': 2.0, 'output_limit': 100.0},
    'position': {'kp': 1.5,  'ki': 0.0, 'kd': 0.5, 'integrator_limit': 3.0, 'output_limit': 0.4},
}


def hover_state(t: float = 0.0) -> DroneState:
    """A drone level at 0 m altitude (hovering target at 5 m)."""
    return DroneState(time_s=t, x=0.0, y=0.0, z=0.0,
                      roll=0.0, pitch=0.0, yaw=0.0)


class TestFlightControllerConstruction:
    def test_creates_with_valid_gains(self):
        fc = FlightController(gains=GAINS, dt_s=0.032)
        assert fc.name == 'Test Controller'

    def test_has_all_pid_loops(self):
        fc = FlightController(gains=GAINS)
        for attr in ('alt_pid', 'roll_pid', 'pitch_pid', 'yaw_pid',
                     'x_pos_pid', 'z_pos_pid'):
            assert hasattr(fc, attr)


class TestFlightControllerCompute:
    def test_compute_returns_motor_commands(self):
        fc = FlightController(gains=GAINS)
        state = hover_state(0.0)
        cmds, diag = fc.compute(state, target_x=0.0, target_y=5.0, target_z=0.0)
        assert isinstance(cmds, MotorCommands)

    def test_motor_commands_non_negative(self):
        """All motor speeds must be ≥ 0 (rotors cannot reverse)."""
        fc = FlightController(gains=GAINS)
        state = hover_state(0.0)
        cmds, _ = fc.compute(state, target_x=0.0, target_y=5.0, target_z=0.0)
        for val in cmds.as_list():
            assert val >= 0.0, f"Motor speed {val} is negative"

    def test_as_list_has_four_elements(self):
        fc = FlightController(gains=GAINS)
        state = hover_state(0.0)
        cmds, _ = fc.compute(state, target_x=0.0, target_y=5.0, target_z=0.0)
        lst = cmds.as_list()
        assert len(lst) == 4

    def test_altitude_below_target_increases_thrust(self):
        """Below target altitude → higher thrust than when at target altitude."""
        fc_low = FlightController(gains=GAINS)
        state_low = hover_state(0.0)
        cmds_low, _ = fc_low.compute(state_low, target_x=0.0, target_y=5.0, target_z=0.0)

        fc_on = FlightController(gains=GAINS)
        state_on = DroneState(time_s=0.0, x=0.0, y=5.0, z=0.0)
        cmds_on, _ = fc_on.compute(state_on, target_x=0.0, target_y=5.0, target_z=0.0)

        avg_low = sum(cmds_low.as_list()) / 4
        avg_on  = sum(cmds_on.as_list())  / 4
        assert avg_low > avg_on, (
            f"Below-target thrust ({avg_low:.2f}) should exceed on-target ({avg_on:.2f})"
        )

    def test_diagnostics_dict_has_expected_keys(self):
        fc = FlightController(gains=GAINS)
        state = hover_state(0.0)
        _, diag = fc.compute(state, target_x=0.0, target_y=5.0, target_z=0.0)
        for key in ('vertical_input', 'desired_roll', 'desired_pitch',
                    'roll_input', 'pitch_input', 'yaw_input', 'base_thrust'):
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_gyro_rate_damping_damps_roll(self):
        """When drone has positive roll rate (tilting right), damping creates restoring torque (right > left)."""
        fc = FlightController(gains=GAINS)
        # Level attitude but spinning right (roll_rate = +1.0)
        state_spinning = DroneState(time_s=0.1, x=0.0, y=5.0, z=0.0, roll=0.0, roll_rate=1.0)
        cmds, diag = fc.compute(state_spinning, target_x=0.0, target_y=5.0, target_z=0.0)
        # Rate damping produces negative roll input (less thrust left, more right)
        assert diag['roll_input'] < 0.0
        assert cmds.fr > cmds.fl, "Right motors must exceed left to counteract positive roll rate"
        assert cmds.rr > cmds.rl



class TestFlightControllerReset:
    def test_reset_clears_all_pids(self):
        fc = FlightController(gains=GAINS)
        # Run a few cycles to accumulate state
        for t in range(5):
            fc.compute(hover_state(float(t)), 0.0, 5.0, 0.0)
        # Reset
        fc.reset()
        # Integral should be zero in all loops
        for pid_name in ('alt_pid', 'roll_pid', 'pitch_pid', 'yaw_pid',
                         'x_pos_pid', 'z_pos_pid'):
            pid = getattr(fc, pid_name)
            assert pid.integral == 0.0, f"{pid_name} integral not reset"
            assert not pid._state.initialized, f"{pid_name} not marked un-initialized"


class TestFlightControllerMotorMixing:
    """Verifies that attitude demands map to the correct physical motor differentials."""

    def test_pitch_forward_increases_rear_motors(self):
        """Moving forward (target_x > state.x) requires nose-down tilt: rear motors > front motors."""
        fc = FlightController(gains=GAINS)
        state = DroneState(time_s=0.0, x=0.0, y=5.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
        # target_x = 5.0 (ahead) -> desired pitch > 0 (nose down)
        cmds, diag = fc.compute(state, target_x=5.0, target_y=5.0, target_z=0.0)
        assert diag['desired_pitch'] > 0.0, "Desired pitch should be positive (nose down) to move forward"
        assert diag['pitch_input'] > 0.0
        # Rear motors (RL, RR) must produce more thrust than Front motors (FL, FR)
        assert cmds.rl > cmds.fl, f"RL ({cmds.rl:.2f}) must exceed FL ({cmds.fl:.2f}) to pitch forward"
        assert cmds.rr > cmds.fr, f"RR ({cmds.rr:.2f}) must exceed FR ({cmds.fr:.2f}) to pitch forward"

    def test_roll_right_increases_left_motors(self):
        """Moving right/South (target_z < state.z) requires left-up tilt: left motors > right motors."""
        fc = FlightController(gains=GAINS)
        state = DroneState(time_s=0.0, x=0.0, y=5.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
        # target_z = -5.0 (right/South) -> desired roll > 0 (left up, right down)
        cmds, diag = fc.compute(state, target_x=0.0, target_y=5.0, target_z=-5.0)
        assert diag['desired_roll'] > 0.0, "Desired roll should be positive (left up) to accelerate towards -Z"
        assert diag['roll_input'] > 0.0
        # Left motors (FL, RL) must produce more thrust than Right motors (FR, RR)
        assert cmds.fl > cmds.fr, f"FL ({cmds.fl:.2f}) must exceed FR ({cmds.fr:.2f}) to roll right"
        assert cmds.rl > cmds.rr, f"RL ({cmds.rl:.2f}) must exceed RR ({cmds.rr:.2f}) to roll right"

    def test_yaw_ccw_increases_cw_motors(self):
        """Positive yaw setpoint (CCW) requires CW reaction torque: CW motors (FR, RL) > CCW (FL, RR)."""
        fc = FlightController(gains=GAINS)
        state = DroneState(time_s=0.0, x=0.0, y=5.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
        cmds, diag = fc.compute(state, target_x=0.0, target_y=5.0, target_z=0.0, target_yaw=0.5)
        assert diag['yaw_input'] > 0.0
        # CW motors: FR, RL; CCW motors: FL, RR
        assert cmds.fr > cmds.fl, f"FR ({cmds.fr:.2f}) must exceed FL ({cmds.fl:.2f}) for CCW yaw"
        assert cmds.rl > cmds.rr, f"RL ({cmds.rl:.2f}) must exceed RR ({cmds.rr:.2f}) for CCW yaw"

