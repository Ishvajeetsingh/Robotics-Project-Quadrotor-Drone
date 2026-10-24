"""
test_pid_controller.py
======================
Unit tests for the PIDController class.

Tests cover:
    - Basic proportional response
    - Integral accumulation
    - Derivative calculation
    - Anti-windup integrator clamping
    - Output saturation
    - Reset functionality
    - Invalid gain detection
    - Zero-dt edge case
    - from_config factory method
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from control.pid_controller import PIDController


class TestPIDBasicResponse:
    def test_p_only_positive_error(self):
        """Positive error with P-only controller gives positive output."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0, output_limit=1000.0)
        out = pid.compute(setpoint=5.0, measurement=3.0, current_time=0.0)
        assert out == pytest.approx(4.0, rel=1e-6)

    def test_p_only_negative_error(self):
        """Negative error gives negative output."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0, output_limit=1000.0)
        out = pid.compute(setpoint=3.0, measurement=5.0, current_time=0.0)
        assert out == pytest.approx(-4.0, rel=1e-6)

    def test_zero_error_gives_zero_output(self):
        """Zero error → zero output for P-only."""
        pid = PIDController(kp=5.0, ki=0.0, kd=0.0, output_limit=1000.0)
        out = pid.compute(setpoint=2.0, measurement=2.0, current_time=0.0)
        assert out == pytest.approx(0.0, abs=1e-9)

    def test_integral_accumulates_over_time(self):
        """Integral term grows when error is sustained over multiple steps."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0,
                            integrator_limit=100.0, output_limit=1000.0)
        # 1s interval, error=1.0 → integral should accumulate
        pid.compute(setpoint=1.0, measurement=0.0, current_time=0.0)   # init
        pid.compute(setpoint=1.0, measurement=0.0, current_time=1.0)   # +1 s
        out = pid.compute(setpoint=1.0, measurement=0.0, current_time=2.0)
        # integral ≈ 1+1 = 2, output ≈ 2
        assert out > 1.0

    def test_derivative_on_changing_error(self):
        """Derivative term responds to rate of error change."""
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0, output_limit=1000.0)
        pid.compute(setpoint=0.0, measurement=0.0, current_time=0.0)   # init
        out = pid.compute(setpoint=2.0, measurement=0.0, current_time=1.0)
        # d(error)/dt = 2/1 = 2, output = kd * 2 = 2
        assert out == pytest.approx(2.0, rel=1e-5)


class TestPIDSaturation:
    def test_output_saturates_positive(self):
        """Output must not exceed output_limit."""
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=50.0)
        out = pid.compute(setpoint=10.0, measurement=0.0, current_time=0.0)
        assert out == pytest.approx(50.0)

    def test_output_saturates_negative(self):
        """Output must not go below -output_limit."""
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=50.0)
        out = pid.compute(setpoint=-10.0, measurement=0.0, current_time=0.0)
        assert out == pytest.approx(-50.0)

    def test_integrator_anti_windup(self):
        """Integral term is clamped to integrator_limit."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0,
                            integrator_limit=2.0, output_limit=1000.0)
        pid.compute(setpoint=10.0, measurement=0.0, current_time=0.0)
        # Step over many seconds to drive integral far past limit
        for i in range(1, 20):
            pid.compute(setpoint=10.0, measurement=0.0, current_time=float(i))
        assert abs(pid.integral) <= 2.0 + 1e-6  # clamped


class TestPIDReset:
    def test_reset_clears_integral(self):
        """After reset, integral starts from zero."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0,
                            integrator_limit=100.0, output_limit=1000.0)
        pid.compute(setpoint=5.0, measurement=0.0, current_time=0.0)
        pid.compute(setpoint=5.0, measurement=0.0, current_time=1.0)
        assert abs(pid.integral) > 0
        pid.reset()
        assert pid.integral == 0.0

    def test_reset_allows_fresh_computation(self):
        """After reset, controller behaves as if newly created."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0, output_limit=1000.0)
        pid.compute(setpoint=5.0, measurement=0.0, current_time=0.0)
        pid.reset()
        out = pid.compute(setpoint=5.0, measurement=0.0, current_time=0.0)
        # P-only on first call: kp * error = 2 * 5 = 10
        assert out == pytest.approx(10.0, rel=1e-6)


class TestPIDEdgeCases:
    def test_zero_dt_returns_last_p_term(self):
        """When dt = 0, return saturated P term without updating integral."""
        pid = PIDController(kp=2.0, ki=1.0, kd=0.5, output_limit=1000.0)
        pid.compute(setpoint=5.0, measurement=0.0, current_time=1.0)
        integral_before = pid.integral
        # Same timestamp
        pid.compute(setpoint=5.0, measurement=0.0, current_time=1.0)
        assert pid.integral == integral_before   # integral unchanged

    def test_invalid_negative_gain_raises(self):
        """Negative gains should raise ValueError."""
        with pytest.raises(ValueError):
            PIDController(kp=-1.0, ki=0.0, kd=0.0)

    def test_invalid_integrator_limit_raises(self):
        with pytest.raises(ValueError):
            PIDController(kp=1.0, ki=1.0, kd=0.0, integrator_limit=0.0)

    def test_invalid_output_limit_raises(self):
        with pytest.raises(ValueError):
            PIDController(kp=1.0, ki=0.0, kd=0.0, output_limit=-5.0)


class TestPIDFromConfig:
    def test_from_config_constructs_correctly(self):
        """from_config factory creates a PID with expected gains."""
        gains_dict = {
            'kp': 10.0, 'ki': 2.0, 'kd': 5.0,
            'integrator_limit': 5.0, 'output_limit': 100.0,
        }
        pid = PIDController.from_config(gains_dict, name='TestAlt')
        assert pid.kp == 10.0
        assert pid.ki == 2.0
        assert pid.kd == 5.0
        assert pid.output_limit == 100.0
        assert pid.name == 'TestAlt'

    def test_from_config_missing_optional_keys(self):
        """from_config should use defaults for optional keys."""
        gains_dict = {'kp': 5.0, 'ki': 0.0, 'kd': 1.0}
        pid = PIDController.from_config(gains_dict)
        assert pid.integrator_limit == 10.0   # default
        assert pid.output_limit == 100.0      # default


class TestPIDRateDamping:
    """Verifies that direct measurement_rate correctly provides derivative damping without noise."""

    def test_rate_damping_opposes_velocity(self):
        """When measurement has a positive rate of change, d_term = -kd * rate opposes motion."""
        pid = PIDController(kp=10.0, ki=0.0, kd=2.0)
        # Target = 0, angle = 0, but rotating at +1.5 rad/s
        output = pid.compute(setpoint=0.0, measurement=0.0, current_time=0.1, measurement_rate=1.5)
        # P = 0, D = -2.0 * 1.5 = -3.0
        assert output == pytest.approx(-3.0)

    def test_rate_damping_on_first_step(self):
        """Direct rate provides immediate damping on the very first time step."""
        pid = PIDController(kp=10.0, ki=0.0, kd=3.0)
        output = pid.compute(setpoint=0.0, measurement=0.0, current_time=0.0, measurement_rate=2.0)
        assert output == pytest.approx(-6.0)

