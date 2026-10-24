"""
pid_controller.py
=================
Generic discrete-time PID controller with anti-windup integrator clamping
and output saturation.

All gain parameters are configurable; no hard-coded drone-specific values live
here – those are loaded from config/mission.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PIDState:
    """Mutable runtime state for one PID instance."""
    integral: float = 0.0
    prev_error: float = 0.0
    prev_time: float = 0.0
    initialized: bool = False


class PIDController:
    """
    Discrete PID controller.

    Parameters
    ----------
    kp, ki, kd         : Proportional, integral, derivative gains.
    integrator_limit   : Anti-windup clamp on the integral term (±limit).
    output_limit       : Saturation limit on the controller output (±limit).
    name               : Optional label for debug output.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integrator_limit: float = 10.0,
        output_limit: float = 100.0,
        name: str = "PID",
    ) -> None:
        if kp < 0 or ki < 0 or kd < 0:
            raise ValueError(f"[{name}] PID gains must be non-negative.")
        if integrator_limit <= 0:
            raise ValueError(f"[{name}] integrator_limit must be positive.")
        if output_limit <= 0:
            raise ValueError(f"[{name}] output_limit must be positive.")

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator_limit = integrator_limit
        self.output_limit = output_limit
        self.name = name
        self._state = PIDState()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the controller to an uninitialized state."""
        self._state = PIDState()

    def compute(
        self,
        setpoint: float,
        measurement: float,
        current_time: float,
        measurement_rate: Optional[float] = None,
    ) -> float:
        """
        Compute the next control output.

        Parameters
        ----------
        setpoint         : Desired value.
        measurement      : Actual measured value.
        current_time     : Current simulation time in seconds.
        measurement_rate : Optional direct rate of change of measurement (e.g. gyro rate).
                           When provided, d_term = -kd * measurement_rate (avoids noise/lag).

        Returns
        -------
        float : Saturated control output.
        """
        error = setpoint - measurement

        if not self._state.initialized:
            # First call: no derivative available unless rate provided
            self._state.prev_error = error
            self._state.prev_time = current_time
            self._state.initialized = True
            d_term = -self.kd * measurement_rate if measurement_rate is not None else 0.0
            output = self.kp * error + d_term
            return self._saturate(output)

        dt = current_time - self._state.prev_time
        if dt <= 0.0:
            # Same time step – return last proportional contribution
            return self._saturate(self.kp * error)

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup clamping
        self._state.integral += error * dt
        self._state.integral = self._clamp(
            self._state.integral, self.integrator_limit
        )
        i_term = self.ki * self._state.integral

        # Derivative term: direct rate feedback or backward difference
        if measurement_rate is not None:
            d_term = -self.kd * measurement_rate
        else:
            d_term = self.kd * (error - self._state.prev_error) / dt

        # Update state
        self._state.prev_error = error
        self._state.prev_time = current_time

        output = p_term + i_term + d_term
        return self._saturate(output)

    # ------------------------------------------------------------------
    # Properties for inspection
    # ------------------------------------------------------------------

    @property
    def integral(self) -> float:
        return self._state.integral

    @property
    def last_error(self) -> float:
        return self._state.prev_error

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def _saturate(self, value: float) -> float:
        return self._clamp(value, self.output_limit)

    @classmethod
    def from_config(cls, gains_dict: dict, name: str = "PID") -> "PIDController":
        """
        Construct a PIDController from a gains dictionary loaded from YAML.

        Expected keys: kp, ki, kd, integrator_limit, output_limit
        """
        return cls(
            kp=float(gains_dict['kp']),
            ki=float(gains_dict['ki']),
            kd=float(gains_dict['kd']),
            integrator_limit=float(gains_dict.get('integrator_limit', 10.0)),
            output_limit=float(gains_dict.get('output_limit', 100.0)),
            name=name,
        )

    def __repr__(self) -> str:
        return (
            f"PIDController(name={self.name!r}, "
            f"kp={self.kp}, ki={self.ki}, kd={self.kd}, "
            f"output_limit={self.output_limit})"
        )
