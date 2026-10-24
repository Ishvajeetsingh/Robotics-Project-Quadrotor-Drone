# Engineering Notes and Design Decisions

## Project: Quadrotor Drone Altitude and Waypoint Control
**Course:** AISR — Algorithms for Intelligent Systems and Robotics  
**Platform:** Webots R2023b, DJI Mavic 2 Pro model, Python 3.12

---

## 1. Coordinate System

Webots uses a **right-handed coordinate system with Y pointing up**:
- X = East (positive is East)
- Y = Up (altitude)
- Z = South (positive is South)

All waypoint coordinates in `config/mission.yaml` use this convention.

The GPS sensor returns `[X, Y, Z]` in world frame.  
The InertialUnit returns `[roll, pitch, yaw]` in radians.

---

## 2. Motor Convention (DJI Mavic 2 Pro in Webots)

Webots Mavic 2 Pro motor layout (viewed from above):

```
        +X (East)
          ↑
  FL(CCW) │ FR(CW)
  ─────── │ ───────
          │
   ←──────┼──────→ +Z (South)
          │
  RL(CW)  │ RR(CCW)
```

- **Front-Left (FL)** and **Rear-Right (RR)**: CCW → positive velocity
- **Front-Right (FR)** and **Rear-Left (RL)**: CW → negative velocity

In Webots, `RotationalMotor.setVelocity()` takes a signed angular rate.  
The controller sets `+v` for CCW motors and `-v` for CW motors.

**Motor mixing equations (X-frame quadrotor):**
```
FL = base + roll + pitch - yaw   (CCW)
FR = base - roll + pitch + yaw   (CW,  set to -FR)
RL = base + roll - pitch + yaw   (CW,  set to -RL)
RR = base - roll - pitch - yaw   (CCW)
```

---

## 3. Hover Thrust Constant

From the Webots Mavic 2 Pro example controller:
- `K_VERTICAL_THRUST = 68.5`
- Each motor produces approximately `K_THRUST = (mass * g) / 68.5` N per unit

This baseline is added to the altitude PID output before mixing.  
The tilt compensation `K_VERTICAL_OFFSET * cos(roll) * cos(pitch)` corrects
for power loss when the drone is banked.

---

## 4. PID Tuning Methodology

### Altitude loop

Tuning sequence (Ziegler-Nichols heuristic + manual refinement):

1. Set ki = kd = 0. Increase kp until the drone climbs without excessive
   oscillation. Starting point: kp = 10.
2. Add kd to damp overshoot. Starting point: kd = 5.
3. Add ki to eliminate steady-state error. Starting point: ki = 2.
4. For the tuned experiment, increase kp to 15, kd to 8, ki to 3.

### Attitude loops

Attitude loops must be significantly faster than position loops.  
Roll and pitch kp were set 3–5× higher than the altitude kp.

### Position loops

Position loop output is limited to ±0.4 rad (baseline) / ±0.5 rad (tuned)
to prevent excessive banking that would destabilise the drone.

---

## 5. Standalone Physics Model

The standalone simulator (`simulation/simulate_mission.py`) uses:

- **Euler integration** with dt = 32 ms (matching Webots timestep)
- **First-order attitude lag** (τ = 0.15 s) on roll and pitch
- **Linear drag** proportional to velocity:
  - Horizontal: coefficient 0.35
  - Vertical: coefficient 0.45
- **Thrust decomposition**: `Fy = F * cos(roll) * cos(pitch)`, `Fx = F * sin(pitch)`, `Fz = -F * sin(roll)`
- **Ground clamp**: altitude cannot go below 0 m

This model captures the qualitative behaviour of closed-loop PID control
(settling, overshoot, steady-state error) without the full Webots rigid-body
physics. **Results should be verified against real Webots simulation.**

---

## 6. Settling Time Definition

Altitude settling time is defined as:

> The first time T such that |actual_altitude − desired_altitude| ≤ ε
> continuously for at least D seconds.

Parameters (configurable in `config/mission.yaml`):
- ε (tolerance) = 0.3 m
- D (duration)  = 1.5 s

If the altitude never satisfies this criterion, settling time is reported as
**"Never"** (not fabricated or extrapolated).

---

## 7. Waypoint Arrival Detection

Arrival is checked in the **horizontal plane only** (XZ):

```
distance = sqrt((target_x - actual_x)² + (target_z - actual_z)²)
arrived  = distance ≤ arrival_radius_m
```

Altitude is controlled independently. This matches real-world practice where
horizontal position is the primary survey constraint.

---

## 8. Experiment Design

| | Baseline | Tuned |
|---|---|---|
| **Goal** | Conservative, stable tracking | Faster response, tighter tracking |
| **Altitude kp** | 10.0 | 15.0 |
| **Altitude kd** | 5.0 | 8.0 |
| **Attitude kp** | roll=50, pitch=30 | roll=70, pitch=50 |
| **Position kp** | 1.5 | 2.5 |
| **Expected trade-off** | Lower overshoot, slower settling | Faster settling, possible higher overshoot |

Both experiments use identical waypoints, altitude setpoint, and simulation conditions.

---

## 9. Known Limitations

1. **Standalone simulator** is a simplified point-mass model; it does not
   capture blade flapping, motor dynamics, gyroscopic effects, or air currents.

2. **Yaw control** is minimal in the standalone simulator (yaw is held at 0).
   In Webots, the compass provides absolute heading reference.

3. **Landing** is not implemented — the mission ends at the final waypoint
   and motors are cut. In a real deployment, a controlled descent loop would
   be added.

4. **Webots R2023b** is required for the real simulation. Older versions may
   have slightly different motor device names or physics parameters.

---

## 10. File I/O and Path Resolution

All paths use `pathlib.Path` with `Path(__file__).resolve().parent` anchoring
to avoid hard-coded personal directory paths.  
The project root is resolved as two levels up from any module file.  
All output directories are created with `mkdir(parents=True, exist_ok=True)`.
