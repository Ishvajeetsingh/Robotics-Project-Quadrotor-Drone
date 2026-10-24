# Quadrotor Drone Altitude and Waypoint Control

**AISR (Algorithms for Intelligent Systems and Robotics) — University Project**

A complete Python simulation of a DJI Mavic 2 Pro quadrotor performing an
autonomous site-survey mission with closed-loop PID control, waypoint
navigation, data logging, performance analysis, and quantitative controller
comparison.

---

## Project Overview

This project implements:

| Feature | Description |
|---|---|
| **Closed-loop PID control** | Altitude, roll, pitch, yaw, and XZ-position loops |
| **Waypoint navigation** | Ordered survey route with arrival detection and dwell |
| **Standalone simulator** | Physics-based integrator — runs without Webots |
| **Webots controller** | Full Webots DJI Mavic 2 Pro Python controller |
| **CSV data logging** | Real-time logging of desired/actual state and errors |
| **Performance metrics** | RMSE, settling time, overshoot, waypoint success rate |
| **Graphs** | Altitude, XZ position, 3-D trajectory, error plots |
| **Comparison experiments** | Baseline vs tuned PID gains with quantitative table |
| **Automated tests** | 70+ unit tests across all modules |

---

## Directory Structure

```
Robotics Project Quadrotor Drone/
│
├── config/
│   ├── mission.yaml              ← Waypoints, PID gains, logging settings
│   └── config_loader.py          ← YAML loader with typed accessors
│
├── control/
│   ├── pid_controller.py         ← Generic discrete PID with anti-windup
│   └── flight_controller.py      ← Altitude + attitude + position controller
│
├── navigation/
│   └── waypoint_manager.py       ← State-machine waypoint sequencer
│
├── simulation/
│   ├── mavic_survey.wbt          ← Webots world file (R2023a/b)
│   ├── simulate_mission.py       ← Standalone physics simulator
│   └── data_logger.py            ← Real-time CSV logger
│
├── controllers/
│   └── mavic2pro_controller/
│       └── mavic2pro_controller.py ← Webots robot controller script
│
├── analysis/
│   ├── metrics.py                ← All performance metrics
│   └── plotter.py                ← Matplotlib graph generation
│
├── scripts/
│   ├── run_experiment.py         ← End-to-end experiment runner
│   └── run_analysis.py           ← Analysis and plotting script
│
├── tests/
│   ├── test_pid_controller.py
│   ├── test_flight_controller.py
│   ├── test_waypoint_manager.py
│   ├── test_data_logger.py
│   ├── test_metrics.py
│   └── test_config_loader.py
│
├── results/
│   ├── experiment_baseline/      ← CSV logs for baseline experiment
│   ├── experiment_tuned/         ← CSV logs for tuned experiment
│   ├── graphs/                   ← Generated PNG plots
│   └── metrics/                  ← JSON metrics and comparison table
│
├── docs/
│   └── engineering_notes.md      ← Design decisions and tuning log
│
├── README.md
├── requirements.txt
├── pytest.ini
└── .gitignore
```

---

## Requirements

### Python packages

```
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
scipy >= 1.10
pyyaml >= 6.0
pytest >= 7.4
```

Install with:

```powershell
pip install -r requirements.txt
```

### Webots (for real simulation only)

Download **Webots R2023b** (or newer compatible release) from:
https://cyberbotics.com/

> **Note:** The standalone simulator (`simulation/simulate_mission.py`) runs
> entirely without Webots. Install Webots only to run the real physics simulation.

---

## Installation

```powershell
# 1. Clone / open the project
cd "C:\Users\ISHVAJEET\Documents\Robotics Project Quadrotor Drone"

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import numpy, pandas, matplotlib, scipy; print('All packages OK')"
```

---

## Running the Standalone Simulator (No Webots Required)

The standalone simulator approximates the drone dynamics with a point-mass
physics model. It produces CSV logs that feed the full analysis pipeline.

```powershell
# Run both baseline and tuned experiments, then generate all analysis/graphs:
python scripts/run_experiment.py

# Run only the baseline experiment:
python scripts/run_experiment.py --experiment baseline

# Run the simulator without analysis:
python scripts/run_experiment.py --no-analysis

# Or run the simulator directly:
python simulation/simulate_mission.py --experiment both
```

Output files:
- `results/experiment_baseline/flight_log_baseline.csv`
- `results/experiment_tuned/flight_log_tuned.csv`

---

## Running the Analysis

```powershell
python scripts/run_analysis.py
```

This produces:
- `results/graphs/baseline/`  — 6 PNG plots for the baseline experiment
- `results/graphs/tuned/`     — 6 PNG plots for the tuned experiment
- `results/graphs/comparison_chart.png`  — Bar chart comparing both controllers
- `results/metrics/metrics_baseline.json`
- `results/metrics/metrics_tuned.json`
- `results/metrics/comparison_table.txt` — Formatted numeric comparison

---

## Running Automated Tests

```powershell
# Run all unit tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_pid_controller.py -v
python -m pytest tests/test_metrics.py -v
python -m pytest tests/test_waypoint_manager.py -v
```

Test coverage:

| File | What is tested |
|---|---|
| `test_pid_controller.py` | P/I/D terms, saturation, anti-windup, reset, from_config |
| `test_flight_controller.py` | Motor commands, thrust sign, diagnostics, reset |
| `test_waypoint_manager.py` | State machine, arrival detection, dwell, abort |
| `test_data_logger.py` | CSV creation, row writing, error columns, context manager |
| `test_metrics.py` | RMSE, settling time, overshoot, load_log validation |
| `test_config_loader.py` | YAML load, validation, all accessors |

---

## Running the Real Webots Simulation

> **Prerequisite:** Webots R2023b must be installed.
> Download: https://cyberbotics.com/

### Steps

1. **Open the world file** in Webots:
   - File → Open World → browse to `simulation/mavic_survey.wbt`

2. The controller `mavic2pro_controller` is already set in the world file.
   Webots will launch it automatically.

3. To switch between baseline and tuned gains, edit the drone's
   `controllerArgs` in the world file:
   ```
   controllerArgs "--experiment baseline"
   controllerArgs "--experiment tuned"
   ```
   Or modify the `Mavic2Pro` node in Webots directly.

4. Press **Play** (▶) in Webots.

5. Progress is printed to the Webots console.

6. When the mission completes, the CSV log is written to:
   `results/experiment_<name>/flight_log_<name>.csv`

7. Run the analysis:
   ```powershell
   python scripts/run_analysis.py
   ```

### Controller device names (Webots Mavic 2 Pro)

| Device | Webots name |
|---|---|
| GPS | `gps` |
| IMU (InertialUnit) | `inertial unit` |
| Gyro | `gyro` |
| Compass | `compass` |
| Front-left motor | `front left propeller` |
| Front-right motor | `front right propeller` |
| Rear-left motor | `rear left propeller` |
| Rear-right motor | `rear right propeller` |

---

## Survey Mission Waypoints

Defined in `config/mission.yaml`. Default route (rectangular survey):

| ID | Label | X (m) | Z (m) |
|---|---|---|---|
| 0 | Takeoff / Home | 0.0 | 0.0 |
| 1 | Survey Point A | 5.0 | 0.0 |
| 2 | Survey Point B | 5.0 | -5.0 |
| 3 | Survey Point C | 0.0 | -5.0 |
| 4 | Survey Point D | -5.0 | -5.0 |
| 5 | Survey Point E | -5.0 | 0.0 |
| 6 | Return Home | 0.0 | 0.0 |

Cruise altitude: **5.0 m**. Arrival radius: **0.8 m**. Dwell time: **2.0 s**.

To change waypoints, edit `config/mission.yaml` — no code changes required.

---

## Comparison Experiments

| Parameter | Baseline | Tuned |
|---|---|---|
| Altitude kp | 10.0 | 15.0 |
| Altitude ki | 2.0 | 3.0 |
| Altitude kd | 5.0 | 8.0 |
| Roll kp | 50.0 | 70.0 |
| Position kp | 1.5 | 2.5 |

The tuned controller uses higher proportional and derivative gains for faster
response, at the cost of potentially higher instantaneous overshoot.

---

## Output Graphs

Each experiment produces:

1. **Altitude tracking** — desired vs actual altitude over time
2. **X-position tracking** — desired vs actual X over time
3. **Z-position tracking** — desired vs actual Z over time
4. **3-D trajectory** — bird's-eye and 3-D view of the flight path
5. **Position errors** — |X|, |Z|, and 3-D Euclidean error over time
6. **Altitude error** — altitude tracking and absolute error panel

Plus a **comparison bar chart** across both experiments.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: pyyaml` | `pip install pyyaml` |
| `pandas` import error | `pip install --upgrade pandas numpy` |
| `No module named 'controller'` | Expected — the Webots `controller` module is only available inside Webots. Run the standalone simulator instead. |
| Webots world does not load | Ensure Webots R2023b is installed; older versions may use different device names. |
| CSV log is empty after Webots run | Check the Webots console for Python exceptions; verify `PYTHONPATH` includes the project root. |
| Tests fail with import errors | Run `pip install -r requirements.txt` to install all dependencies. |

---

## Data Provenance

All data in this repository is clearly labelled:

- **Standalone simulator output** — generated by `simulation/simulate_mission.py`.
  Labelled as *"Standalone (no Webots)"* in all output file names.
- **Webots simulation output** — generated by the Webots controller.
  Stored in `results/experiment_<name>/`.
- **Synthetic test data** — used only inside the unit tests; never saved to disk.

No results have been fabricated or hard-coded.

---

## Relevant AISR Topics

- **Closed-Loop Control** — PID altitude, attitude, and position loops
- **Stability and Performance** — settling time, overshoot, RMSE metrics
- **Moving** — autonomous waypoint navigation state machine
- **Application Domains** — infrastructure inspection, surveying, delivery drones
