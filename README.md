# RoboPong: UR5e Air Hockey Rebound Simulation

MuJoCo simulation and control of a 6-DOF Universal Robots UR5e manipulator playing air hockey. The robot intercepts and rebounds high-speed pucks (up to 4.8 m/s) with both classical control and reinforcement learning (PPO).

---

## Quick Start

Create and activate your Python virtual environment and run from the `RoboPong/` root directory:

```bash
# Navigate to RoboPong directory and activate virtual environment
git clone https://github.com/felipesanmartin/RoboPong.git
cd RoboPong
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Directory Structure

```
RoboPong/
├── configs/                           # Experiment and PPO hyperparameters
│   └── ppo_config.yaml
├── controllers/                       # Task-space controllers and robot kinematics
│   ├── __init__.py
│   ├── classical_ur5e_controller.py   # Ballistic trajectory prediction & classical controller
│   └── ur5e_kinematics.py             # Damped Least Squares (DLS) kinematics solver
├── envs/                              # Gymnasium simulation environments
│   ├── __init__.py
│   ├── base_ur5e_env.py               # Abstract BaseUR5eEnv (kinematics, physics substepping, perception)
│   ├── hockey_ur5e_env.py             # 23-DOF HockeyUR5eEnv rebound task environment
│   └── rewards.py                     # Modular reward classes (BaseRewardFunction, Rebound, GoalScoring)
├── vision/                            # Camera & perception tools
│   ├── __init__.py
│   └── puck_detector.py               # Bird's-eye camera puck color detector
├── scripts/                           # Main training and execution entry points
│   ├── train_ur5e.py                  # Vectorized parallel PPO training
│   └── run_ur5e.py                    # Trained policy evaluator & interactive visualizer
├── models/                            # Simulation descriptions and meshes
│   └── ur5e/
│       ├── hockey_ur5e_world_walled.xml
│       ├── hockey_ur5e_world.xml
│       ├── ur5e.xml
│       └── assets/                    # Robot link 3D mesh assets (.obj)
└── saved_models/                      # Checkpoints, logs, and evaluation models
    └── ur5e_ppo/
```

---

## Executables & Usage

### 1. Classical Controller (`controllers/classical_ur5e_controller.py`)

Deterministic task-space controller utilizing analytic trajectory reflection prediction, Damped Least Squares (DLS) inverse kinematics, and gravity compensation feedforward.

* **Basic Intercept (with side walls):**
  ```bash
  python3 controllers/classical_ur5e_controller.py --side_walls true
  ```
* **Camera-Based Tracking (using top-down color detector):**
  ```bash
  python3 controllers/classical_ur5e_controller.py --side_walls true --use_camera
  ```
* **Headless Evaluation Mode (5 episodes):**
  ```bash
  python3 controllers/classical_ur5e_controller.py --side_walls true --headless --episodes 5
  ```

---

### 2. Train UR5e (`scripts/train_ur5e.py`)

Trains a continuous-action Proximal Policy Optimization (PPO) agent using Stable-Baselines3 with parallel vectorized environments (`SubprocVecEnv`).

* **Start Training (default 1M timesteps across 8 parallel envs):**
  ```bash
  python3 scripts/train_ur5e.py --num_envs 8 --timesteps 1000000
  ```
* **Quick Test Training (e.g., 20k steps, 2 envs):**
  ```bash
  python3 scripts/train_ur5e.py --num_envs 2 --timesteps 20000
  ```
* Checkpoints and logs are saved automatically to `saved_models/ur5e_ppo/`.

---

### 3. Run UR5e (`scripts/run_ur5e.py`)

Visualizes and evaluates a trained PPO policy in the interactive MuJoCo GUI. Automatically searches for checkpoints in `saved_models/ur5e_ppo/`.

* **Run Default Best Model:**
  ```bash
  python3 scripts/run_ur5e.py
  ```
* **Run a Specific Checkpoint:**
  ```bash
  python3 scripts/run_ur5e.py --model_path saved_models/ur5e_ppo/best_model/best_model.zip
  ```
* **Run with Bird's-Eye Camera Detector:**
  ```bash
  python3 scripts/run_ur5e.py --use_camera
  ```
* **Headless Evaluation Mode:**
  ```bash
  python3 scripts/run_ur5e.py --headless --episodes 10
  ```
