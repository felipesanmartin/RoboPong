"""Base Gymnasium Environment for 6-DOF UR5e Manipulator in MuJoCo Air Hockey Arena.

Provides common physical modeling, DLS inverse kinematics execution, camera/puck
state estimation, action application, substepping, and lifecycle management.
Task-specific environments (e.g., rebound defense, goal shooting) inherit from this
base class and provide their own observation spaces, reset conditions, task logic,
and reward functions.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

# Ensure project root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from controllers.ur5e_kinematics import UR5eKinematics
from vision.puck_detector import PuckColorDetector


class BaseUR5eEnv(gym.Env, ABC):
    """Abstract Base Class for UR5e Air Hockey tasks in MuJoCo.

    Handles:
        - MuJoCo model and data loading (walled or open-court)
        - UR5e kinematics solver initialization and DLS IK command tracking
        - Actuator low-level control, frame skipping, and gravity compensation
        - Optional vision-based camera perception for puck state estimation
        - Paddle and puck collision detection
        - Passive interactive viewer rendering and cleanup
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: Optional[str] = None,
        side_walls: bool = True,
        use_camera: bool = False,
        show_camera: bool = False,
        frame_skip: int = 20,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.side_walls = side_walls
        self.use_camera = use_camera
        self.show_camera = show_camera
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Resolve model XML path if not explicitly provided
        if model_path is None:
            xml_file = (
                "hockey_ur5e_world_walled.xml"
                if side_walls
                else "hockey_ur5e_world.xml"
            )
            candidate_dirs = [
                os.path.join(_project_root, "models", "ur5e"),
                os.path.join(_current_dir, "models", "ur5e"),
            ]
            for c_dir in candidate_dirs:
                c_path = os.path.join(c_dir, xml_file)
                if os.path.exists(c_path):
                    model_path = c_path
                    break
            if model_path is None:
                model_path = os.path.join(_project_root, "models", "ur5e", xml_file)

        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Kinematics solver
        self.kin = UR5eKinematics(self.model, site_name="paddle_site")

        # Geom and body IDs
        self.paddle_geom_id = self.model.geom("paddle_geom").id
        self.puck_geom_id = self.model.geom("puck_geom").id
        self.puck_body_id = self.model.body("hockey_puck").id
        self.site_id = self.model.site("paddle_site").id

        # Workspace limits on table surface (Z = 0.035 m)
        self.x_bounds = (-1.88, -1.45)
        self.y_bounds = (-0.70, 0.70)
        self.z_plane = 0.035
        self.base_xe = -1.75

        # Maximum Cartesian velocities
        self.max_vel_x = 1.8  # m/s forward/back
        self.max_vel_y = 2.5  # m/s lateral
        self.dt_ctrl = self.frame_skip * self.model.opt.timestep  # e.g. 0.02s (50 Hz)

        # Optional camera detector
        self.detector = None
        self.last_camera_frame = None
        if self.use_camera or self.show_camera:
            self.detector = PuckColorDetector(self.model, camera_name="top_down")

        # Action space: 2 continuous velocities [v_x, v_y] normalized in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Subclasses must define observation_space
        self.observation_space: spaces.Space = None

        # Internal state tracking
        self.step_count = 0
        self.target_pos = np.array([self.base_xe, 0.0, self.z_plane], dtype=np.float64)
        self.q_cmd = np.zeros(6, dtype=np.float64)
        self.prev_paddle_pos = np.zeros(2, dtype=np.float64)
        self.paddle_vel = np.zeros(2, dtype=np.float64)

        # Viewer handle
        self.viewer = None

    def get_paddle_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end-effector paddle 2D position (x, y) and velocity (vx, vy)."""
        pos = self.data.site_xpos[self.site_id][:2].copy()
        vel = self.paddle_vel.copy()
        return pos, vel

    def get_puck_state(self) -> Tuple[float, float, float, float]:
        """Get puck position (x, y) and velocity (vx, vy).

        Uses offscreen camera detector if use_camera is enabled; otherwise uses
        simulator ground-truth physics state.
        """
        if self.use_camera and self.detector is not None:
            puck_x, puck_y, puck_vx, puck_vy, self.last_camera_frame, _ = (
                self.detector.detect(self.data)
            )
        else:
            puck_x = -0.5 + float(self.data.qpos[6])
            puck_y = float(self.data.qpos[7])
            puck_vx = float(self.data.qvel[6])
            puck_vy = float(self.data.qvel[7])
            if self.show_camera and self.detector is not None:
                _, _, _, _, self.last_camera_frame, _ = self.detector.detect(self.data)

        return puck_x, puck_y, puck_vx, puck_vy

    def apply_action(self, action: np.ndarray) -> bool:
        """Integrate Cartesian action, solve DLS IK, substep physics, and check contact.

        Args:
            action: 2D continuous action [v_x, v_y] in [-1.0, 1.0].

        Returns:
            Boolean indicating whether paddle-puck contact occurred during this step.
        """
        action = np.clip(action, -1.0, 1.0)

        # 1. Integrate Cartesian action to target paddle position
        dx = float(action[0] * self.max_vel_x * self.dt_ctrl)
        dy = float(action[1] * self.max_vel_y * self.dt_ctrl)

        self.target_pos[0] = np.clip(
            self.target_pos[0] + dx, self.x_bounds[0], self.x_bounds[1]
        )
        self.target_pos[1] = np.clip(
            self.target_pos[1] + dy, self.y_bounds[0], self.y_bounds[1]
        )
        self.target_pos[2] = self.z_plane

        # 2. DLS Inverse Kinematics
        q_sol, ok = self.kin.inverse(self.target_pos, q_init=self.q_cmd, max_iter=25)
        if ok:
            max_dq = 0.30  # rad per step (~15 rad/s)
            dq = np.clip(q_sol - self.q_cmd, -max_dq, max_dq)
            self.q_cmd = self.q_cmd + dq
        else:
            self.q_cmd = 0.85 * self.q_cmd + 0.15 * q_sol

        self.data.ctrl[:6] = self.q_cmd

        # 3. Physics Substepping with gravity compensation
        contact_detected = False
        for _ in range(self.frame_skip):
            self.data.qfrc_applied[:6] = self.kin.get_gravity_compensation(self.data)
            mujoco.mj_step(self.model, self.data)

            if not contact_detected:
                for c in range(self.data.ncon):
                    con = self.data.contact[c]
                    if (
                        con.geom1 == self.paddle_geom_id
                        and con.geom2 == self.puck_geom_id
                    ) or (
                        con.geom1 == self.puck_geom_id
                        and con.geom2 == self.paddle_geom_id
                    ):
                        contact_detected = True
                        break

        # 4. Measure paddle actual velocity
        curr_paddle_pos = self.data.site_xpos[self.site_id][:2].copy()
        self.paddle_vel = (curr_paddle_pos - self.prev_paddle_pos) / self.dt_ctrl
        self.prev_paddle_pos = curr_paddle_pos

        return contact_detected

    def reset_robot_ready(self):
        """Reset robot joint positions and controllers to nominal ready keyframe."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.key("ready").qpos
        self.data.ctrl[:] = self.model.key("ready").ctrl

        self.target_pos = np.array([self.base_xe, 0.0, self.z_plane], dtype=np.float64)
        self.q_cmd = self.data.qpos[:6].copy()
        mujoco.mj_forward(self.model, self.data)

        self.prev_paddle_pos = self.data.site_xpos[self.site_id][:2].copy()
        self.paddle_vel = np.zeros(2, dtype=np.float64)

        if self.detector is not None:
            self.detector.reset()

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector for this task."""
        raise NotImplementedError

    def render(self):
        """Render the environment according to render_mode."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 4.2
                self.viewer.cam.elevation = -40.0
                self.viewer.cam.azimuth = 90.0
                self.viewer.cam.lookat = [-0.5, 0.0, 0.0]
            self.viewer.sync()

            if (self.use_camera or self.show_camera) and self.detector is not None:
                if self.last_camera_frame is not None:
                    self.detector.show(self.last_camera_frame)

        elif self.render_mode == "rgb_array":
            if self.detector is not None and self.last_camera_frame is not None:
                return self.last_camera_frame

    def close(self):
        """Release viewer and perception resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.detector is not None:
            self.detector.close()
            self.detector = None
