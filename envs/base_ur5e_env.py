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
from typing import List, Optional, Tuple

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


class PaddleArm:
    """Joint indices and paddle tracking state for one UR5e arm in a MuJoCo model.

    A mirrored arm faces the other way, rotated 180 degrees about x = mirror_x, and works
    in its own local frame where it looks exactly like the single-arm robot.
    """

    JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")

    def __init__(self, model: mujoco.MjModel, prefix: str = "", mirror_x: Optional[float] = None):
        self.mirror_x = mirror_x
        self.qpos_idx = np.array([model.joint(prefix + j).qposadr[0] for j in self.JOINTS])
        self.dof_idx = np.array([model.joint(prefix + j).dofadr[0] for j in self.JOINTS])
        self.act_idx = np.array([model.actuator(prefix + a).id for a in self.ACTUATORS])
        self.paddle_geom_id = model.geom(prefix + "paddle_geom").id
        self.site_name = prefix + "paddle_site"
        self.site_id = model.site(self.site_name).id

        self.target = np.zeros(3, dtype=np.float64)
        self.q_cmd = np.zeros(6, dtype=np.float64)
        self.prev_paddle = np.zeros(2, dtype=np.float64)
        self.paddle_vel = np.zeros(2, dtype=np.float64)
        self.contact = False

    def to_local(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """Convert a world position and velocity into this arm's frame."""
        if self.mirror_x is None:
            return x, y, vx, vy
        return 2.0 * self.mirror_x - x, -y, -vx, -vy

    def paddle_world(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.site_id][:2].copy()

    def reset(self, data: mujoco.MjData, target: np.ndarray):
        """Call after mj_forward so the paddle position is current."""
        self.target = np.array(target, dtype=np.float64)
        self.q_cmd = data.qpos[self.qpos_idx].copy()
        self.prev_paddle = self.paddle_world(data)
        self.paddle_vel = np.zeros(2, dtype=np.float64)
        self.contact = False

    def drive(self, kin: UR5eKinematics, data: mujoco.MjData):
        """Rate-limited DLS IK step toward self.target (local frame)."""
        q_sol, ok = kin.inverse(self.target, q_init=self.q_cmd, max_iter=25)
        if ok:
            max_dq = 0.30  # rad per step (~15 rad/s)
            dq = np.clip(q_sol - self.q_cmd, -max_dq, max_dq)
            self.q_cmd = self.q_cmd + dq
        else:
            self.q_cmd = 0.85 * self.q_cmd + 0.15 * q_sol
        data.ctrl[self.act_idx] = self.q_cmd

    def update_velocity(self, data: mujoco.MjData, dt: float):
        """Measured paddle velocity in this arm's frame."""
        cur = self.paddle_world(data)
        vel = (cur - self.prev_paddle) / dt
        self.paddle_vel = vel if self.mirror_x is None else -vel
        self.prev_paddle = cur


class HitCounter:
    """Counts paddle or wall contacts once each, ignoring repeats within min_gap seconds."""

    def __init__(self, min_gap: float, rising_edge: bool):
        self.min_gap = min_gap
        self.rising_edge = rising_edge
        self.reset()

    def reset(self):
        self.count = 0
        self.last_time = -1.0
        self.prev_contact = False

    def update(self, contact: bool, time: float, valid: bool = True) -> bool:
        hit = contact and valid and time - self.last_time > self.min_gap
        if self.rising_edge:
            hit = hit and not self.prev_contact
            self.prev_contact = contact
        if hit:
            self.count += 1
            self.last_time = time
        return hit


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
        arm_prefixes: Tuple[str, ...] = ("",),
        mirror_x: float = 0.0,
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

        # Every arm after the first is mirrored about x = mirror_x
        self.arms: List[PaddleArm] = [
            PaddleArm(self.model, prefix, mirror_x=None if i == 0 else mirror_x)
            for i, prefix in enumerate(arm_prefixes)
        ]

        # IK always runs on the first arm, in each arm's local frame (UR5eKinematics uses the first 6 joints)
        assert list(self.arms[0].qpos_idx) == list(range(6))
        self.kin = UR5eKinematics(self.model, site_name=self.arms[0].site_name)

        # Geom and body IDs
        self.paddle_geom_id = self.arms[0].paddle_geom_id
        self.puck_geom_id = self.model.geom("puck_geom").id
        self.puck_body_id = self.model.body("hockey_puck").id
        self.site_id = self.arms[0].site_id
        self.puck_qpos = self.model.joint("puck_x").qposadr[0]
        self.puck_dof = self.model.joint("puck_x").dofadr[0]
        self.back_wall_geom_ids = {
            self.model.geom(name).id
            for name in ("front_wall", "front_wall_left", "front_wall_right")
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
        }
        # x position of the back wall's inner face
        self.back_wall_x = 1.95

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
        # True if the puck touched the back wall during the last apply_action()
        self.wall_contact_detected = False

        # Viewer handle and camera framing
        self.viewer = None
        self.viewer_distance, self.viewer_lookat = 4.2, [-0.5, 0.0, 0.0]

    def get_paddle_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end-effector paddle 2D position (x, y) and velocity (vx, vy)."""
        arm = self.arms[0]
        return arm.paddle_world(self.data), arm.paddle_vel.copy()

    def get_puck_state(self, ground_truth: bool = False) -> Tuple[float, float, float, float]:
        """Get puck position (x, y) and velocity (vx, vy).

        Uses offscreen camera detector if use_camera is enabled; otherwise uses
        simulator ground-truth physics state.
        """
        q, v = self.puck_qpos, self.puck_dof
        if ground_truth:
            return (
                -0.5 + float(self.data.qpos[q]),
                float(self.data.qpos[q + 1]),
                float(self.data.qvel[v]),
                float(self.data.qvel[v + 1]),
            )
        if self.use_camera and self.detector is not None:
            puck_x, puck_y, puck_vx, puck_vy, self.last_camera_frame, _ = (
                self.detector.detect(self.data)
            )
        else:
            puck_x = -0.5 + float(self.data.qpos[q])
            puck_y = float(self.data.qpos[q + 1])
            puck_vx = float(self.data.qvel[v])
            puck_vy = float(self.data.qvel[v + 1])
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
        self.set_velocity_target(self.arms[0], action)
        self.arms[0].drive(self.kin, self.data)
        self.simulate()
        return self.arms[0].contact

    def set_velocity_target(self, arm: PaddleArm, action: np.ndarray):
        """Integrate a [v_x, v_y] command into the arm's paddle target (local frame)."""
        dx = float(action[0] * self.max_vel_x * self.dt_ctrl)
        dy = float(action[1] * self.max_vel_y * self.dt_ctrl)
        arm.target[0] = np.clip(arm.target[0] + dx, self.x_bounds[0], self.x_bounds[1])
        arm.target[1] = np.clip(arm.target[1] + dy, self.y_bounds[0], self.y_bounds[1])
        arm.target[2] = self.z_plane

    def simulate(self):
        """Substep physics with gravity compensation and record puck contacts."""
        for arm in self.arms:
            arm.contact = False
        self.wall_contact_detected = False
        for _ in range(self.frame_skip):
            for arm in self.arms:
                self.data.qfrc_applied[arm.dof_idx] = self.data.qfrc_bias[arm.dof_idx]
            mujoco.mj_step(self.model, self.data)

            for c in range(self.data.ncon):
                con = self.data.contact[c]
                if con.geom1 == self.puck_geom_id:
                    other = con.geom2
                elif con.geom2 == self.puck_geom_id:
                    other = con.geom1
                else:
                    continue
                if other in self.back_wall_geom_ids:
                    self.wall_contact_detected = True
                for arm in self.arms:
                    if other == arm.paddle_geom_id:
                        arm.contact = True

        for arm in self.arms:
            arm.update_velocity(self.data, self.dt_ctrl)

    def arm_obs(self, arm: PaddleArm, puck: Tuple[float, float, float, float]) -> np.ndarray:
        """23-dim single-arm observation in the arm's local frame (unclipped)."""
        pad_x, pad_y, _, _ = arm.to_local(*arm.paddle_world(self.data))
        pad_vx, pad_vy = arm.paddle_vel
        puck_x, puck_y, puck_vx, puck_vy = arm.to_local(*puck)
        rel_x = puck_x - pad_x
        rel_y = puck_y - pad_y
        return np.concatenate(
            [
                [pad_x, pad_y, pad_vx, pad_vy],
                [puck_x, puck_y, puck_vx, puck_vy],
                [rel_x, rel_y, np.hypot(rel_x, rel_y)],
                self.data.qpos[arm.qpos_idx],
                self.data.qvel[arm.dof_idx],
            ]
        ).astype(np.float32)

    def reset_robot_ready(self):
        """Reset robot joint positions and controllers to nominal ready keyframe."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.key("ready").qpos
        self.data.ctrl[:] = self.model.key("ready").ctrl

        mujoco.mj_forward(self.model, self.data)
        for arm in self.arms:
            arm.reset(self.data, [self.base_xe, 0.0, self.z_plane])

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
                self.viewer.cam.distance = self.viewer_distance
                self.viewer.cam.elevation = -40.0
                self.viewer.cam.azimuth = 90.0
                self.viewer.cam.lookat = self.viewer_lookat
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
