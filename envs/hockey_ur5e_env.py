"""Gymnasium Environment for 6-DOF UR5e Manipulator Air Hockey Rebound in MuJoCo.

Observation Space (23 dims):
    [paddle_x, paddle_y, paddle_vx, paddle_vy,
     puck_x, puck_y, puck_vx, puck_vy,
     rel_x, rel_y, distance,
     q_1..q_6 (6 joint positions),
     qdot_1..qdot_6 (6 joint velocities)]

Action Space (2 dims, continuous in [-1.0, 1.0]):
    [vx_cmd, vy_cmd] -> mapped to Cartesian paddle displacement on the table (Z = 0.035 m)
    and resolved to 6-DOF joint angles via DLS Inverse Kinematics.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

# Ensure project root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from envs.base_ur5e_env import BaseUR5eEnv
from envs.rewards import (
    BaseRewardFunction,
    ReboundRewardFunction,
    ReboundRewardWeights,
)


class HockeyUR5eEnv(BaseUR5eEnv):
    """Air hockey table environment where a 6-DOF UR5e robot rebounds an elastic puck.

    Inherits core physics, DLS kinematics, action execution, and perception
    from BaseUR5eEnv. Uses modular reward functions (defaults to ReboundRewardFunction).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        side_walls: bool = True,
        use_camera: bool = False,
        show_camera: bool = False,
        frame_skip: int = 20,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
        reward_weights: Optional[ReboundRewardWeights] = None,
        reward_function: Optional[BaseRewardFunction] = None,
    ):
        super().__init__(
            model_path=model_path,
            side_walls=side_walls,
            use_camera=use_camera,
            show_camera=show_camera,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        )

        # Configurable reward weights & modular reward function
        self.reward_weights = (
            reward_weights if reward_weights is not None else ReboundRewardWeights()
        )
        if reward_function is not None:
            self.reward_fn = reward_function
        else:
            self.reward_fn = ReboundRewardFunction(
                weights=self.reward_weights,
                base_xe=self.base_xe,
                side_walls=self.side_walls,
            )

        # Observation space: 23 dimensions
        obs_low = np.array(
            [
                -2.2, -1.0, -10.0, -10.0,  # paddle x, y, vx, vy
                -2.5, -1.5, -20.0, -20.0,  # puck x, y, vx, vy
                -4.0, -2.5, 0.0,           # rel_x, rel_y, distance
                -6.3, -6.3, -3.15, -6.3, -6.3, -6.3,  # q_1..q_6
                -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,  # qdot_1..qdot_6
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                -1.2, 1.0, 10.0, 10.0,
                2.5, 1.5, 20.0, 20.0,
                4.0, 2.5, 5.0,
                6.3, 6.3, 3.15, 6.3, 6.3, 6.3,
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Rebound task specific metrics
        self.rebounds = 0
        self.last_contact_time = -1.0
        self.prev_puck_vx = 0.0

    def _get_obs(self) -> np.ndarray:
        """Construct the 23-dimensional state observation."""
        paddle_pos, paddle_vel = self.get_paddle_state()
        pad_x, pad_y = paddle_pos[0], paddle_pos[1]
        pad_vx, pad_vy = paddle_vel[0], paddle_vel[1]

        puck_x, puck_y, puck_vx, puck_vy = self.get_puck_state()

        rel_x = puck_x - pad_x
        rel_y = puck_y - pad_y
        dist = np.hypot(rel_x, rel_y)

        q_joints = self.data.qpos[:6]
        qdot_joints = self.data.qvel[:6]

        obs = np.concatenate(
            [
                [pad_x, pad_y, pad_vx, pad_vy],
                [puck_x, puck_y, puck_vx, puck_vy],
                [rel_x, rel_y, dist],
                q_joints,
                qdot_joints,
            ]
        ).astype(np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.step_count = 0
        self.rebounds = 0
        self.last_contact_time = -1.0

        # Reset UR5e to ready keyframe
        self.reset_robot_ready()

        # Randomize initial puck state
        # 50% start near paddle heading toward front wall (+X)
        # 50% start returning from front wall (-X)
        if self.np_random.random() < 0.5:
            self.data.qpos[6] = self.np_random.uniform(-0.3, 0.2)  # x in [-0.8, -0.3]
            self.data.qpos[7] = self.np_random.uniform(-0.35, 0.35)
            self.data.qvel[6] = self.np_random.uniform(2.8, 4.8)
            self.data.qvel[7] = self.np_random.uniform(-0.8, 0.8)
        else:
            self.data.qpos[6] = self.np_random.uniform(1.0, 2.0)  # x in [0.5, 1.5]
            self.data.qpos[7] = self.np_random.uniform(-0.5, 0.5)
            self.data.qvel[6] = self.np_random.uniform(-4.8, -2.8)
            self.data.qvel[7] = self.np_random.uniform(-0.8, 0.8)

        self.data.qpos[8] = 0.0
        self.data.qvel[8] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.prev_puck_vx = self.data.qvel[6]
        self.prev_paddle_pos = self.data.site_xpos[self.site_id][:2].copy()
        self.paddle_vel = np.zeros(2, dtype=np.float64)

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        # 1. Apply action and substep physics via BaseUR5eEnv
        contact_detected = self.apply_action(action)

        # 2. Extract state observation
        obs = self._get_obs()
        pad_x, pad_y = obs[0], obs[1]
        puck_x, puck_y = obs[4], obs[5]
        puck_vx, puck_vy = obs[6], obs[7]

        # 3. Rebound tracking (debounced collision check)
        rebound_occurred = False
        if contact_detected and (self.data.time - self.last_contact_time > 0.18):
            if puck_vx > 0.3:
                self.rebounds += 1
                self.last_contact_time = self.data.time
                rebound_occurred = True

        # 4. Check termination conditions
        terminated = False
        truncated = False
        if puck_x < -1.95:
            # Goal conceded (missed puck)
            terminated = True
        elif not self.side_walls and abs(puck_y) > 1.45:
            # Puck exited open lateral perimeter
            terminated = True
        elif self.side_walls and abs(puck_y) > 0.95:
            # Puck clipped through lateral wall
            terminated = True

        # Truncation on maximum episode steps
        if self.step_count >= self.max_episode_steps:
            truncated = True

        # 5. Compute reward via modular reward function
        reward_info = {
            "contact_detected": rebound_occurred,
            "side_walls": self.side_walls,
            "rebounds": self.rebounds,
            "step_count": self.step_count,
        }
        reward = self.get_reward(
            obs=obs,
            action=action,
            contact_detected=rebound_occurred,
            info=reward_info,
        )

        self.prev_puck_vx = puck_vx

        info = {
            "rebounds": self.rebounds,
            "step_count": self.step_count,
            "puck_pos": (float(puck_x), float(puck_y)),
            "paddle_pos": (float(pad_x), float(pad_y)),
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def get_reward(
        self,
        obs: np.ndarray,
        action: Optional[np.ndarray] = None,
        contact_detected: bool = False,
        weights: Optional[ReboundRewardWeights] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute the scalar reward using the modular reward function."""
        if info is None:
            info = {
                "contact_detected": contact_detected,
                "side_walls": self.side_walls,
                "rebounds": self.rebounds,
                "step_count": self.step_count,
            }
        else:
            info["contact_detected"] = contact_detected

        if weights is not None:
            # Temporary override weights if caller passed custom weights
            temp_fn = ReboundRewardFunction(
                weights=weights, base_xe=self.base_xe, side_walls=self.side_walls
            )
            return temp_fn.compute(obs=obs, action=action, info=info)

        return self.reward_fn.compute(obs=obs, action=action, info=info)
