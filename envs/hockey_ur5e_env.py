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

from envs.base_ur5e_env import BaseUR5eEnv, HitCounter
from envs.rewards import (
    BaseRewardFunction,
    ReboundRewardFunction,
    ReboundRewardWeights,
    make_reward_function,
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
        reward_type: str = "rebound",
        reward_params: Optional[Dict[str, Any]] = None,
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
        elif reward_type == "rebound" and not reward_params:
            self.reward_fn = ReboundRewardFunction(
                weights=self.reward_weights,
                base_xe=self.base_xe,
                side_walls=self.side_walls,
            )
        else:
            self.reward_fn = make_reward_function(
                reward_type,
                reward_params,
                base_xe=self.base_xe,
                side_walls=self.side_walls,
                back_wall_x=self.back_wall_x,
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
        self.rebound_counter = HitCounter(min_gap=0.18, rising_edge=False)
        self.wall_hit_counter = HitCounter(min_gap=0.1, rising_edge=True)
        self.prev_puck_vx = 0.0

    @property
    def rebounds(self) -> int:
        return self.rebound_counter.count

    @property
    def wall_hits(self) -> int:
        return self.wall_hit_counter.count

    def _get_obs(self, ground_truth: bool = False) -> np.ndarray:
        """Construct the 23-dimensional state observation."""
        obs = self.arm_obs(self.arms[0], self.get_puck_state(ground_truth=ground_truth))
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.step_count = 0
        self.rebound_counter.reset()
        self.wall_hit_counter.reset()

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

        obs = self._get_obs()
        self.reward_fn.reset(self._get_obs(ground_truth=True) if self.use_camera else obs)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        # 1. Apply action and substep physics via BaseUR5eEnv
        contact_detected = self.apply_action(action)

        # 2. Extract state observation
        obs = self._get_obs()
        # Reward and termination use the true puck state, even when the policy sees the camera estimate
        gt_obs = self._get_obs(ground_truth=True) if self.use_camera else obs
        pad_x, pad_y = gt_obs[0], gt_obs[1]
        puck_x, puck_y = gt_obs[4], gt_obs[5]
        puck_vx, puck_vy = gt_obs[6], gt_obs[7]

        # 3. Rebound and wall-hit tracking (debounced collision checks)
        rebound_occurred = self.rebound_counter.update(contact_detected, self.data.time, valid=puck_vx > 0.3)
        wall_hit = self.wall_hit_counter.update(self.wall_contact_detected, self.data.time)

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
            "wall_hit": wall_hit,
            "wall_hits": self.wall_hits,
            "terminated": terminated,
        }
        reward = self.get_reward(
            obs=gt_obs,
            action=action,
            contact_detected=rebound_occurred,
            info=reward_info,
        )

        self.prev_puck_vx = puck_vx

        info = {
            "rebounds": self.rebounds,
            "wall_hits": self.wall_hits,
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
