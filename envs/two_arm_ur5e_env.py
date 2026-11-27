"""Gymnasium Environment for Two UR5e Arms Cooperatively Keeping a Puck in Play.

The table is the single-arm table with arm B where the back wall was, mirrored about
the table centre (x = 0). Each arm works in its own local frame, which matches the
single-arm frame exactly: its paddle sits at x = -1.75 and its partner's paddle takes
the place of the back wall.

Observation Space (46 dims):
    Arm A's 23-dim single-arm observation followed by arm B's, each in its own frame.

Action Space (4 dims, continuous in [-1.0, 1.0]):
    [vx_a, vy_a, vx_b, vy_b], each pair in that arm's local frame.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

# Ensure project root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from envs.base_ur5e_env import BaseUR5eEnv, HitCounter, PaddleArm
from envs.rewards import make_reward_function

TWO_ARM_MODEL = os.path.join(_project_root, "models", "ur5e", "hockey_ur5e_two_arm_world.xml")
PADDLE_RADIUS = 0.08


class TwoArmHockeyUR5eEnv(BaseUR5eEnv):
    """Two UR5e arms passing a puck back and forth on an air hockey table."""

    def __init__(
        self,
        frame_skip: int = 20,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        reward_type: str = "rebound",
        reward_params: Optional[Dict[str, Any]] = None,
        render_size: Tuple[int, int] = (720, 1280),
        render_camera: str = "side_view",
    ):
        super().__init__(
            model_path=TWO_ARM_MODEL,
            side_walls=True,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            arm_prefixes=("a_", "b_"),
            mirror_x=0.0,
        )
        self.render_size = render_size
        self.render_camera = render_camera
        self.renderer = None
        self.viewer_distance, self.viewer_lookat = 5.0, [0.0, 0.0, 0.0]

        # The partner's paddle face plays the role of the single-arm back wall
        partner_face_x = 2.0 * self.arms[1].mirror_x - self.base_xe - PADDLE_RADIUS
        self.reward_fns = [
            make_reward_function(
                reward_type, reward_params, base_xe=self.base_xe, side_walls=True, back_wall_x=partner_face_x
            )
            for _ in self.arms
        ]
        # Per arm: strikes it makes, and times the puck reaches its paddle
        self.rebound_counters = [HitCounter(min_gap=0.18, rising_edge=False) for _ in self.arms]
        self.hit_counters = [HitCounter(min_gap=0.1, rising_edge=True) for _ in self.arms]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(46,), dtype=np.float32)

    def puck_local(self, arm: PaddleArm) -> Tuple[float, float, float, float]:
        """Ground-truth puck position and velocity in the given arm's frame."""
        return arm.to_local(*self.get_puck_state(ground_truth=True))

    def local_obs(self, arm: PaddleArm) -> np.ndarray:
        """23-dim single-arm observation for one arm in its own frame (unclipped)."""
        return self.arm_obs(arm, self.get_puck_state(ground_truth=True))

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.local_obs(arm) for arm in self.arms])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.reset_robot_ready()
        for counter in self.rebound_counters + self.hit_counters:
            counter.reset()

        # Launch the puck at a random arm, matching the single-arm env's incoming puck
        receiver = self.arms[self.np_random.integers(2)]
        x_l = self.np_random.uniform(0.5, 1.5)
        y_l = self.np_random.uniform(-0.5, 0.5)
        vx_l = self.np_random.uniform(-4.8, -2.8)
        vy_l = self.np_random.uniform(-0.8, 0.8)
        x, y, vx, vy = receiver.to_local(x_l, y_l, vx_l, vy_l)  # mirroring is its own inverse
        self.data.qpos[self.puck_qpos : self.puck_qpos + 3] = [x + 0.5, y, 0.0]
        self.data.qvel[self.puck_dof : self.puck_dof + 3] = [vx, vy, 0.0]
        mujoco.mj_forward(self.model, self.data)

        for arm, fn in zip(self.arms, self.reward_fns):
            fn.reset(self.local_obs(arm))
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Step with per-arm velocity commands, as in the single-arm env."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for i, arm in enumerate(self.arms):
            self.set_velocity_target(arm, action[2 * i : 2 * i + 2])
        return self._advance(action)

    def step_targets(self, targets: List[np.ndarray]):
        """Step with a paddle target per arm (local frame), used by the classical controller."""
        for arm, target in zip(self.arms, targets):
            arm.target = np.array([target[0], target[1], self.z_plane], dtype=np.float64)
        return self._advance(None)

    def _advance(self, action: Optional[np.ndarray]):
        self.step_count += 1
        for arm in self.arms:
            arm.drive(self.kin, self.data)
        self.simulate()

        rebound, hit = [], []
        for arm, rc, hc in zip(self.arms, self.rebound_counters, self.hit_counters):
            puck_vx = self.puck_local(arm)[2]
            rebound.append(rc.update(arm.contact, self.data.time, valid=puck_vx > 0.3))
            hit.append(hc.update(arm.contact, self.data.time))

        puck_x, puck_y, _, _ = self.get_puck_state(ground_truth=True)
        mirror_x = self.arms[1].mirror_x
        missed = [puck_x < -1.95, puck_x > 2.0 * mirror_x + 1.95]
        out_of_bounds = abs(puck_y) > 0.95
        terminated = any(missed) or out_of_bounds
        truncated = self.step_count >= self.max_episode_steps

        # Each arm gets the single-arm reward with its partner's paddle as the back wall
        reward = 0.0
        for i, (arm, fn) in enumerate(zip(self.arms, self.reward_fns)):
            info_i = {
                "contact_detected": rebound[i],
                "side_walls": True,
                "rebounds": self.rebound_counters[i].count,
                "step_count": self.step_count,
                "wall_hit": hit[1 - i],
                "wall_hits": self.hit_counters[1 - i].count,
                "terminated": terminated,
                # The team pays the miss penalty once per lost puck, as in the single-arm env
                "missed": missed[i] or out_of_bounds,
            }
            a_i = None if action is None else action[2 * i : 2 * i + 2]
            reward += fn.compute(obs=self.local_obs(arm), action=a_i, info=info_i)

        info = {
            "rebounds": sum(c.count for c in self.rebound_counters),
            "wall_hits": sum(c.count for c in self.hit_counters),
            "rebounds_a": self.rebound_counters[0].count,
            "rebounds_b": self.rebound_counters[1].count,
            "step_count": self.step_count,
            "missed_by": "a" if missed[0] else "b" if missed[1] else None,
        }

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return super().render()
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, *self.render_size)
        self.renderer.update_scene(self.data, camera=self.render_camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        super().close()
