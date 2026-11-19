"""Modular Reward Functions and Weight Configurations for UR5e Air Hockey Tasks.

Defines the abstract BaseRewardFunction interface and concrete implementations:
- ReboundRewardFunction: Multi-objective reward for returning/rebounding high-speed pucks.
- GoalScoringRewardFunction: Reward for driving puck toward opponent's goal or target.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseRewardFunction(ABC):
    """Abstract base class for modular reward functions."""

    @abstractmethod
    def compute(
        self,
        obs: np.ndarray,
        action: Optional[np.ndarray],
        info: Dict[str, Any],
    ) -> float:
        """Compute scalar reward given the observation, action, and step info dict.

        Args:
            obs: State observation array.
            action: Continuous action array.
            info: Step dictionary containing collision flags, positions, and velocities.

        Returns:
            Scalar reward float.
        """
        raise NotImplementedError


@dataclass
class ReboundRewardWeights:
    """Configurable reward weights and thresholds for rebound defense tasks.

    Attributes:
        w_align: Reward scale for predictive intercept alignment (default: 2.0).
        sigma_align: Width of intercept alignment tolerance in meters (default: 0.08).
        w_strike: Bonus scale for forward striking based on time-to-impact (default: 1.5).
        target_tti: Target time-to-impact in seconds for strike bonus (default: 0.12).
        sigma_tti: Width of time-to-impact strike window in seconds (default: 0.05).
        w_center: Reward scale for returning to table center when puck departs (default: 0.5).
        sigma_center: Width of centering tolerance in meters (default: 0.20).
        w_rebound: Base bonus on successful forward rebound strike (default: 35.0).
        w_speed: Velocity bonus scale on outgoing puck Vx (default: 3.5).
        w_deflect: Penalty scale on outgoing lateral velocity |Vy| (default: 1.0).
        w_goal_conceded: Penalty when puck crosses defense goal line (default: 25.0).
        w_boundary: Penalty when puck exits arena perimeter (default: 10.0).
        w_action_reg: L2 regularization penalty on action magnitude (default: 0.003).

        puck_approach_vx: Vx threshold below which puck is treated as approaching (default: -0.2 m/s).
        min_rebound_vx: Minimum outgoing puck Vx required to qualify as a rebound (default: 0.3 m/s).
        max_rebound_vel: Maximum puck Vx considered for velocity bonus cap (default: 8.0 m/s).
    """

    # 1. Predictive intercept tracking (approaching puck)
    w_align: float = 2.0
    sigma_align: float = 0.08

    # 2. Dynamic time-to-impact striking
    w_strike: float = 1.5
    target_tti: float = 0.12
    sigma_tti: float = 0.05

    # 3. Defense & centering (departing puck)
    w_center: float = 0.5
    sigma_center: float = 0.20

    # 4. Rebound strike quality
    w_rebound: float = 35.0
    w_speed: float = 3.5
    w_deflect: float = 1.0

    # 5. Failure & boundary penalties
    w_goal_conceded: float = 25.0
    w_boundary: float = 10.0

    # 6. Control regularization
    w_action_reg: float = 0.003

    # Thresholds & limits
    puck_approach_vx: float = -0.2
    min_rebound_vx: float = 0.3
    max_rebound_vel: float = 8.0

    def to_dict(self) -> Dict[str, float]:
        """Convert weights to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReboundRewardWeights":
        """Construct ReboundRewardWeights from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: float(v) for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# Backward-compatibility alias
RewardWeights = ReboundRewardWeights


class ReboundRewardFunction(BaseRewardFunction):
    """Modular reward calculator for air hockey rebound tasks."""

    def __init__(
        self,
        weights: Optional[ReboundRewardWeights] = None,
        base_xe: float = -1.75,
        side_walls: bool = True,
    ):
        self.weights = weights if weights is not None else ReboundRewardWeights()
        self.base_xe = base_xe
        self.side_walls = side_walls

    def compute(
        self,
        obs: np.ndarray,
        action: Optional[np.ndarray],
        info: Dict[str, Any],
    ) -> float:
        """Compute scalar reward for puck interception and rebound."""
        from controllers.classical_ur5e_controller import predict_intercept_y

        w = self.weights
        pad_x, pad_y = float(obs[0]), float(obs[1])
        pad_vx = float(obs[2])
        puck_x, puck_y = float(obs[4]), float(obs[5])
        puck_vx, puck_vy = float(obs[6]), float(obs[7])

        contact_detected = bool(info.get("contact_detected", False))
        side_walls = bool(info.get("side_walls", self.side_walls))

        reward = 0.0

        # 1. Puck approaching (-X): Predictive interception & dynamic TTI striking
        if puck_vx < w.puck_approach_vx:
            y_target = predict_intercept_y(
                puck_x,
                puck_y,
                puck_vx,
                puck_vy,
                paddle_x=self.base_xe,
                side_walls=side_walls,
            )
            # Smooth Gaussian alignment with predicted intercept
            reward += w.w_align * float(
                np.exp(-((pad_y - y_target) ** 2) / (2.0 * (w.sigma_align**2)))
            )

            # Dynamic Time-To-Impact (TTI) forward striking bonus
            tti = (self.base_xe - puck_x) / puck_vx
            if 0.0 < tti < 0.25 and pad_vx > 0.0:
                tti_bell = float(
                    np.exp(-((tti - w.target_tti) ** 2) / (2.0 * (w.sigma_tti**2)))
                )
                reward += w.w_strike * pad_vx * tti_bell

        # 2. Puck departing (+X): Smooth return to center
        else:
            reward += w.w_center * float(
                np.exp(-(pad_y**2) / (2.0 * (w.sigma_center**2)))
            )

        # 3. Rebound bonus: puck was struck and is moving forward (+X)
        if contact_detected and puck_vx > w.min_rebound_vx:
            speed_bonus = w.w_speed * min(puck_vx, w.max_rebound_vel)
            deflect_pen = w.w_deflect * abs(puck_vy)
            reward += w.w_rebound + speed_bonus - deflect_pen

        # 4. Termination penalties
        if puck_x < -1.95:
            # Goal conceded (missed puck)
            reward -= w.w_goal_conceded
        elif (side_walls and abs(puck_y) > 0.95) or (
            not side_walls and abs(puck_y) > 1.45
        ):
            reward -= w.w_boundary

        # 5. Action regularization (energy penalty)
        if action is not None:
            reward -= w.w_action_reg * float(np.sum(action**2))

        return float(reward)


class GoalScoringRewardFunction(BaseRewardFunction):
    """Modular reward calculator for scoring goals into an opponent target."""

    def __init__(
        self,
        target_pos: Tuple[float, float] = (1.85, 0.0),
        goal_reward: float = 100.0,
        dist_scale: float = 2.0,
        velocity_scale: float = 5.0,
        action_reg: float = 0.003,
    ):
        self.target_pos = target_pos
        self.goal_reward = goal_reward
        self.dist_scale = dist_scale
        self.velocity_scale = velocity_scale
        self.action_reg = action_reg

    def compute(
        self,
        obs: np.ndarray,
        action: Optional[np.ndarray],
        info: Dict[str, Any],
    ) -> float:
        """Compute reward for driving puck toward opponent's goal."""
        puck_x, puck_y = float(obs[4]), float(obs[5])
        puck_vx, puck_vy = float(obs[6]), float(obs[7])

        dist_to_goal = float(
            np.hypot(self.target_pos[0] - puck_x, self.target_pos[1] - puck_y)
        )
        reward = -self.dist_scale * dist_to_goal

        # Bonus for forward velocity toward target
        if puck_vx > 0:
            reward += self.velocity_scale * puck_vx

        # Goal scored condition (reached goal line within goal width)
        if puck_x >= self.target_pos[0] and abs(puck_y - self.target_pos[1]) < 0.25:
            reward += self.goal_reward

        if action is not None:
            reward -= self.action_reg * float(np.sum(action**2))

        return float(reward)
