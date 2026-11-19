"""16-709 UR5e Manipulator Air Hockey Simulation and Reinforcement Learning Package."""

import os
import sys

# Ensure package directory is available
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from controllers.classical_ur5e_controller import predict_intercept_y
from controllers.ur5e_kinematics import UR5eKinematics
from envs.base_ur5e_env import BaseUR5eEnv
from envs.hockey_ur5e_env import HockeyUR5eEnv
from envs.rewards import (
    BaseRewardFunction,
    GoalScoringRewardFunction,
    ReboundRewardFunction,
    ReboundRewardWeights,
    RewardWeights,
)
from vision.puck_detector import PuckColorDetector

__all__ = [
    "BaseUR5eEnv",
    "HockeyUR5eEnv",
    "BaseRewardFunction",
    "ReboundRewardWeights",
    "RewardWeights",
    "ReboundRewardFunction",
    "GoalScoringRewardFunction",
    "UR5eKinematics",
    "predict_intercept_y",
    "PuckColorDetector",
]
