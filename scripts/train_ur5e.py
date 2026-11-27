#!/usr/bin/env python3
"""Train Stable-Baselines3 PPO Agent for 6-DOF UR5e Air Hockey Rebound.

Uses vectorized multiprocessing (SubprocVecEnv) to train a continuous-action
PPO policy on the UR5e manipulator environment.

Usage (from RoboPong/ directory):
    python3 scripts/train_ur5e.py --timesteps 1000000 --num_envs 8
    python3 scripts/train_ur5e.py --config configs/ppo_config.yaml
    python3 scripts/train_ur5e.py --reward dense
    python3 scripts/train_ur5e.py --env two_arm --reward sparse
    python3 scripts/train_ur5e.py --env two_arm --reward sparse --until_converged \\
        --resume saved_models/ur5e_ppo_two_arm_sparse/ppo_ur5e_final.zip
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Ensure project root is on python import path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from envs.two_arm_ur5e_env import TwoArmHockeyUR5eEnv
from envs.hockey_ur5e_env import HockeyUR5eEnv
from envs.rewards import REWARD_TYPES


class ConvergenceCallback(BaseCallback):
    """Stop once episode length and hits improve by less than `tol` over `window` blocks."""

    def __init__(self, block_steps: int = 1_000_000, window: int = 3, tol: float = 0.03):
        super().__init__()
        self.block_steps, self.window, self.tol = block_steps, window, tol
        self.blocks = []
        self.lengths, self.hits = [], []
        self.block_start = None

    def _on_step(self) -> bool:
        if self.block_start is None:
            self.block_start = self.num_timesteps
        for info in self.locals["infos"]:
            if "episode" in info:
                self.lengths.append(info["episode"]["l"])
                self.hits.append(info.get("wall_hits", 0))
        if self.num_timesteps - self.block_start < self.block_steps or not self.lengths:
            return True

        self.blocks.append((np.mean(self.lengths), np.mean(self.hits)))
        self.lengths, self.hits = [], []
        self.block_start = self.num_timesteps
        length, hits = self.blocks[-1]
        print(f"[convergence] {self.num_timesteps:,} steps: ep length {length:.0f}, hits/ep {hits:.2f}", flush=True)

        w = self.window
        if len(self.blocks) < 2 * w:
            return True
        recent = np.mean(self.blocks[-w:], axis=0)
        previous = np.mean(self.blocks[-2 * w : -w], axis=0)
        if np.all(recent <= previous * (1 + self.tol)):
            print(
                f"[convergence] no improvement over the last {w} blocks "
                f"(length {previous[0]:.0f} -> {recent[0]:.0f}, hits {previous[1]:.2f} -> {recent[1]:.2f}), stopping",
                flush=True,
            )
            return False
        return True


def load_config(config_path: str = "") -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to configuration file. If omitted, checks default YAML configs.

    Returns:
        Configuration dictionary.
    """
    if not config_path:
        default_yaml = os.path.join(_project_root, "configs", "ppo_config.yaml")
        config_path = default_yaml

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def reward_params_for(cfg: dict, reward_type: str) -> dict:
    """Reward weights for reward_type from the config."""
    params = dict(cfg.get("reward", {}).get(reward_type) or {})
    if reward_type == "dense":
        # Dense shaping needs the same discount as PPO
        params["gamma"] = cfg.get("ppo", {}).get("gamma", 0.99)
    return params


def default_model_dir(reward_type: str, env_type: str = "single") -> str:
    """Where models for this reward and env are saved by default."""
    if env_type == "two_arm":
        name = f"ur5e_ppo_two_arm_{reward_type}"
    else:
        # Rebound keeps the original directory name
        name = "ur5e_ppo" if reward_type == "rebound" else f"ur5e_ppo_{reward_type}"
    return os.path.join(_project_root, "saved_models", name)


def main(args: argparse.Namespace):
    """Train UR5e air hockey rebound policy using Stable-Baselines3 PPO.

    Args:
        args: Parsed command-line arguments specifying hyperparameters and training options.
    """
    cfg = load_config(args.config)
    cfg_env = cfg.get("env", {})
    cfg_ppo = cfg.get("ppo", {})
    cfg_train = cfg.get("training", {})
    cfg_reward = cfg.get("reward", {})

    # CLI args override YAML config
    side_walls_val = (
        args.side_walls.lower() in ("true", "1", "yes")
        if args.side_walls is not None
        else cfg_env.get("side_walls", True)
    )
    total_timesteps = args.timesteps if args.timesteps is not None else cfg_train.get("total_timesteps", 1000000)
    num_envs = args.num_envs if args.num_envs is not None else cfg_train.get("num_envs", 8)
    lr = args.lr if args.lr is not None else cfg_ppo.get("learning_rate", 3e-4)
    seed = args.seed if args.seed is not None else cfg_train.get("seed", 42)
    reward_type = args.reward or cfg_reward.get("type", "rebound")
    env_type = args.env or cfg_env.get("type", "single")
    reward_params = reward_params_for(cfg, reward_type)

    save_dir = args.save_dir or default_model_dir(reward_type, env_type)
    log_dir = os.path.join(save_dir, "logs")
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    best_model_dir = os.path.join(save_dir, "best_model")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    print("=" * 68)
    print("6-DOF UR5e Manipulator RL Training with Stable-Baselines3 (PPO)")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Envs:   {num_envs}")
    print(
        f"Side Walls:      {'ENABLED (rebound bank shots)' if side_walls_val else 'DISABLED (open court)'}"
    )
    print(f"Environment:     {env_type}")
    print(f"Reward:          {reward_type} {reward_params or ''}")
    print(f"Learning Rate:   {lr}")
    print(f"Save Directory:  {save_dir}")
    print("=" * 68)

    if env_type == "two_arm":
        env_cls = TwoArmHockeyUR5eEnv
        env_kwargs = dict(reward_type=reward_type, reward_params=reward_params)
    else:
        env_cls = HockeyUR5eEnv
        env_kwargs = dict(
            side_walls=side_walls_val,
            use_camera=cfg_env.get("use_camera", False),
            reward_type=reward_type,
            reward_params=reward_params,
        )
    # Logged for every reward so runs can be compared
    monitor_keys = ("rebounds", "wall_hits")

    # 1. Vectorized parallel training environments
    vec_env = make_vec_env(
        env_cls,
        n_envs=num_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if num_envs > 1 else None,
        # fork lets workers share the parent's loaded libraries, which saves RAM
        vec_env_kwargs=dict(start_method="fork") if num_envs > 1 else None,
    )
    # A resumed run logs to new files so the original history is kept
    suffix = "_resumed" if args.resume else ""
    vec_env = VecMonitor(
        vec_env, filename=os.path.join(log_dir, f"train{suffix}_monitor.csv"), info_keywords=monitor_keys
    )

    # 2. Evaluation environment
    eval_env = make_vec_env(
        env_cls,
        n_envs=1,
        seed=seed + 100,
        env_kwargs=env_kwargs,
    )
    eval_env = VecMonitor(
        eval_env, filename=os.path.join(log_dir, f"eval{suffix}_monitor.csv"), info_keywords=monitor_keys
    )

    # 3. Callbacks: periodic evaluation and model checkpoints
    eval_freq = cfg_train.get("eval_freq", max(25000 // num_envs, 2000))
    eval_log_dir = os.path.join(log_dir, "resumed") if args.resume else log_dir
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=15,
        deterministic=True,
        render=False,
    )
    previous_evals = os.path.join(log_dir, "evaluations.npz")
    if args.resume and os.path.exists(previous_evals):
        # Only replace best_model.zip if the resumed run beats the earlier best
        eval_callback.best_mean_reward = float(np.load(previous_evals)["results"].mean(axis=1).max())

    save_freq = cfg_train.get("save_freq", max(100000 // num_envs, 10000))
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_ur5e",
    )

    # 4. Neural Network Policy Architecture
    policy_kwargs = dict(
        net_arch=cfg_ppo.get("net_arch", dict(pi=[256, 256], vf=[256, 256])),
    )

    # 5. PPO Model Configuration
    callbacks = [eval_callback, checkpoint_callback]
    if args.until_converged:
        callbacks.append(ConvergenceCallback())
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr,
        n_steps=cfg_ppo.get("n_steps", 2048),
        batch_size=cfg_ppo.get("batch_size", 128),
        n_epochs=cfg_ppo.get("n_epochs", 10),
        gamma=cfg_ppo.get("gamma", 0.99),
        gae_lambda=cfg_ppo.get("gae_lambda", 0.95),
        clip_range=cfg_ppo.get("clip_range", 0.2),
        ent_coef=cfg_ppo.get("ent_coef", 0.005),
        target_kl=cfg_ppo.get("target_kl"),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
    )
    if args.resume:
        model = PPO.load(args.resume, env=vec_env)
        print(f"Resuming from {args.resume} at {model.num_timesteps:,} steps")

    start_time = time.time()
    print(
        f"\nStarting {total_timesteps:,} steps training run across {num_envs} parallel environments..."
    )
    steps_before = model.num_timesteps
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=not args.resume,
    )
    elapsed = time.time() - start_time

    # 6. Save final model
    final_model_path = os.path.join(save_dir, "ppo_ur5e_final.zip")
    model.save(final_model_path)

    print("=" * 68)
    print(f"Training Complete in {elapsed:.1f}s ({elapsed/60.0:.2f} mins)!")
    print(f"Total Timesteps:    {model.num_timesteps:,}")
    print(f"Average Throughput: {(model.num_timesteps - steps_before) / elapsed:.0f} steps/second")
    print(f"Final Model Saved:  {final_model_path}")
    print(f"Best Model Saved:   {os.path.join(best_model_dir, 'best_model.zip')}")
    print("=" * 68)

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO Agent for UR5e Air Hockey Rebound"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML configuration file (default: configs/ppo_config.yaml)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config if provided)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel worker environments (overrides config if provided)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="PPO learning rate (overrides config if provided)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config if provided)",
    )
    parser.add_argument(
        "--side_walls",
        type=str,
        default=None,
        help="Include lateral side rebound walls (true/false, overrides config if provided)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["single", "two_arm"],
        default=None,
        help="single-arm rebound task or two-arm task (overrides config env.type)",
    )
    parser.add_argument(
        "--reward",
        type=str,
        choices=REWARD_TYPES,
        default=None,
        help="Reward function (overrides config reward.type if provided)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Continue training from this saved model (--timesteps is then the number of extra steps)",
    )
    parser.add_argument(
        "--until_converged",
        action="store_true",
        help="Stop early once episode length and hits stop improving (--timesteps is the cap)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory to save checkpoints, logs, and final model",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
