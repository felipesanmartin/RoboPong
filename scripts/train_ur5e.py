#!/usr/bin/env python3
"""Train Stable-Baselines3 PPO Agent for 6-DOF UR5e Air Hockey Rebound.

Uses vectorized multiprocessing (SubprocVecEnv) to train a continuous-action
PPO policy on the UR5e manipulator environment.

Usage (from RoboPong/ directory):
    python3 scripts/train_ur5e.py --timesteps 1000000 --num_envs 8
    python3 scripts/train_ur5e.py --config configs/ppo_config.yaml
"""

import argparse
import os
import sys
import time

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Ensure project root is on python import path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from envs.hockey_ur5e_env import HockeyUR5eEnv


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


def main(args: argparse.Namespace):
    """Train UR5e air hockey rebound policy using Stable-Baselines3 PPO.

    Args:
        args: Parsed command-line arguments specifying hyperparameters and training options.
    """
    cfg = load_config(args.config)
    cfg_env = cfg.get("env", {})
    cfg_ppo = cfg.get("ppo", {})
    cfg_train = cfg.get("training", {})

    # CLI args override YAML config
    side_walls_val = (
        args.side_walls.lower() in ("true", "1", "yes")
        if args.side_walls is not None
        else cfg_env.get("side_walls", True)
    )
    total_timesteps = args.timesteps if args.timesteps is not None else cfg_train.get("total_timesteps", 1000000)
    num_envs = args.num_envs if args.num_envs is not None else cfg_train.get("num_envs", 8)
    lr = args.lr if args.lr is not None else cfg_ppo.get("learning_rate", 3e-3)
    seed = args.seed if args.seed is not None else cfg_train.get("seed", 42)

    save_dir = args.save_dir or os.path.join(_project_root, "saved_models", "ur5e_ppo")
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
    print(f"Learning Rate:   {lr}")
    print(f"Save Directory:  {save_dir}")
    print("=" * 68)

    # 1. Vectorized parallel training environments
    vec_env = make_vec_env(
        HockeyUR5eEnv,
        n_envs=num_envs,
        seed=seed,
        env_kwargs=dict(side_walls=side_walls_val, use_camera=cfg_env.get("use_camera", False)),
        vec_env_cls=SubprocVecEnv if num_envs > 1 else None,
    )
    vec_env = VecMonitor(vec_env, filename=os.path.join(log_dir, "train_monitor.csv"))

    # 2. Evaluation environment
    eval_env = make_vec_env(
        HockeyUR5eEnv,
        n_envs=1,
        seed=seed + 100,
        env_kwargs=dict(side_walls=side_walls_val, use_camera=cfg_env.get("use_camera", False)),
    )
    eval_env = VecMonitor(eval_env, filename=os.path.join(log_dir, "eval_monitor.csv"))

    # 3. Callbacks: periodic evaluation and model checkpoints
    eval_freq = cfg_train.get("eval_freq", max(25000 // num_envs, 2000))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=15,
        deterministic=True,
        render=False,
    )

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
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
    )

    start_time = time.time()
    print(
        f"\nStarting {total_timesteps:,} steps training run across {num_envs} parallel environments..."
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False,
    )
    elapsed = time.time() - start_time

    # 6. Save final model
    final_model_path = os.path.join(save_dir, "ppo_ur5e_final.zip")
    model.save(final_model_path)

    print("=" * 68)
    print(f"Training Complete in {elapsed:.1f}s ({elapsed/60.0:.2f} mins)!")
    print(f"Average Throughput: {total_timesteps / elapsed:.0f} steps/second")
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
        "--save_dir",
        type=str,
        default="",
        help="Directory to save checkpoints, logs, and final model",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
