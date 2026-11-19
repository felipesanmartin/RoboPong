#!/usr/bin/env python3
"""Run and Visualize Trained UR5e PPO Policy in MuJoCo Interactive Viewer.

Usage (from RoboPong/ directory):
    python3 scripts/run_ur5e.py
    python3 scripts/run_ur5e.py --use_camera
    python3 scripts/run_ur5e.py --model_path saved_models/ur5e_ppo/best_model/best_model.zip
    python3 scripts/run_ur5e.py --headless --episodes 5
"""

import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO

# Ensure project root is on python import path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from envs.hockey_ur5e_env import HockeyUR5eEnv


def find_default_model() -> str:
    """Locate the best or final trained UR5e PPO model."""
    candidates = [
        os.path.join(_project_root, "saved_models", "ur5e_ppo", "best_model", "best_model.zip"),
        os.path.join(_project_root, "saved_models", "ur5e_ppo", "ppo_ur5e_final.zip"),
        os.path.join(_current_dir, "saved_models", "ur5e_ppo", "best_model", "best_model.zip"),
        os.path.join(_current_dir, "saved_models", "ur5e_ppo", "ppo_ur5e_final.zip"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def main(args: argparse.Namespace):
    """Evaluate and visualize trained UR5e air hockey policy.

    Args:
        args: Parsed command-line arguments specifying evaluation settings.
    """
    side_walls = args.side_walls.lower() in ("true", "1", "yes")

    model_path = args.model_path or find_default_model()
    if not model_path or not os.path.exists(model_path):
        print(f"Error: Trained UR5e model not found at '{model_path}'!")
        print("Please train a model first using train_ur5e_sb3.py or specify --model_path.")
        sys.exit(1)

    print("=" * 68)
    print("Evaluating Trained 6-DOF UR5e Hockey Rebound Policy")
    print(f"Model:       {model_path}")
    print(f"Side Walls:  {'ENABLED (rebound bank shots)' if side_walls else 'DISABLED (open court)'}")
    print(f"Puck Sensor: {'BIRD-EYE CAMERA (Color Detector)' if args.use_camera else 'Ground Truth State'}")
    print(f"Episodes:    {args.episodes}")
    print(f"Rendering:   {'Headless' if args.headless else 'Interactive MuJoCo GUI (50 FPS) + Camera Feed'}")
    print("=" * 68)

    render_mode = None if args.headless else "human"
    env = HockeyUR5eEnv(
        render_mode=render_mode,
        side_walls=side_walls,
        use_camera=args.use_camera,
        show_camera=not args.headless,
    )
    model = PPO.load(model_path, env=env)

    total_rebounds = 0
    total_rewards = []
    episode_lengths = []
    dt = env.dt_ctrl

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            ep_reward = 0.0
            step_count = 0
            done = False

            while not done:
                t_start = time.time()

                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step_count += 1
                done = terminated or truncated

                if not args.headless and env.viewer is not None:
                    if not env.viewer.is_running():
                        done = True
                        break
                    elapsed = time.time() - t_start
                    if dt - elapsed > 0:
                        time.sleep(dt - elapsed)

            total_rebounds += info.get("rebounds", 0)
            total_rewards.append(ep_reward)
            episode_lengths.append(step_count)

            status = "SURVIVED (500 STEPS)" if step_count >= 500 else "MISSED / OUT"
            print(
                f"Episode {ep:2d} | Status: {status:20s} | "
                f"Reward: {ep_reward:7.1f} | Rebounds: {info.get('rebounds', 0):2d} | "
                f"Steps: {step_count:3d}"
            )

            if not args.headless and env.viewer is not None and not env.viewer.is_running():
                break

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        env.close()

    print("=" * 68)
    print("Evaluation Results Summary:")
    print(f"Episodes Completed: {len(total_rewards)}")
    print(f"Mean Episode Reward: {np.mean(total_rewards):.1f} +/- {np.std(total_rewards):.1f}")
    print(f"Total Rebounds:     {total_rebounds} ({total_rebounds/max(1, len(total_rewards)):.1f} per episode)")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} steps")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate & Visualize Trained UR5e Air Hockey PPO Policy"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to trained .zip model (default: searches saved_models/ur5e_ppo/)",
    )
    parser.add_argument(
        "--side_walls",
        type=str,
        default="true",
        help="Include lateral side rebound walls (true/false, default: true)",
    )
    parser.add_argument(
        "--use_camera",
        action="store_true",
        default=False,
        help="Estimate puck state from bird's-eye camera color detector instead of simulator state",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI rendering",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
