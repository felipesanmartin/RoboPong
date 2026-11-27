#!/usr/bin/env python3
"""Run and Visualize Trained UR5e PPO Policy in MuJoCo Interactive Viewer.

Usage (from RoboPong/ directory):
    python3 scripts/run_ur5e.py
    python3 scripts/run_ur5e.py --use_camera
    python3 scripts/run_ur5e.py --model_path saved_models/ur5e_ppo/best_model/best_model.zip
    python3 scripts/run_ur5e.py --headless --episodes 5
    python3 scripts/run_ur5e.py --headless --episodes 100 --modes gt camera \\
        --models rebound=saved_models/ur5e_ppo/ppo_ur5e_final.zip \\
                 sparse=saved_models/ur5e_ppo_sparse/ppo_ur5e_final.zip
    python3 scripts/run_ur5e.py --env two_arm --reward sparse
    python3 scripts/run_ur5e.py --env two_arm --reward sparse --independent
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

from envs.two_arm_ur5e_env import TwoArmHockeyUR5eEnv
from envs.hockey_ur5e_env import HockeyUR5eEnv
from envs.rewards import REWARD_TYPES
from scripts.train_ur5e import default_model_dir, load_config, reward_params_for


def find_default_model(reward_type: str = "rebound", env_type: str = "single") -> str:
    """Locate the best or final trained UR5e PPO model for the given reward and env."""
    model_dir = default_model_dir(reward_type, env_type)
    for path in [
        os.path.join(model_dir, "best_model", "best_model.zip"),
        os.path.join(model_dir, "ppo_ur5e_final.zip"),
    ]:
        if os.path.exists(path):
            return path
    return ""


def make_policy(model, env, deterministic: bool):
    """Return obs -> action for env, running single-arm policies on each arm of the two-arm env."""
    single_arm = model.observation_space.shape[0] == 23
    if isinstance(env, TwoArmHockeyUR5eEnv) and single_arm:
        low, high = model.observation_space.low, model.observation_space.high

        def act(obs):
            # Each arm sees only its own observation, clipped like in single-arm training
            return np.concatenate([
                model.predict(np.clip(env.local_obs(arm), low, high), deterministic=deterministic)[0]
                for arm in env.arms
            ])

        return act
    return lambda obs: model.predict(obs, deterministic=deterministic)[0]


def run_episodes(policy, env, args: argparse.Namespace, verbose: bool) -> dict:
    """Run seeded episodes and collect per-episode results.

    Args:
        policy: Function mapping an observation to an action.
        env: HockeyUR5eEnv to run in.
        args: Parsed command-line arguments.
        verbose: Print a line per episode.

    Returns:
        Dictionary of per-episode reward, length, rebounds and wall hits lists.
    """
    results = {"rewards": [], "lengths": [], "rebounds": [], "wall_hits": []}
    dt = env.dt_ctrl

    for ep in range(1, args.episodes + 1):
        # Same seeds for every model so they all face the same pucks
        obs, info = env.reset(seed=args.seed_start + ep - 1)
        ep_reward = 0.0
        step_count = 0
        done = False

        while not done:
            t_start = time.time()

            action = policy(obs)
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

        results["rewards"].append(ep_reward)
        results["lengths"].append(step_count)
        results["rebounds"].append(info.get("rebounds", 0))
        results["wall_hits"].append(info.get("wall_hits", 0))

        if verbose:
            status = f"SURVIVED ({env.max_episode_steps} STEPS)" if step_count >= env.max_episode_steps else "MISSED / OUT"
            hit_label = "Paddle hits" if isinstance(env, TwoArmHockeyUR5eEnv) else "Wall hits"
            print(
                f"Episode {ep:2d} | Status: {status:21s} | "
                f"Reward: {ep_reward:7.1f} | Rebounds: {info.get('rebounds', 0):2d} | "
                f"{hit_label}: {info.get('wall_hits', 0):2d} | "
                f"Steps: {step_count:3d}"
            )

        if not args.headless and env.viewer is not None and not env.viewer.is_running():
            break

    return results


def print_summary(results: dict, max_steps: int, hit_label: str = "Wall Hits"):
    """Print the summary block for one model and puck sensor."""
    n = max(1, len(results["rewards"]))
    total_rebounds = sum(results["rebounds"])
    total_wall_hits = sum(results["wall_hits"])
    survived = sum(l >= max_steps for l in results["lengths"])
    print("=" * 68)
    print("Evaluation Results Summary:")
    print(f"Episodes Completed: {len(results['rewards'])}")
    print(f"Mean Episode Reward: {np.mean(results['rewards']):.1f} +/- {np.std(results['rewards']):.1f}")
    print(f"Total Rebounds:     {total_rebounds} ({total_rebounds / n:.1f} per episode)")
    print(f"Total {hit_label + ':':13s}{total_wall_hits} ({total_wall_hits / n:.1f} per episode)")
    print(f"Survival Rate:      {survived}/{len(results['lengths'])} ({survived / n:.0%})")
    print(f"Mean Episode Length: {np.mean(results['lengths']):.1f} steps")
    print("=" * 68)


def main(args: argparse.Namespace):
    """Evaluate and visualize trained UR5e air hockey policies.

    Args:
        args: Parsed command-line arguments specifying evaluation settings.
    """
    side_walls = args.side_walls.lower() in ("true", "1", "yes")
    reward_params = reward_params_for(load_config(args.config), args.reward)

    if args.models:
        models = dict(spec.split("=", 1) for spec in args.models)
    elif args.independent:
        models = {f"{args.reward}_independent": args.model_path or find_default_model(args.reward)}
    else:
        models = {args.reward: args.model_path or find_default_model(args.reward, args.env)}
    for name, path in models.items():
        if not path or not os.path.exists(path):
            print(f"Error: Trained UR5e model '{name}' not found at '{path}'!")
            print("Please train a model first using train_ur5e.py or specify --model_path.")
            sys.exit(1)

    modes = args.modes or (["camera"] if args.use_camera else ["gt"])
    if args.env == "two_arm" and "camera" in modes:
        print("Error: the two-arm env only supports ground-truth puck state (--modes gt).")
        sys.exit(1)
    verbose = len(models) == 1 and len(modes) == 1

    print("=" * 68)
    print("Evaluating Trained 6-DOF UR5e Hockey Rebound Policy")
    for name, path in models.items():
        print(f"Model:       {path}" if verbose else f"Model:       {name} = {path}")
    print(f"Side Walls:  {'ENABLED (rebound bank shots)' if side_walls else 'DISABLED (open court)'}")
    print(f"Puck Sensor: {', '.join('BIRD-EYE CAMERA (Color Detector)' if m == 'camera' else 'Ground Truth State' for m in modes)}")
    print(f"Environment: {args.env}")
    print(f"Reward:      {args.reward}")
    print(f"Episodes:    {args.episodes} (seeds {args.seed_start}-{args.seed_start + args.episodes - 1})")
    print(f"Rendering:   {'Headless' if args.headless else 'Interactive MuJoCo GUI (50 FPS) + Camera Feed'}")
    print("=" * 68)

    results = {}
    for mode in modes:
        if args.env == "two_arm":
            env = TwoArmHockeyUR5eEnv(
                render_mode=None if args.headless else "human",
                reward_type=args.reward,
                reward_params=reward_params,
            )
        else:
            env = HockeyUR5eEnv(
                render_mode=None if args.headless else "human",
                side_walls=side_walls,
                use_camera=(mode == "camera"),
                show_camera=not args.headless,
                reward_type=args.reward,
                reward_params=reward_params,
            )
        hit_label = "Paddle Hits" if args.env == "two_arm" else "Wall Hits"
        try:
            for name, path in models.items():
                # Small policy, so CPU keeps the GPU free for other work
                model = PPO.load(path, device="cpu")
                policy = make_policy(model, env, args.deterministic)
                results[(name, mode)] = run_episodes(policy, env, args, verbose)
                if verbose:
                    print_summary(results[(name, mode)], env.max_episode_steps, hit_label)
                else:
                    print(f"[{mode}] {name} done", flush=True)
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
            break
        finally:
            max_steps = env.max_episode_steps
            env.close()

    # Wall hits are mean ± standard error over episodes
    header = f"{'policy':24s}" + "".join(
        f" | {mode.upper() + ' hits':>13s} {mode.upper() + ' survive':>14s}" for mode in modes
    )
    print("\n" + header + "\n" + "-" * len(header))
    for name in models:
        line = f"{name:24s}"
        for mode in modes:
            r = results.get((name, mode))
            if not r or not r["lengths"]:
                line += f" | {'-':>13s} {'-':>14s}"
                continue
            hits = np.array(r["wall_hits"], dtype=float)
            survive = np.mean([l >= max_steps for l in r["lengths"]])
            line += f" | {hits.mean():6.2f} ± {hits.std() / np.sqrt(len(hits)):4.2f} {survive:14.0%}"
        print(line)


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
        "--models",
        nargs="+",
        default=[],
        help="Evaluate several models as name=path.zip entries (overrides --model_path)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["single", "two_arm"],
        default="single",
        help="single-arm rebound task or two-arm task (default: single)",
    )
    parser.add_argument(
        "--independent",
        action="store_true",
        help="In the two-arm env, run the single-arm model for --reward on each arm",
    )
    parser.add_argument(
        "--reward",
        type=str,
        choices=REWARD_TYPES,
        default="rebound",
        help="Reward the policy was trained with; picks default model dir and reward printout",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="YAML config to read reward params from (default: configs/ppo_config.yaml)",
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
        "--modes",
        nargs="+",
        choices=["gt", "camera"],
        default=None,
        help="Puck sensors to evaluate with, e.g. '--modes gt camera' (overrides --use_camera)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=1000,
        help="Seed of the first episode, kept separate from training seeds (default: 1000)",
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
