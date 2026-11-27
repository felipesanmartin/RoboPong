#!/usr/bin/env python3
"""Classical Controller for Two UR5e Arms Cooperatively Keeping a Puck in Play.

Runs the single-arm classical controller (ballistic intercept prediction, smoothed
Cartesian targets, DLS IK) independently on each arm, in that arm's own frame.

Usage (from RoboPong/ directory):
    python3 controllers/classical_two_arm_controller.py
    python3 controllers/classical_two_arm_controller.py --headless --episodes 20
    python3 controllers/classical_two_arm_controller.py --headless --episodes 3 --record videos/two_arm_classical
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

# Ensure project paths are on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from controllers.classical_ur5e_controller import predict_intercept_y
from envs.two_arm_ur5e_env import TwoArmHockeyUR5eEnv


class VideoWriter:
    """Pipe RGB frames to ffmpeg as an H.264 mp4."""

    def __init__(self, path: str, width: int, height: int, fps: int):
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def main(args: argparse.Namespace):
    """Run the classical controller on both arms.

    Args:
        args: Parsed command-line arguments specifying simulation options.
    """
    render_mode = "rgb_array" if args.record else (None if args.headless else "human")
    env = TwoArmHockeyUR5eEnv(render_mode=render_mode)
    if args.record:
        os.makedirs(args.record, exist_ok=True)

    print("=" * 68)
    print("Two-Arm UR5e Air Hockey Classical Controller")
    print(f"Episodes:    {args.episodes or 'continuous'} (seeds from {args.seed_start})")
    print(f"Rendering:   {'Recording to ' + args.record if args.record else 'Headless' if args.headless else 'Interactive MuJoCo GUI'}")
    print("=" * 68)

    # Same smoothing limits as the single-arm classical controller
    max_dx, max_dy = 0.08, 0.12
    hits, lengths = [], []
    ep = 0
    try:
        while not args.episodes or ep < args.episodes:
            env.reset(seed=args.seed_start + ep)
            ep += 1
            cur = [np.array([env.base_xe, 0.0]) for _ in env.arms]
            writer = None
            if args.record:
                h, w = env.render_size
                writer = VideoWriter(os.path.join(args.record, f"episode_{ep:02d}.mp4"), w, h, env.metadata["render_fps"])
                writer.write(env.render())

            done = False
            while not done:
                t_start = time.time()
                targets = []
                for arm, c in zip(env.arms, cur):
                    px, py, pvx, pvy = env.puck_local(arm)
                    ty = predict_intercept_y(px, py, pvx, pvy, paddle_x=env.base_xe, side_walls=True)
                    c[0] += float(np.clip(env.base_xe - c[0], -max_dx, max_dx))
                    c[1] += float(np.clip(ty - c[1], -max_dy, max_dy))
                    targets.append(c.copy())

                _, _, terminated, truncated, info = env.step_targets(targets)
                done = terminated or truncated

                if writer is not None:
                    writer.write(env.render())
                elif env.viewer is not None:
                    if not env.viewer.is_running():
                        break
                    elapsed = time.time() - t_start
                    if env.dt_ctrl - elapsed > 0:
                        time.sleep(env.dt_ctrl - elapsed)

            if writer is not None:
                writer.close()
            hits.append(info["wall_hits"])
            lengths.append(info["step_count"])
            status = f"SURVIVED ({env.max_episode_steps} STEPS)" if truncated and not terminated else f"MISSED BY ARM {str(info['missed_by']).upper()}"
            print(
                f"Episode {ep:3d} | Status: {status:22s} | Paddle hits: {info['wall_hits']:2d} "
                f"(A {info['rebounds_a']:2d} returns, B {info['rebounds_b']:2d} returns) | Steps: {info['step_count']:4d}"
            )
            if env.viewer is not None and not env.viewer.is_running():
                break
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
    finally:
        env.close()

    if hits:
        n = len(hits)
        survived = sum(l >= env.max_episode_steps for l in lengths)
        print("=" * 68)
        print(f"Episodes: {n} | Paddle hits/ep: {np.mean(hits):.2f} ± {np.std(hits) / np.sqrt(n):.2f} | "
              f"Survival: {survived}/{n} ({survived / n:.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-arm UR5e air hockey classical controller")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes to run (0 = run continuously)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--seed_start", type=int, default=1000, help="Seed of the first episode (default: 1000)")
    parser.add_argument("--record", type=str, default="", help="Directory to save one mp4 per episode")
    main(parser.parse_args())
