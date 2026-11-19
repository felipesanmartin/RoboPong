#!/usr/bin/env python3
"""Classical Task-Space Controller for 6-DOF UR5e Manipulator Rebounding Air Hockey Puck.

Uses Damped Least Squares (DLS) Jacobian Inverse Kinematics, horizontal tool alignment,
null-space posture biasing, and ballistic trajectory prediction with reflection physics
to position the end-effector paddle and execute high-speed intercepts and rebounds.

Usage (from RoboPong/ directory):
    python3 controllers/classical_ur5e_controller.py
    python3 controllers/classical_ur5e_controller.py --side_walls true
    python3 controllers/classical_ur5e_controller.py --use_camera
    python3 controllers/classical_ur5e_controller.py --headless --episodes 5
"""

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# Ensure project paths are on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from controllers.ur5e_kinematics import UR5eKinematics
from vision.puck_detector import PuckColorDetector


def predict_intercept_y(
    puck_x: float,
    puck_y: float,
    puck_vx: float,
    puck_vy: float,
    paddle_x: float = -1.75,
    wall_x: float = 1.89,
    wall_y: float = 0.79,
    side_walls: bool = False,
) -> float:
    """Predict puck Y-intercept at paddle baseline considering reflections.

    Args:
        puck_x: Current puck X coordinate.
        puck_y: Current puck Y coordinate.
        puck_vx: Current puck X velocity.
        puck_vy: Current puck Y velocity.
        paddle_x: Paddle intercept plane X coordinate.
        wall_x: Front wall rebound plane.
        wall_y: Side rebound wall boundary for puck center.
        side_walls: Whether lateral rebound walls are active.

    Returns:
        Predicted Y position clamped to UR5e reachable workspace.
    """
    if abs(puck_vx) < 1e-4:
        return 0.0

    cur_x, cur_y = puck_x, puck_y
    cur_vx, cur_vy = puck_vx, puck_vy

    # If puck is moving toward front wall (+X), project bounce
    if cur_vx > 0:
        dt_to_wall = (wall_x - cur_x) / cur_vx
        cur_x = wall_x
        cur_y += cur_vy * dt_to_wall
        cur_vx = -cur_vx

    # Project trajectory back to paddle baseline
    dt_to_paddle = (paddle_x - cur_x) / cur_vx
    target_y = cur_y + cur_vy * dt_to_paddle

    # Account for side wall reflections
    if side_walls:
        y_min, y_max = -wall_y, wall_y
        y_span = y_max - y_min
        shifted = target_y - y_min
        k = int(np.floor(shifted / y_span))
        rem = shifted - k * y_span
        if k % 2 == 1:
            rem = y_span - rem
        reflected_y = rem + y_min
    else:
        reflected_y = target_y

    return float(np.clip(reflected_y, -0.68, 0.68))


def reset_puck(data: mujoco.MjData, model: mujoco.MjModel):
    """Randomize and launch puck toward the front wall."""
    # Puck body initial offset is (-0.5, 0.0, 0.015)
    # Joints 6, 7 are slide_x, slide_y
    data.qpos[6] = np.random.uniform(-0.3, 0.2)
    data.qpos[7] = np.random.uniform(-0.35, 0.35)
    data.qvel[6] = np.random.uniform(4.5, 4.8)
    data.qvel[7] = np.random.uniform(-0.8, 0.8)
    data.qpos[8] = 0.0
    data.qvel[8] = 0.0


def main(args: argparse.Namespace):
    """Execute classical task-space trajectory tracking and striking control.

    Args:
        args: Parsed command-line arguments specifying simulation options.
    """
    side_walls = args.side_walls.lower() in ("true", "1", "yes")

    model_name = (
        "hockey_ur5e_world_walled.xml" if side_walls else "hockey_ur5e_world.xml"
    )
    candidate_paths = [
        os.path.join(_project_root, "models", "ur5e", model_name),
        os.path.join(_current_dir, "models", "ur5e", model_name),
    ]
    model_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(f"Model file not found in candidates: {candidate_paths}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    kin = UR5eKinematics(model)

    # Initialize vision-based puck color detector & state estimator
    detector = None
    if args.use_camera:  # or not args.headless:
        detector = PuckColorDetector(model, camera_name="top_down")

    print(detector)

    paddle_geom_id = model.geom("paddle_geom").id
    puck_geom_id = model.geom("puck_geom").id
    puck_body_id = model.body("hockey_puck").id
    site_id = model.site("paddle_site").id

    # Initialize robot and puck from ready keyframe
    data.qpos[:] = model.key("ready").qpos
    data.ctrl[:] = model.key("ready").ctrl
    reset_puck(data, model)
    mujoco.mj_forward(model, data)

    print("=" * 68)
    print("6-DOF UR5e Industrial Manipulator Air Hockey Controller")
    print(f"Model:       {model_name}")
    print(
        f"Side Walls:  {'ENABLED (rebound bank shots)' if side_walls else 'DISABLED (open court)'}"
    )
    print(
        f"Puck Sensor: {'BIRD-EYE CAMERA (Color Detector)' if args.use_camera else 'Ground Truth Physics State'}"
    )
    print(
        f"Rendering:   {'Headless mode' if args.headless else 'Interactive MuJoCo GUI (50 FPS) + Camera Feed'}"
    )
    print("=" * 68)

    ep_count = 0
    total_rebounds = 0
    dt_ctrl = 0.02  # 50 Hz control rate
    substeps = int(dt_ctrl / model.opt.timestep)

    viewer_ctx = (
        mujoco.viewer.launch_passive(model, data) if not args.headless else None
    )
    if viewer_ctx:
        viewer_ctx.cam.distance = 4.2
        viewer_ctx.cam.elevation = -40.0
        viewer_ctx.cam.azimuth = 90.0
        viewer_ctx.cam.lookat = [-0.5, 0.0, 0.0]

    q_cmd = data.qpos[:6].copy()
    base_xe = -1.75
    cur_target_xe = base_xe
    cur_target_ye = 0.0
    last_contact_time = -1.0

    try:
        while True:
            ep_count += 1
            reset_puck(data, model)
            if detector is not None:
                detector.reset()
            ep_rebounds = 0
            step_count = 0
            cur_target_xe = base_xe
            cur_target_ye = 0.0

            while step_count < 1000:
                t_step_start = time.time()

                # 0. Vision-based Puck State Estimation vs Ground Truth
                if args.use_camera and detector is not None:
                    px, py, pvx, pvy, cam_frame, _ = detector.detect(data)
                    if not args.headless:
                        detector.show(cam_frame)
                else:
                    px = -0.5 + data.qpos[6]
                    py = data.qpos[7]
                    pvx = data.qvel[6]
                    pvy = data.qvel[7]
                    if not args.headless and detector is not None:
                        _, _, _, _, cam_frame, _ = detector.detect(data)
                        detector.show(cam_frame)

                # 1. Trajectory intercept prediction
                target_ye_desired = predict_intercept_y(
                    px, py, pvx, pvy, paddle_x=base_xe, side_walls=side_walls
                )
                target_xe_desired = base_xe

                # Smooth Cartesian targets to prevent instantaneous jumps upon rebound
                max_dx = 0.08  # ~2.0 m/s max retreat velocity
                max_dy = 0.12  # ~3.0 m/s max tracking velocity
                cur_target_xe += float(
                    np.clip(target_xe_desired - cur_target_xe, -max_dx, max_dx)
                )
                cur_target_ye += float(
                    np.clip(target_ye_desired - cur_target_ye, -max_dy, max_dy)
                )

                target_cartesian = np.array(
                    [cur_target_xe, cur_target_ye, 0.035], dtype=np.float64
                )

                # 3. DLS Inverse Kinematics
                q_sol, ok = kin.inverse(target_cartesian, q_init=q_cmd, max_iter=25)
                if ok:
                    # Rate-limit joint commands to avoid high-frequency jerk
                    max_dq = 0.30  # rad per 20 ms step (~15 rad/s)
                    dq = np.clip(q_sol - q_cmd, -max_dq, max_dq)
                    q_cmd = q_cmd + dq
                else:
                    q_cmd = 0.85 * q_cmd + 0.15 * q_sol

                # 4. Position PD Actuator commands
                data.ctrl[:6] = q_cmd

                # 5. Physics Substepping (1000 Hz)
                contact_hit = False
                for _ in range(substeps):
                    # Gravity compensation feedforward
                    data.qfrc_applied[:6] = kin.get_gravity_compensation(data)
                    mujoco.mj_step(model, data)

                    if not contact_hit and (data.time - last_contact_time > 0.18):
                        for c in range(data.ncon):
                            con = data.contact[c]
                            if (
                                con.geom1 == paddle_geom_id
                                and con.geom2 == puck_geom_id
                            ) or (
                                con.geom1 == puck_geom_id
                                and con.geom2 == paddle_geom_id
                            ):
                                if data.qvel[6] > 0.3:
                                    contact_hit = True
                                    last_contact_time = data.time
                                    break

                if contact_hit:
                    ep_rebounds += 1
                    total_rebounds += 1
                    paddle_x = data.site_xpos[site_id][0]
                    print(
                        f"  [REBOUND #{total_rebounds:2d}] Hit at ({px:+.2f}, {py:+.2f}) | "
                        f"Outgoing Vx: {data.qvel[6]:+.2f} m/s | Paddle X: {paddle_x:+.2f} m"
                    )

                step_count += 1

                # Check puck miss or out of bounds
                if px < -1.95 or (not side_walls and abs(py) > 1.55):
                    break

                if viewer_ctx:
                    if not viewer_ctx.is_running():
                        break
                    viewer_ctx.sync()
                    elapsed = time.time() - t_step_start
                    if dt_ctrl - elapsed > 0:
                        time.sleep(dt_ctrl - elapsed)

            status = "SURVIVED (500 STEPS)" if step_count >= 500 else "MISSED / OUT"
            print(
                f"Episode {ep_count:3d} | Status: {status:20s} | "
                f"Rebounds: {ep_rebounds:2d} | Total: {total_rebounds:3d} | Steps: {step_count:3d}"
            )

            if viewer_ctx and not viewer_ctx.is_running():
                break

            if args.episodes > 0 and ep_count >= args.episodes:
                break

    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
    finally:
        if viewer_ctx and viewer_ctx.is_running():
            viewer_ctx.close()
        if detector is not None:
            detector.close()

    print(
        f"\nFinal Summary: {ep_count} episodes, {total_rebounds} total rebounds executed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="6-DOF UR5e Manipulator Air Hockey Controller"
    )
    parser.add_argument(
        "--side_walls",
        type=str,
        default="false",
        help="Include lateral side rebound walls (true/false, default: false)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 = run continuously)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (headless test mode)",
    )
    parser.add_argument(
        "--use_camera",
        action="store_true",
        default=False,
        help="Estimate puck state from bird's-eye camera color detector instead of simulator state",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
