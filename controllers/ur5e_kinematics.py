"""Damped Least Squares (DLS) Inverse Kinematics and Posture Control for UR5e Manipulator.

Provides 6-DOF task-space Cartesian positioning, horizontal tool alignment,
and null-space joint posture biasing for the air hockey striker paddle.
"""

from typing import Optional, Tuple

import mujoco
import numpy as np


class UR5eKinematics:
    """UR5e Air Hockey Striker Kinematics & Posture Controller."""

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "paddle_site",
    ):
        """Initialize kinematics solver with model reference and scratch data."""
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site_name).id

        # Nominal ready posture facing the table in an elbow-up configuration
        self.q_nominal = np.array(
            [-0.3927, -0.6714, 2.5217, -3.4155, -1.5710, -0.3935],
            dtype=np.float64,
        )

        # Workspace bounding box on the table surface (Z = 0.035 m)
        self.x_bounds = (-1.88, -1.45)
        self.y_bounds = (-0.72, 0.72)
        self.z_plane = 0.035

        # Joint motion limits from model (first 6 revolute joints)
        self.jnt_limits = model.jnt_range[:6].copy()

    def clamp_cartesian(self, target: np.ndarray) -> np.ndarray:
        """Clamp Cartesian target within the UR5e reachable air hockey workspace."""
        clamped = np.array(target, dtype=np.float64)
        clamped[0] = np.clip(clamped[0], self.x_bounds[0], self.x_bounds[1])
        clamped[1] = np.clip(clamped[1], self.y_bounds[0], self.y_bounds[1])
        # clamped[2] = self.z_plane
        return clamped

    def forward(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector Cartesian position and rotation matrix from joint angles."""
        self.data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, rot

    def inverse(
        self,
        target_pos: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iter: int = 40,
        tol_pos: float = 1e-3,
        tol_rot: float = 1.5e-2,
    ) -> Tuple[np.ndarray, bool]:
        """Solve Damped Least Squares Inverse Kinematics for Cartesian position and horizontal orientation.

        Args:
            target_pos: Desired 3D Cartesian position [x, y, z].
            q_init: Initial joint configuration guess (defaults to q_nominal).
            max_iter: Maximum Newton-Raphson iteration count.
            tol_pos: Position convergence tolerance in meters (default: 1.0 mm).
            tol_rot: Tilt convergence tolerance (default: ~0.85 degrees).

        Returns:
            Tuple of (joint_angles, converged_boolean).
        """
        target = self.clamp_cartesian(target_pos)

        if q_init is not None:
            self.data.qpos[:6] = q_init.copy()
        else:
            self.data.qpos[:6] = self.q_nominal.copy()

        converged = False
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        for _ in range(max_iter):
            mujoco.mj_forward(self.model, self.data)

            # Position error
            pos_err = target - self.data.site_xpos[self.site_id]

            # Tilt error: align tool Z-axis with world -Z [0, 0, -1]
            R = self.data.site_xmat[self.site_id].reshape(3, 3)
            tool_z = R[:, 2]
            rot_err = np.cross(tool_z, np.array([0.0, 0.0, -1.0]))

            norm_pos = np.linalg.norm(pos_err)
            norm_rot = np.linalg.norm(rot_err)

            if norm_pos < tol_pos and norm_rot < tol_rot:
                converged = True
                break

            # 6-DOF task space error vector
            err_6d = np.concatenate([pos_err, rot_err * 0.5])

            # Analytical Jacobians at paddle_site
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, :6], jacr[:, :6]])

            # Adaptive Levenberg-Marquardt damping
            lambda_sq = 1e-4 if norm_pos < 0.05 else 1e-3
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))

            # Secondary task: null-space posture bias toward human-like ready pose
            q_null = 0.08 * (self.q_nominal - self.data.qpos[:6])
            N = np.eye(6) - J_pinv @ J
            dq = J_pinv @ err_6d + N @ q_null

            # Limit max joint step to prevent overshoot
            step = np.clip(dq, -0.2, 0.2)
            self.data.qpos[:6] += step

            # Enforce actuator joint limits
            for j in range(6):
                low, high = self.jnt_limits[j]
                if low < high:
                    self.data.qpos[j] = np.clip(self.data.qpos[j], low, high)

        return self.data.qpos[:6].copy(), converged

    def get_gravity_compensation(self, data: mujoco.MjData) -> np.ndarray:
        """Compute gravity and Coriolis feedforward torque vector for UR5e joints."""
        return data.qfrc_bias[:6].copy()
