"""Vision-Based Color Detector & State Estimator for Air Hockey Puck.

Uses offscreen camera rendering from a bird's-eye camera, HSV color thresholding,
and adaptive velocity filtering to estimate puck position and velocity without
direct access to the underlying physics simulator state.
"""

import time
from typing import Optional, Tuple

import cv2
import mujoco
import numpy as np


class PuckColorDetector:
    """Detects and estimates puck position and velocity from bird's-eye camera images."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str = "top_down",
        width: int = 480,
        height: int = 360,
        puck_body_name: str = "hockey_puck",
    ):
        """Initialize the camera renderer, detector parameters, and calibrate projection."""
        self.model = model
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.puck_body_name = puck_body_name

        # Resolve camera ID
        try:
            self.camera_id = model.camera(camera_name).id
        except Exception:
            self.camera_id = 0

        # Offscreen renderer
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

        # State estimation filters
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.last_time: Optional[float] = None
        self.filt_vx: float = 0.0
        self.filt_vy: float = 0.0

        # HSV red detection bounds (red wraps across 0 / 180 in HSV)
        self.lower_red1 = np.array([0, 50, 80], dtype=np.uint8)
        self.upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        self.lower_red2 = np.array([170, 50, 80], dtype=np.uint8)
        self.upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        # Calibration parameters (pixel u, v -> table x, y)
        self.u0 = self.width / 2.0
        self.v0 = self.height / 2.0
        self.scale_x = 0.0126
        self.scale_y = 0.0126

        self._auto_calibrate()

    def _auto_calibrate(self):
        """Calibrate the linear pixel-to-world mapping using a scratch model evaluation."""
        try:
            d_scratch = mujoco.MjData(self.model)
            puck_body_id = self.model.body(self.puck_body_name).id
            puck_jnt_id = self.model.body_jntadr[puck_body_id]
            qpos_adr = self.model.jnt_qposadr[puck_jnt_id]
            body_init_x = self.model.body_pos[puck_body_id][0]

            def get_centroid(data):
                self.renderer.update_scene(data, camera=self.camera_name)
                img = self.renderer.render()
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, self.lower_red1, self.upper_red1) | cv2.inRange(
                    hsv, self.lower_red2, self.upper_red2
                )
                v_c, u_c = np.where(mask > 0)
                if len(u_c) == 0:
                    return self.width / 2.0, self.height / 2.0
                return float(np.mean(u_c)), float(np.mean(v_c))

            # Reference point 1: origin (0.0, 0.0)
            d_scratch.qpos[qpos_adr] = 0.0 - body_init_x
            d_scratch.qpos[qpos_adr + 1] = 0.0
            mujoco.mj_forward(self.model, d_scratch)
            u0, v0 = get_centroid(d_scratch)

            # Reference point 2: offset (1.0, 0.5)
            d_scratch.qpos[qpos_adr] = 1.0 - body_init_x
            d_scratch.qpos[qpos_adr + 1] = 0.5
            mujoco.mj_forward(self.model, d_scratch)
            u1, v1 = get_centroid(d_scratch)

            dv = v1 - v0
            du = u1 - u0
            if abs(dv) > 1.0 and abs(du) > 1.0:
                self.u0 = u0
                self.v0 = v0
                self.scale_x = 1.0 / dv
                self.scale_y = 0.5 / du
        except Exception:
            # Fallback to analytical fovy-based projection
            cam_pos = self.model.cam_pos[self.camera_id]
            fovy = self.model.cam_fovy[self.camera_id]
            dz = max(1.0, cam_pos[2] - 0.03)
            H_vis = 2.0 * dz * np.tan(np.radians(fovy) / 2.0)
            scale = H_vis / self.height
            self.u0 = (self.width - 1) / 2.0
            self.v0 = (self.height - 1) / 2.0
            self.scale_x = scale
            self.scale_y = scale

    def reset(self):
        """Reset velocity and position history."""
        self.last_x = None
        self.last_y = None
        self.last_time = None
        self.filt_vx = 0.0
        self.filt_vy = 0.0

    def detect(
        self,
        data: mujoco.MjData,
        annotate: bool = True,
    ) -> Tuple[float, float, float, float, np.ndarray, bool]:
        """Detect the puck from camera image and estimate position and velocity.

        Args:
            data: Current MuJoCo simulation data.
            annotate: Whether to draw bounding circles, velocity vectors, and HUD on image.

        Returns:
            Tuple of (x_est, y_est, vx_est, vy_est, annotated_bgr_image, found_boolean).
        """
        t_start = time.perf_counter()
        sim_time = data.time

        # 1. Render bird's-eye camera image
        self.renderer.update_scene(data, camera=self.camera_name)
        img_rgb = self.renderer.render()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 2. Color thresholding in HSV color space
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_red1, self.upper_red1) | cv2.inRange(
            hsv, self.lower_red2, self.upper_red2
        )

        v_coords, u_coords = np.where(mask > 0)
        found = len(u_coords) >= 10  # Minimum pixel cluster count

        if found:
            u_c = float(np.mean(u_coords))
            v_c = float(np.mean(v_coords))

            # Approximate radius in pixels
            r_px = int(np.sqrt(len(u_coords) / np.pi) * 1.1)

            # 3. Unproject pixels to table world coordinates
            x_est = float((v_c - self.v0) * self.scale_x)
            y_est = float((u_c - self.u0) * self.scale_y)

            # 4. Filter velocity
            if self.last_x is not None and self.last_time is not None:
                dt = max(1e-4, sim_time - self.last_time)
                vx_raw = (x_est - self.last_x) / dt
                vy_raw = (y_est - self.last_y) / dt

                # Detect sudden bounce (rebound direction reversal)
                if (vx_raw * self.filt_vx < -0.5) or (abs(vx_raw - self.filt_vx) > 3.0):
                    # Fast adaptation on wall or paddle bounce
                    self.filt_vx = vx_raw
                    self.filt_vy = vy_raw
                else:
                    # Exponential moving average filter
                    alpha = 0.45
                    self.filt_vx = (1.0 - alpha) * self.filt_vx + alpha * vx_raw
                    self.filt_vy = (1.0 - alpha) * self.filt_vy + alpha * vy_raw
            else:
                self.filt_vx = 0.0
                self.filt_vy = 0.0

            self.last_x = x_est
            self.last_y = y_est
            self.last_time = sim_time

        else:
            # Fallback to last known position if puck briefly occluded
            x_est = self.last_x if self.last_x is not None else 0.0
            y_est = self.last_y if self.last_y is not None else 0.0
            u_c = int(self.u0 + y_est / self.scale_y)
            v_c = int(self.v0 + x_est / self.scale_x)
            r_px = 10

        vx_est = float(self.filt_vx)
        vy_est = float(self.filt_vy)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # 5. Visual HUD Annotation
        if annotate:
            # Table centerline & baseline guides
            center_u = int(self.u0)
            center_v = int(self.v0)
            cv2.line(img_bgr, (center_u, 0), (center_u, self.height), (60, 60, 60), 1)
            cv2.line(img_bgr, (0, center_v), (self.width, center_v), (60, 60, 60), 1)

            if found:
                pt_center = (int(round(u_c)), int(round(v_c)))

                # Green tracking ring
                cv2.circle(img_bgr, pt_center, max(12, r_px), (0, 255, 0), 2)
                # Crosshair
                cv2.drawMarker(
                    img_bgr, pt_center, (0, 255, 0), cv2.MARKER_CROSS, 14, 1
                )

                # Velocity vector arrow (length proportional to speed)
                arrow_scale = 18.0
                arrow_end = (
                    int(round(u_c + vy_est * arrow_scale)),
                    int(round(v_c + vx_est * arrow_scale)),
                )
                if abs(vx_est) > 0.1 or abs(vy_est) > 0.1:
                    cv2.arrowedLine(
                        img_bgr, pt_center, arrow_end, (0, 255, 255), 2, tipLength=0.3
                    )

            # Top semi-transparent banner
            overlay = img_bgr.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, 68), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)

            status_text = "TRACKING" if found else "SEARCHING / LOST"
            status_color = (0, 255, 0) if found else (0, 0, 255)
            cv2.putText(
                img_bgr,
                f"BIRD'S-EYE PUCK DETECTOR: {status_text}",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                status_color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_bgr,
                f"Est Pos: X={x_est:+.2f}m, Y={y_est:+.2f}m",
                (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            speed = np.hypot(vx_est, vy_est)
            cv2.putText(
                img_bgr,
                f"Est Vel: Vx={vx_est:+.2f}m/s, Vy={vy_est:+.2f}m/s | Speed={speed:.2f}m/s | {latency_ms:.1f}ms",
                (8, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 230, 255),
                1,
                cv2.LINE_AA,
            )

        return x_est, y_est, vx_est, vy_est, img_bgr, found

    def show(
        self,
        img_bgr: np.ndarray,
        window_name: str = "Bird's-Eye Camera - Puck Detector",
        wait_ms: int = 1,
    ):
        """Display the camera image with OpenCV in an interactive window."""
        cv2.imshow(window_name, img_bgr)
        cv2.waitKey(wait_ms)

    def close(self):
        """Destroy OpenCV windows."""
        cv2.destroyAllWindows()
