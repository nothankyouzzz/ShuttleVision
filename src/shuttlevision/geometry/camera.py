"""
Camera model utilities that encapsulate intrinsics and extrinsics.

Coordinate systems:
- World frame is aligned with the court frame (x: length to opponent, y: width left->right, z: up).
- Camera frame follows the pinhole convention with z pointing forward.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from .types import CameraIntrinsics, CameraPose, GeometryError

_EPS = 1e-9


def _ensure_points(points: Float[np.ndarray, "..."]) -> tuple[np.ndarray, bool]:
    """Normalize input points to shape (N, 3), track whether it was a single point."""
    pts = np.asarray(points, dtype=float)
    is_single = False
    if pts.ndim == 1:
        if pts.shape[0] != 3:
            raise GeometryError("Point must have length 3.")
        pts = pts.reshape(1, 3)
        is_single = True
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise GeometryError("Points array must have shape (N, 3).")
    return pts, is_single


@dataclass(slots=True)
class CameraModel:
    """Pinhole camera model bundling intrinsics and pose."""

    pose: CameraPose

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self.pose.intrinsics

    @property
    def R_cw(self) -> Float[np.ndarray, "3 3"]:
        """Rotation matrix camera<-world."""
        return self.pose.R_wc.T

    @property
    def camera_center_m(self) -> Float[np.ndarray, "3"]:
        """Camera optical center expressed in world frame."""
        return -self.R_cw @ self.pose.t_wc_m

    def pixel_to_camera_ray(
        self,
        u_px: float,
        v_px: float,
    ) -> Float[np.ndarray, "3"]:
        """Pixel -> unit direction in camera frame."""
        intr = self.intrinsics
        if intr.fx_px == 0 or intr.fy_px == 0:
            raise GeometryError("Invalid intrinsics: fx or fy is zero")
        x_n = (u_px - intr.cx_px) / intr.fx_px
        y_n = (v_px - intr.cy_px) / intr.fy_px
        v_cam = np.array([x_n, y_n, 1.0], dtype=float)
        norm = float(np.linalg.norm(v_cam))
        if norm < _EPS:
            raise GeometryError("Degenerate ray direction")
        return v_cam / norm

    def pixel_to_world_direction(
        self,
        u_px: float,
        v_px: float,
    ) -> Float[np.ndarray, "3"]:
        """Pixel -> unit direction expressed in world frame."""
        ray_cam = self.pixel_to_camera_ray(u_px, v_px)
        direction_world = self.R_cw @ ray_cam
        norm = float(np.linalg.norm(direction_world))
        if norm < _EPS:
            raise GeometryError("Degenerate world ray direction")
        return direction_world / norm

    def world_to_camera(
        self,
        points_world_m: Float[np.ndarray, "..."],
    ) -> Float[np.ndarray, "N 3"]:
        """Transform points from world to camera frame."""
        pts, is_single = _ensure_points(points_world_m)
        pts_cam = (self.pose.R_wc @ pts.T + self.pose.t_wc_m.reshape(3, 1)).T
        return pts_cam[0] if is_single else pts_cam

    def camera_to_world(
        self,
        points_cam_m: Float[np.ndarray, "..."],
    ) -> Float[np.ndarray, "N 3"]:
        """Transform points from camera to world frame."""
        pts, is_single = _ensure_points(points_cam_m)
        pts_world = (pts - self.pose.t_wc_m.reshape(1, 3)) @ self.R_cw.T
        return pts_world[0] if is_single else pts_world

    def project_world_points(
        self,
        points_world_m: Float[np.ndarray, "..."],
    ) -> Float[np.ndarray, "N 2"]:
        """Project world points to pixel coordinates using current intrinsics."""
        pts_cam = self.world_to_camera(points_world_m)
        pts_cam_arr, is_single = _ensure_points(pts_cam)
        z = pts_cam_arr[:, 2]
        if np.any(np.isclose(z, 0.0)):
            raise GeometryError("Point projects to z=0 in camera frame")

        x = pts_cam_arr[:, 0] / z
        y = pts_cam_arr[:, 1] / z
        intr = self.intrinsics
        u_proj = intr.fx_px * x + intr.skew * y + intr.cx_px
        v_proj = intr.fy_px * y + intr.cy_px
        proj = np.stack([u_proj, v_proj], axis=1)
        return proj[0] if is_single else proj
