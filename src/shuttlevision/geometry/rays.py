from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from shuttlevision.tracking import Track2D, TrackPoint2D

from .camera import CameraModel
from .types import CameraIntrinsics, CameraPose, GeometryError

_EPS = 1e-9


@dataclass(slots=True)
class Ray3D:
    origin_m: Float[np.ndarray, "3"]  # camera optical center in world frame
    direction: Float[np.ndarray, "3"]  # unit vector in world frame


@dataclass(slots=True)
class RayObservation:
    """Link a 2D track point with its corresponding 3D ray."""

    track_id: int
    frame_index: int
    timestamp_s: float
    ray: Ray3D
    score: float


def pixel_to_camera_ray(
    u_px: float,
    v_px: float,
    K: CameraIntrinsics,
) -> Float[np.ndarray, "3"]:
    """Transform pixel coordinate into a unit direction in the camera frame."""
    if K.fx_px == 0 or K.fy_px == 0:
        raise GeometryError("Invalid intrinsics: fx or fy is zero")

    x_n = (u_px - K.cx_px) / K.fx_px
    y_n = (v_px - K.cy_px) / K.fy_px
    v_cam = np.array([x_n, y_n, 1.0], dtype=float)
    norm = float(np.linalg.norm(v_cam))
    if norm < _EPS:
        raise GeometryError("Degenerate ray direction")

    return v_cam / norm


def camera_to_world_ray(
    ray_cam: Float[np.ndarray, "3"],
    pose: CameraPose,
) -> Ray3D:
    """Convert camera-frame ray to world-frame ray, origin at camera center."""
    camera = CameraModel(pose)
    direction_world = camera.R_cw @ ray_cam
    dir_norm = float(np.linalg.norm(direction_world))
    if dir_norm < _EPS:
        raise GeometryError("Degenerate world ray direction")

    origin_m = camera.camera_center_m
    return Ray3D(origin_m=origin_m, direction=direction_world / dir_norm)


def pixel_to_world_ray(
    u_px: float,
    v_px: float,
    pose: CameraPose,
    camera_model: CameraModel | None = None,
) -> Ray3D:
    """Convenience wrapper combining pixel_to_camera_ray and camera_to_world_ray."""
    camera = camera_model or CameraModel(pose)
    direction_world = camera.pixel_to_world_direction(u_px, v_px)
    return Ray3D(origin_m=camera.camera_center_m, direction=direction_world)


def _resolve_frame_index(idx: int, point: TrackPoint2D) -> int:
    """Backward-compatible shim for future TrackPoint2D.frame_index."""
    if hasattr(point, "frame_index"):
        frame_val = getattr(point, "frame_index")
        if isinstance(frame_val, int):
            return frame_val
    return idx


def convert_track_to_rays(
    track: Track2D,
    pose: CameraPose,
    camera_model: CameraModel | None = None,
) -> list[RayObservation]:
    """Convert one 2D track into a sequence of RayObservation."""
    obs: list[RayObservation] = []
    camera = camera_model or CameraModel(pose)
    for idx, p in enumerate(track.points):
        ray = pixel_to_world_ray(p.u_px, p.v_px, pose, camera_model=camera)
        obs.append(
            RayObservation(
                track_id=track.track_id,
                frame_index=_resolve_frame_index(idx, p),
                timestamp_s=p.timestamp_s,
                ray=ray,
                score=p.score,
            )
        )
    return obs


def convert_tracks_to_rays(
    tracks: list[Track2D],
    camera_pose: CameraPose,
) -> list[RayObservation]:
    """Batch convert all tracks to RayObservation."""
    rays: list[RayObservation] = []
    camera = CameraModel(camera_pose)
    for t in tracks:
        rays.extend(convert_track_to_rays(t, camera_pose, camera_model=camera))
    return rays
