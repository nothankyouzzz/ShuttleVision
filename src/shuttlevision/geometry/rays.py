from dataclasses import dataclass

import numpy as np

from shuttlevision.tracking import Track2D

from .types import CameraIntrinsics, CameraPose

Array1F = np.ndarray
Array2F = np.ndarray


@dataclass(slots=True)
class Ray3D:
    origin_m: Array1F  # shape=(3,)
    direction: Array1F  # shape=(3,), 归一化


@dataclass(slots=True)
class RayObservation:
    """
    将 TrackPoint2D 与几何信息联系起来。
    """

    track_id: int
    frame_index: int
    timestamp_s: float
    ray: Ray3D
    score: float


def pixel_to_camera_ray(
    u_px: float,
    v_px: float,
    K: CameraIntrinsics,
) -> Array1F:
    """
    将像素坐标变换到相机坐标系中的单位方向向量。
    """
    # 去中心化
    x_n = (u_px - K.cx_px) / K.fx_px
    y_n = (v_px - K.cy_px) / K.fy_px
    # 在相机坐标系中，z = 1 平面上的一点
    v_cam = np.array([x_n, y_n, 1.0], dtype=float)
    # 归一化
    v_cam /= np.linalg.norm(v_cam)
    return v_cam


def camera_to_world_ray(
    ray_cam: Array1F,
    pose: CameraPose,
) -> Ray3D:
    """
    将相机坐标系下的射线方向转换到世界坐标系，并设置 origin 为相机光心。
    """
    # R_wc: world->camera, 所以 R_cw = R_wc^T
    R_cw = pose.R_wc.T
    direction_world = R_cw @ ray_cam
    direction_world /= np.linalg.norm(direction_world)

    # 相机光心在世界坐标系下的位置：
    # x_cam = R_wc * x_world + t_wc
    # 令 x_cam = 0 => x_world = -R_cw * t_wc
    origin_m = -R_cw @ pose.t_wc_m

    return Ray3D(origin_m=origin_m, direction=direction_world)


def pixel_to_world_ray(
    u_px: float,
    v_px: float,
    pose: CameraPose,
) -> Ray3D:
    """
    组合 pixel_to_camera_ray + camera_to_world_ray。
    """
    ray_cam = pixel_to_camera_ray(u_px, v_px, pose.intrinsics)
    return camera_to_world_ray(ray_cam, pose)


def convert_track_to_rays(
    track: Track2D,
    pose: CameraPose,
) -> list[RayObservation]:
    """
    将单条 2D 轨迹转为 RayObservation 序列。
    """
    obs: list[RayObservation] = []
    for idx, p in enumerate(track.points):
        ray = pixel_to_world_ray(p.u_px, p.v_px, pose)
        obs.append(
            RayObservation(
                track_id=track.track_id,
                frame_index=idx,  # 或实际 frame_index，视 TrackPoint2D 是否携带
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
    """批量转换所有轨迹。"""
    rays: list[RayObservation] = []
    for t in tracks:
        rays.extend(convert_track_to_rays(t, camera_pose))
    return rays
