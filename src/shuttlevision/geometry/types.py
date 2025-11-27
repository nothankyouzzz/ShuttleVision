from dataclasses import dataclass

import numpy as np

Array1F = np.ndarray
Array2F = np.ndarray


@dataclass(slots=True)
class CourtDimensions:
    length_m: float
    width_m: float
    net_height_m: float
    # 以后可加单/双打不同线的尺寸


@dataclass(slots=True)
class CourtCoordinateSystem:
    """
    定义球场坐标系的约定：
    - 原点：左后场角（面对球网时左手边、后端线与边线交点）
    - x 轴：沿底线到对面底线方向（正向指向对面）
    - y 轴：沿左到右方向
    - z 轴：竖直向上
    """

    origin_description: str = "rear-left corner"
    axes_description: str = "x: length, y: width, z: up"


@dataclass(slots=True)
class CameraIntrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    skew: float = 0.0
    # 畸变先预留
    k1: float | None = None
    k2: float | None = None
    p1: float | None = None
    p2: float | None = None
    k3: float | None = None


@dataclass(slots=True)
class CameraPose:
    """
    世界坐标系 = 球场坐标系
    """

    R_wc: Array2F  # shape=(3, 3)，world->camera 旋转
    t_wc_m: Array1F  # shape=(3,)，world->camera 平移（单位：米）
    intrinsics: CameraIntrinsics


@dataclass(slots=True)
class CalibrationInput:
    """
    来自标定工具的输入。
    """

    image_width_px: int
    image_height_px: int
    # 像素点：球场四角、网柱点等
    points_image_px: Array2F  # shape=(N, 2)
    points_world_m: Array2F  # shape=(N, 3)


@dataclass(slots=True)
class CalibrationResult:
    camera_pose: CameraPose
    reprojection_error_px: float
    num_points: int


class CalibrationError(RuntimeError):
    pass
