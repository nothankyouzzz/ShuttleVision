from dataclasses import dataclass
from jaxtyping import Float
import numpy as np


@dataclass(slots=True)
class CourtDimensions:
    length_m: float
    width_m: float
    net_height_m: float
    # Room to extend for single/doubles specific line offsets.


@dataclass(slots=True)
class CourtCoordinateSystem:
    """
    Court coordinate convention:
    - Origin: rear-left corner (facing the net, left hand side on the near baseline)
    - x-axis: along court length pointing to the opponent side
    - y-axis: along court width from left to right
    - z-axis: up
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
    # Distortion coefficients are optional placeholders.
    k1: float | None = None
    k2: float | None = None
    p1: float | None = None
    p2: float | None = None
    k3: float | None = None


@dataclass(slots=True)
class CameraPose:
    """World coordinate system is aligned with the court frame."""

    R_wc: Float[np.ndarray, "3 3"]  # world->camera rotation
    t_wc_m: Float[np.ndarray, "3"]  # world->camera translation in meters
    intrinsics: CameraIntrinsics


@dataclass(slots=True)
class CalibrationInput:
    """Inputs produced by the calibration tool."""

    image_width_px: int
    image_height_px: int
    points_image_px: Float[np.ndarray, "N 2"]
    points_world_m: Float[np.ndarray, "N 3"]


@dataclass(slots=True)
class CalibrationResult:
    camera_pose: CameraPose
    reprojection_error_px: float
    num_points: int


class CalibrationError(RuntimeError):
    pass


class GeometryError(RuntimeError):
    pass


@dataclass(slots=True)
class CalibrationConfig:
    """
    Tunable thresholds and parameters for camera calibration.
    """

    ransac_reprojection_error_px: float = 8.0
    ransac_confidence: float = 0.99
    reprojection_error_threshold_px: float = 4.0
