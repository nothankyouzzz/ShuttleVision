from .types import (
    CalibrationInput,
    CalibrationResult,
    CameraIntrinsics,
    CameraPose,
)


def estimate_camera_pose(
    calib_input: CalibrationInput,
    initial_intrinsics: CameraIntrinsics | None = None,
) -> CalibrationResult:
    """
    基于 PnP 求解相机位姿：
    - 如果提供 intrinsics，则只估计 R, t
    - 否则可以先用简单 pinhole 模型估计 fx, fy, cx, cy
    """
    ...


def compute_reprojection_error(
    calib_input: CalibrationInput,
    camera_pose: CameraPose,
) -> float:
    """用于校验与报告标定质量。"""
    ...


def save_calibration(result: CalibrationResult, path: str) -> None: ...
def load_calibration(path: str) -> CalibrationResult: ...
