import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .types import (
    CalibrationConfig,
    CalibrationError,
    CalibrationInput,
    CalibrationResult,
    CameraIntrinsics,
    CameraPose,
)

logger = logging.getLogger(__name__)


def _build_camera_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Construct a 3x3 pinhole camera matrix from intrinsics."""
    return np.array(
        [
            [intrinsics.fx_px, intrinsics.skew, intrinsics.cx_px],
            [0.0, intrinsics.fy_px, intrinsics.cy_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _build_dist_coeffs(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Pack optional distortion parameters into OpenCV style vector."""
    coeffs = [
        intrinsics.k1 or 0.0,
        intrinsics.k2 or 0.0,
        intrinsics.p1 or 0.0,
        intrinsics.p2 or 0.0,
        intrinsics.k3 or 0.0,
    ]
    return np.asarray(coeffs, dtype=float)


def _default_intrinsics(calib_input: CalibrationInput) -> CameraIntrinsics:
    """Create a coarse pinhole intrinsics guess based on image size."""
    focal_guess = float(max(calib_input.image_width_px, calib_input.image_height_px))
    return CameraIntrinsics(
        fx_px=focal_guess,
        fy_px=focal_guess,
        cx_px=calib_input.image_width_px / 2.0,
        cy_px=calib_input.image_height_px / 2.0,
    )


def estimate_camera_pose(
    calib_input: CalibrationInput,
    initial_intrinsics: CameraIntrinsics | None = None,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """
    Solve camera pose via PnP:
    - If intrinsics are provided, only estimate R and t
    - Otherwise use a resolution-based pinhole guess as initial intrinsics
    - Start with RANSAC for a robust seed, then refine iteratively
    """
    cfg = config or CalibrationConfig()
    pts_img = np.asarray(calib_input.points_image_px, dtype=float)
    pts_world = np.asarray(calib_input.points_world_m, dtype=float)

    if pts_img.shape[0] < 4 or pts_world.shape[0] < 4:
        raise CalibrationError("Not enough points for PnP (need >= 4)")
    if pts_img.shape[0] != pts_world.shape[0]:
        raise CalibrationError("Mismatch between image and world point counts")

    intrinsics = initial_intrinsics or _default_intrinsics(calib_input)
    camera_matrix = _build_camera_matrix(intrinsics)
    dist_coeffs = _build_dist_coeffs(intrinsics)

    object_points = pts_world.reshape(-1, 3)
    image_points = pts_img.reshape(-1, 2)

    # Robust initial pose estimate using minimal-set P3P inside RANSAC
    try:
        success_ransac, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            reprojectionError=cfg.ransac_reprojection_error_px,
            confidence=cfg.ransac_confidence,
            flags=cv2.SOLVEPNP_AP3P,
        )
    except (
        cv2.error
    ) as exc:  # pragma: no cover - OpenCV errors are environment specific
        raise CalibrationError(f"solvePnP failed: {exc}") from exc

    if not success_ransac:
        raise CalibrationError("solvePnPRansac failed to find a valid solution")

    if inliers is not None and len(inliers) > 0:
        inlier_idx = inliers[:, 0].astype(int)
        inlier_obj = np.take(object_points, inlier_idx, axis=0)
        inlier_img = np.take(image_points, inlier_idx, axis=0)
    else:
        inlier_obj = object_points
        inlier_img = image_points

    # Refine pose with all inliers using iterative Gauss-Newton
    try:
        success_refine, rvec, tvec = cv2.solvePnP(
            inlier_obj,
            inlier_img,
            camera_matrix,
            dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as exc:  # pragma: no cover
        raise CalibrationError(f"solvePnP refine failed: {exc}") from exc

    if not success_refine:
        raise CalibrationError("solvePnP refinement failed to find a valid solution")

    R_wc, _ = cv2.Rodrigues(rvec)
    t_wc_m = tvec.reshape(3)

    camera_pose = CameraPose(R_wc=R_wc, t_wc_m=t_wc_m, intrinsics=intrinsics)
    reproj_err = compute_reprojection_error(calib_input, camera_pose)

    if reproj_err > cfg.reprojection_error_threshold_px:
        raise CalibrationError(
            (
                "Reprojection error too large: "
                f"{reproj_err:.3f}px (threshold={cfg.reprojection_error_threshold_px}px)"
            )
        )

    logger.debug(
        "Calibration completed: reprojection_error_px=%.4f, num_points=%d",
        reproj_err,
        pts_img.shape[0],
    )

    return CalibrationResult(
        camera_pose=camera_pose,
        reprojection_error_px=reproj_err,
        num_points=pts_img.shape[0],
    )


def compute_reprojection_error(
    calib_input: CalibrationInput,
    camera_pose: CameraPose,
) -> float:
    """Compute mean reprojection error for the current pose and intrinsics."""
    pts_img = np.asarray(calib_input.points_image_px, dtype=float)
    pts_world = np.asarray(calib_input.points_world_m, dtype=float)
    if pts_img.size == 0:
        return 0.0

    R_wc = camera_pose.R_wc
    t_wc = camera_pose.t_wc_m.reshape(3, 1)
    intr = camera_pose.intrinsics

    pts_world_cam = (R_wc @ pts_world.T) + t_wc  # shape=(3, N)
    z = pts_world_cam[2]
    if np.any(np.isclose(z, 0.0)):
        raise CalibrationError("Point projects to z=0 in camera frame")

    x = pts_world_cam[0] / z
    y = pts_world_cam[1] / z

    u_proj = intr.fx_px * x + intr.skew * y + intr.cx_px
    v_proj = intr.fy_px * y + intr.cy_px
    proj = np.vstack([u_proj, v_proj]).T  # shape=(N,2)

    err = np.linalg.norm(proj - pts_img, axis=1)
    return float(err.mean())


def save_calibration(result: CalibrationResult, path: str) -> None:
    """Persist calibration result as JSON for later reuse."""
    data = {
        "intrinsics": {
            "fx_px": result.camera_pose.intrinsics.fx_px,
            "fy_px": result.camera_pose.intrinsics.fy_px,
            "cx_px": result.camera_pose.intrinsics.cx_px,
            "cy_px": result.camera_pose.intrinsics.cy_px,
            "skew": result.camera_pose.intrinsics.skew,
            "k1": result.camera_pose.intrinsics.k1,
            "k2": result.camera_pose.intrinsics.k2,
            "p1": result.camera_pose.intrinsics.p1,
            "p2": result.camera_pose.intrinsics.p2,
            "k3": result.camera_pose.intrinsics.k3,
        },
        "R_wc": np.asarray(result.camera_pose.R_wc, dtype=float).tolist(),
        "t_wc_m": np.asarray(result.camera_pose.t_wc_m, dtype=float).tolist(),
        "reprojection_error_px": result.reprojection_error_px,
        "num_points": result.num_points,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: str) -> CalibrationResult:
    """Load calibration result from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    intr_data = data["intrinsics"]
    intrinsics = CameraIntrinsics(
        fx_px=intr_data["fx_px"],
        fy_px=intr_data["fy_px"],
        cx_px=intr_data["cx_px"],
        cy_px=intr_data["cy_px"],
        skew=intr_data.get("skew", 0.0),
        k1=intr_data.get("k1"),
        k2=intr_data.get("k2"),
        p1=intr_data.get("p1"),
        p2=intr_data.get("p2"),
        k3=intr_data.get("k3"),
    )
    camera_pose = CameraPose(
        R_wc=np.asarray(data["R_wc"], dtype=float),
        t_wc_m=np.asarray(data["t_wc_m"], dtype=float),
        intrinsics=intrinsics,
    )
    return CalibrationResult(
        camera_pose=camera_pose,
        reprojection_error_px=float(data.get("reprojection_error_px", 0.0)),
        num_points=int(data.get("num_points", 0)),
    )
