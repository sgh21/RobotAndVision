"""
PnP 评估核心函数。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class PoseEstimate:
    success: bool
    T_cam_board: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    image_points: Optional[np.ndarray] = None
    object_points: Optional[np.ndarray] = None
    projected_points: Optional[np.ndarray] = None
    reproj_each_px: Optional[np.ndarray] = None
    reproj_mean_px: Optional[float] = None
    reproj_rms_px: Optional[float] = None
    message: str = ""


def make_centered_checkerboard_object_points(board_grid, square_size_mm):
    """
    生成棋盘格内角点三维坐标，单位: 米。

    约定：
    - board_grid = (cols, rows) 表示内角点数量。
    - 采用与 OpenCV findChessboardCorners 一致的顺序：
      x 沿列方向递增，y 沿行方向递增。
    - 为减小微小转动误差对平移比较的影响，这里把原点放在棋盘格几何中心。
    """
    cols, rows = board_grid
    square_size_m = float(square_size_mm) / 1000.0

    objp = np.zeros((cols * rows, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m

    center_x = (cols - 1) * square_size_m / 2.0
    center_y = (rows - 1) * square_size_m / 2.0
    objp[:, 0] -= center_x
    objp[:, 1] -= center_y
    return objp


def detect_checkerboard_corners(image_bgr, board_grid, subpix_win=(11, 11), max_iters=50, eps=1e-3):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
    )
    ok, corners = cv2.findChessboardCorners(gray, board_grid, flags)

    if (not ok) or corners is None:
        if hasattr(cv2, "findChessboardCornersSB"):
            ok, corners = cv2.findChessboardCornersSB(gray, board_grid, flags=0)
        if (not ok) or corners is None:
            return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        int(max_iters),
        float(eps),
    )
    corners = cv2.cornerSubPix(
        gray,
        np.asarray(corners, dtype=np.float32),
        winSize=tuple(subpix_win),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return True, np.asarray(corners, dtype=np.float64).reshape(-1, 2)


def _project_and_pack(object_points, image_points, K, dist, rvec, tvec):
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    reproj_each = np.linalg.norm(proj - image_points, axis=1)
    reproj_mean = float(np.mean(reproj_each))
    reproj_rms = float(np.sqrt(np.mean(np.square(reproj_each))))
    return proj, reproj_each, reproj_mean, reproj_rms


def solve_checkerboard_pose(object_points, image_points, K, dist, use_ippe=True, refine_lm=True):
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    candidates = []

    if use_ippe:
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                object_points,
                image_points,
                K,
                dist,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if ok:
                for rvec, tvec in zip(rvecs, tvecs):
                    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
                    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
                    if float(tvec[2]) <= 0.0:
                        continue
                    proj, reproj_each, reproj_mean, reproj_rms = _project_and_pack(
                        object_points, image_points, K, dist, rvec, tvec
                    )
                    candidates.append((reproj_mean, reproj_rms, rvec, tvec, proj, reproj_each))
        except cv2.error:
            pass

    if not candidates:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return PoseEstimate(success=False, message="solvePnP 失败")
        proj, reproj_each, reproj_mean, reproj_rms = _project_and_pack(
            object_points, image_points, K, dist, rvec, tvec
        )
        candidates.append((reproj_mean, reproj_rms, rvec, tvec, proj, reproj_each))

    candidates.sort(key=lambda x: (x[0], x[1]))
    reproj_mean, reproj_rms, rvec, tvec, proj, reproj_each = candidates[0]

    if refine_lm:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                K,
                dist,
                rvec,
                tvec,
            )
            proj, reproj_each, reproj_mean, reproj_rms = _project_and_pack(
                object_points, image_points, K, dist, rvec, tvec
            )
        except cv2.error:
            pass

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)

    return PoseEstimate(
        success=True,
        T_cam_board=T,
        rvec=rvec,
        tvec=tvec,
        image_points=image_points,
        object_points=object_points,
        projected_points=proj,
        reproj_each_px=reproj_each,
        reproj_mean_px=reproj_mean,
        reproj_rms_px=reproj_rms,
        message="ok",
    )


def estimate_pose_from_image(
    image_bgr,
    board_grid,
    square_size_mm,
    K,
    dist,
    use_ippe=True,
    refine_lm=True,
    subpix_win=(11, 11),
    max_iters=50,
    eps=1e-3,
):
    ok, image_points = detect_checkerboard_corners(
        image_bgr,
        board_grid,
        subpix_win=subpix_win,
        max_iters=max_iters,
        eps=eps,
    )
    if not ok:
        return PoseEstimate(success=False, message="未检测到完整棋盘格")

    object_points = make_centered_checkerboard_object_points(board_grid, square_size_mm)
    return solve_checkerboard_pose(
        object_points=object_points,
        image_points=image_points,
        K=K,
        dist=dist,
        use_ippe=use_ippe,
        refine_lm=refine_lm,
    )


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def relative_transform(T_ref, T_cur):
    """返回 T_ref_cur = inv(T_ref) @ T_cur。"""
    return invert_transform(T_ref) @ T_cur


def rotation_angle_deg(R):
    rvec, _ = cv2.Rodrigues(R)
    return float(np.linalg.norm(np.degrees(rvec.reshape(3))))


def translation_metrics_mm(est_vec_m, gt_vec_mm):
    est_vec_m = np.asarray(est_vec_m, dtype=np.float64).reshape(3)
    gt_vec_mm = np.asarray(gt_vec_mm, dtype=np.float64).reshape(3)

    est_vec_mm = est_vec_m * 1000.0
    err_vec_mm = est_vec_mm - gt_vec_mm

    return {
        "est_xyz_mm": est_vec_mm,
        "gt_xyz_mm": gt_vec_mm,
        "err_xyz_mm": err_vec_mm,
        "est_norm_mm": float(np.linalg.norm(est_vec_mm)),
        "gt_norm_mm": float(np.linalg.norm(gt_vec_mm)),
        "err_norm_mm": float(np.linalg.norm(est_vec_mm) - np.linalg.norm(gt_vec_mm)),
        "vec_error_norm_mm": float(np.linalg.norm(err_vec_mm)),
    }


def build_gt_translation_mm(stage_x_mm, stage_y_mm, *, swap_xy=False, sign_x=1.0, sign_y=1.0, fixed_z_mm=0.0):
    x = float(stage_x_mm)
    y = float(stage_y_mm)
    if swap_xy:
        x, y = y, x
    return np.array([sign_x * x, sign_y * y, float(fixed_z_mm)], dtype=np.float64)


def draw_pose_debug(image_bgr, pose: PoseEstimate, board_grid, text_lines=None):
    vis = image_bgr.copy()

    if pose.image_points is not None:
        for p in pose.image_points:
            x, y = np.round(p).astype(int)
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)

    if pose.projected_points is not None:
        for p in pose.projected_points:
            x, y = np.round(p).astype(int)
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)

    if pose.image_points is not None and len(pose.image_points) == board_grid[0] * board_grid[1]:
        cv2.drawChessboardCorners(
            vis,
            board_grid,
            pose.image_points.reshape(-1, 1, 2).astype(np.float32),
            True,
        )

    lines = []
    if pose.reproj_mean_px is not None:
        lines.append(f"reproj_mean = {pose.reproj_mean_px:.4f} px")
    if pose.reproj_rms_px is not None:
        lines.append(f"reproj_rms  = {pose.reproj_rms_px:.4f} px")
    if text_lines:
        lines.extend(text_lines)

    y0 = 30
    for i, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (20, y0 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def summarize_scalar_list(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if arr.size else None,
        "std": float(np.std(arr)) if arr.size else None,
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
    }


def build_summary(step_results):
    if not step_results:
        return {}

    reproj = [x["reproj_mean_px"] for x in step_results]
    trans_dist_err = [x["relative_error"]["err_norm_mm"] for x in step_results]
    trans_vec_err = [x["relative_error"]["vec_error_norm_mm"] for x in step_results]
    rot_rel_deg = [x["relative_rotation_deg"] for x in step_results]
    ex = [x["relative_error"]["err_xyz_mm"][0] for x in step_results]
    ey = [x["relative_error"]["err_xyz_mm"][1] for x in step_results]
    ez = [x["relative_error"]["err_xyz_mm"][2] for x in step_results]

    return {
        "reprojection_mean_px": summarize_scalar_list(reproj),
        "translation_distance_error_mm": summarize_scalar_list(trans_dist_err),
        "translation_vector_error_norm_mm": summarize_scalar_list(trans_vec_err),
        "relative_rotation_deg": summarize_scalar_list(rot_rel_deg),
        "translation_component_error_mm": {
            "x": summarize_scalar_list(ex),
            "y": summarize_scalar_list(ey),
            "z": summarize_scalar_list(ez),
        },
    }
