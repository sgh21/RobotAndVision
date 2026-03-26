import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from utils import  pnp_eval_core
from config.SystemConfig import Camera, Coordinate
from config.IntrinsicConfig import BOARD_GRID, SQUARE_SIZE
from algorithm.FlangeEstimate import FlangeEstimate 
from controller.Transforms import invT, rtde2T, pose2T

def load_pose_file(pose_file):
    """
    优先读取 T_base_tool；
    其次兼容 tcp_pose_rtde；
    再兼容旧格式 tool_pose_std / tool_poses。
    """
    data = np.load(pose_file, allow_pickle=True)
    filenames = data["filenames"].astype(str)

    if "T_base_tool" in data:
        T_base_tool = np.asarray(data["T_base_tool"], dtype=float)
    elif "tcp_pose_rtde" in data:
        T_base_tool = np.array([rtde2T(p) for p in data["tcp_pose_rtde"]], dtype=float)
    elif "tool_pose_std" in data:
        T_base_tool = np.array([pose2T(p) for p in data["tool_pose_std"]], dtype=float)
    elif "tool_poses" in data:
        T_base_tool = np.array([pose2T(p) for p in data["tool_poses"]], dtype=float)
    else:
        raise KeyError("位姿文件中未找到 T_base_tool / tcp_pose_rtde / tool_pose_std / tool_poses")

    return filenames, T_base_tool

def rot_err_deg(T_est, T_gt):
    R_err = T_gt[:3, :3].T @ T_est[:3, :3]
    rvec, _ = cv2.Rodrigues(R_err)
    return float(np.linalg.norm(np.degrees(rvec.reshape(3))))


def trans_err_mm(T_est, T_gt):
    return float(np.linalg.norm((T_est[:3, 3] - T_gt[:3, 3]) * 1000.0))


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    return {
        "mean": float(vals.mean()) if len(vals) else None,
        "std": float(vals.std()) if len(vals) else None,
        "max": float(vals.max()) if len(vals) else None,
        "min": float(vals.min()) if len(vals) else None,
    }


def run_eval(data_dir, image_dir_name="calibration_images", pose_file_name="robot_poses.npz",
             out_name="flange_pose_eval.json"):
    data_dir = Path(data_dir)
    image_dir = data_dir / image_dir_name
    pose_file = data_dir / pose_file_name
    pnp_solver = pnp_eval_core.estimate_pose_from_image
    flange_estimator = FlangeEstimate(mode="Vision").estimate_T_base_flange

    filenames, T_base_tool_all = load_pose_file(pose_file)

    K = np.asarray(Camera.K, dtype=np.float64)
    D = np.asarray(Camera.D, dtype=np.float64).reshape(-1, 1)
    T_C2T = np.asarray(Coordinate.T_C2T, dtype=float)
    T_P2B = np.asarray(Coordinate.T_P2B, dtype=float)

    results = []
    trans_err_list = []
    rot_err_list = []
    reproj_list = []

    for name, T_base_tool_gt in zip(filenames, T_base_tool_all):
        img_path = image_dir / name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 读图失败，跳过: {img_path}")
            continue

        pose = pnp_solver(
            img,
            BOARD_GRID,
            SQUARE_SIZE,
            K,
            D,
            use_ippe=False,
            refine_lm=True,
        )
        if not pose.success:
            print(f"[WARN] PnP 失败，跳过: {name} | {pose.message}")
            continue

        T_cam_board = np.asarray(pose.T_cam_board, dtype=float)
        T_base_flange_est = flange_estimator(
            T_cam_board=T_cam_board,
            T_C2T=T_C2T,
            T_P2B=T_P2B,
        )

        terr_mm = trans_err_mm(T_base_flange_est, T_base_tool_gt)
        rerr_deg = rot_err_deg(T_base_flange_est, T_base_tool_gt)

        trans_err_list.append(terr_mm)
        rot_err_list.append(rerr_deg)
        reproj_list.append(float(pose.reproj_mean_px))

        results.append({
            "image_name": str(name),
            "reproj_mean_px": float(pose.reproj_mean_px),
            "translation_error_mm": terr_mm,
            "rotation_error_deg": rerr_deg,
            "T_cam_board": T_cam_board.tolist(),
            "T_base_flange_est": T_base_flange_est.tolist(),
            "T_base_flange_gt": np.asarray(T_base_tool_gt, dtype=float).tolist(),
        })

        print(f"[INFO] {name} | reproj={pose.reproj_mean_px:.4f}px | "
              f"terr={terr_mm:.3f}mm | rerr={rerr_deg:.4f}deg")

    summary = {
        "num_total": int(len(filenames)),
        "num_valid": int(len(results)),
        "reprojection_error_px": summarize(reproj_list),
        "translation_error_mm": summarize(trans_err_list),
        "rotation_error_deg": summarize(rot_err_list),
    }

    out = {
        "config": {
            "BOARD_GRID": list(BOARD_GRID),
            "SQUARE_SIZE_mm": float(SQUARE_SIZE),
            "T_C2T_as_T_tool_cam": T_C2T.tolist(),
            "T_P2B_as_T_base_board": T_P2B.tolist(),
        },
        "summary": summary,
        "results": results,
    }

    out_path = data_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n===== 汇总结果 =====")
    print(f"有效样本: {summary['num_valid']}/{summary['num_total']}")
    print("重投影误差 reproj (px):", summary["reprojection_error_px"])
    print("平移误差 translation (mm):", summary["translation_error_mm"])
    print("旋转误差 rotation (deg):", summary["rotation_error_deg"])
    print(f"结果已保存到: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="数据根目录")
    parser.add_argument("--image_dir", default="calibration_images", help="图片子目录名")
    parser.add_argument("--pose_file", default="robot_poses.npz", help="位姿 npz 文件名")
    parser.add_argument("--out", default="flange_pose_eval.json", help="输出 json 文件名")
    args = parser.parse_args()

    run_eval(args.data_dir, args.image_dir, args.pose_file, args.out)
