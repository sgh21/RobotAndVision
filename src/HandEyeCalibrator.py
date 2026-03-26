import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from config.IntrinsicConfig import BOARD_GRID, SQUARE_SIZE, K, D
from algorithm.PnP import estimate_pose_from_image
from controller.Transforms import rtde2T, pose2T, invT


def mean_T(T_list):
    Rs = np.stack([T[:3, :3] for T in T_list], axis=0)
    ts = np.stack([T[:3, 3] for T in T_list], axis=0)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_matrix(Rs).mean().as_matrix()
    T[:3, 3] = ts.mean(axis=0)
    return T


def rot_err_deg(R_ref, R_cur):
    return float(np.degrees((R.from_matrix(R_ref.T @ R_cur)).magnitude()))


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


def pose_consistency_metrics(T_list):
    """
    把“理论上应一致”的一组齐次变换，转成易理解的：
    1) 平移一致性
    2) 旋转一致性
    """
    T_mean = mean_T(T_list)
    ts = np.stack([T[:3, 3] for T in T_list], axis=0)
    dxyz_mm = (ts - T_mean[:3, 3]) * 1000.0
    terr_mm = np.linalg.norm(dxyz_mm, axis=1)
    rerr_deg = np.array([rot_err_deg(T_mean[:3, :3], T[:3, :3]) for T in T_list], dtype=float)

    return T_mean, {
        "translation_xyz_std_mm": (ts.std(axis=0) * 1000.0).tolist(),
        "translation_to_mean_mm_mean": float(np.mean(terr_mm)),
        "translation_to_mean_mm_std": float(np.std(terr_mm)),
        "translation_to_mean_mm_max": float(np.max(terr_mm)),
        "rotation_to_mean_deg_mean": float(np.mean(rerr_deg)),
        "rotation_to_mean_deg_std": float(np.std(rerr_deg)),
        "rotation_to_mean_deg_max": float(np.max(rerr_deg)),
    }


def axxb_residual_metrics(T_base_tool_list, T_cam_board_list, T_tool_cam):
    """
    直接检查 AX=XB 在所有样本对上的残差一致性，
    并转成更容易理解的平移/旋转残差指标。
    """
    terr_mm, rerr_deg = [], []
    n = len(T_base_tool_list)

    for i in range(n):
        for j in range(i + 1, n):
            A = invT(T_base_tool_list[i]) @ T_base_tool_list[j]
            B = T_cam_board_list[i] @ invT(T_cam_board_list[j])
            E = invT(A @ T_tool_cam) @ (T_tool_cam @ B)
            terr_mm.append(float(np.linalg.norm(E[:3, 3]) * 1000.0))
            rerr_deg.append(rot_err_deg(np.eye(3), E[:3, :3]))

    terr_mm = np.asarray(terr_mm, dtype=float)
    rerr_deg = np.asarray(rerr_deg, dtype=float)

    return {
        "pair_count": int(len(terr_mm)),
        "translation_mm_mean": float(np.mean(terr_mm)),
        "translation_mm_std": float(np.std(terr_mm)),
        "translation_mm_max": float(np.max(terr_mm)),
        "rotation_deg_mean": float(np.mean(rerr_deg)),
        "rotation_deg_std": float(np.std(rerr_deg)),
        "rotation_deg_max": float(np.max(rerr_deg)),
    }


def run_handeye(data_dir, image_dir_name="images", pose_file_name="robot_poses.npz",
                out_name="handeye_result.json", method_name="TSAI"):
    data_dir = Path(data_dir)
    image_dir = data_dir / image_dir_name
    pose_file = data_dir / pose_file_name

    filenames, T_base_tool_all = load_pose_file(pose_file)

    K_np = np.asarray(K, dtype=np.float64)
    D_np = np.asarray(D, dtype=np.float64).reshape(-1, 1)

    method_map = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    method = method_map[method_name.upper()]

    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []
    valid_names, reproj_means, T_cam_board_all = [], [], []

    for name, T_base_tool in zip(filenames, T_base_tool_all):
        img_path = image_dir / name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 读图失败，跳过: {img_path}")
            continue

        pose = estimate_pose_from_image(
            img, BOARD_GRID, SQUARE_SIZE, K_np, D_np,
            use_ippe=False, refine_lm=True,
        )
        if not pose.success:
            print(f"[WARN] PnP 失败，跳过: {name} | {pose.message}")
            continue

        T_cam_board = np.asarray(pose.T_cam_board, dtype=float)
        print(f"[INFO] {name} | PnP 平均重投影误差: {pose.reproj_mean_px:.4f} px")
        print(f"       T_cam_board:\n{np.array2string(T_cam_board, precision=6, suppress_small=True)}")
        T_cam_board_all.append(T_cam_board)
        valid_names.append(str(name))
        reproj_means.append(float(pose.reproj_mean_px))

        # OpenCV calibrateHandEye:
        #   R_gripper2base / t_gripper2base  <- T_base_tool
        #   R_target2cam   / t_target2cam    <- T_cam_board
        R_g2b.append(T_base_tool[:3, :3])
        t_g2b.append(T_base_tool[:3, 3])
        R_t2c.append(T_cam_board[:3, :3])
        t_t2c.append(T_cam_board[:3, 3])

    if len(R_g2b) < 3:
        raise RuntimeError(f"有效样本不足，当前只有 {len(R_g2b)} 组，至少需要 3 组，建议 >= 10 组。")

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=method
    )

    T_tool_cam = np.eye(4, dtype=float)
    T_tool_cam[:3, :3] = R_cam2gripper
    T_tool_cam[:3, 3] = np.asarray(t_cam2gripper, dtype=float).reshape(3)

    name_to_T = {str(n): T for n, T in zip(filenames, T_base_tool_all)}
    T_base_tool_valid = [name_to_T[name] for name in valid_names]
    T_base_board_list = [
        T_base_tool @ T_tool_cam @ T_cam_board
        for T_base_tool, T_cam_board in zip(T_base_tool_valid, T_cam_board_all)
    ]

    T_base_board, board_consistency = pose_consistency_metrics(T_base_board_list)
    axxb_consistency = axxb_residual_metrics(T_base_tool_valid, T_cam_board_all, T_tool_cam)

    result = {
        "method": method_name.upper(),
        "num_total": int(len(filenames)),
        "num_valid": int(len(valid_names)),
        "used_images": valid_names,
        "reproj_mean_px_mean": float(np.mean(reproj_means)),
        "reproj_mean_px_std": float(np.std(reproj_means)),
        "T_tool_cam": T_tool_cam.tolist(),
        "T_base_board": T_base_board.tolist(),
        "board_pose_consistency": board_consistency,
        "axxb_residual_consistency": axxb_consistency,
    }

    out_path = data_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== 手眼标定结果 =====")
    print(f"方法: {method_name.upper()}")
    print("T_tool_cam (相机在末端):")
    print(np.array2string(T_tool_cam, precision=6, suppress_small=True))
    print("\nT_base_board (标定板在基坐标系):")
    print(np.array2string(T_base_board, precision=6, suppress_small=True))
    print(f"\n有效样本: {len(valid_names)}/{len(filenames)}")
    print(f"PnP 平均重投影误差: {np.mean(reproj_means):.4f} ± {np.std(reproj_means):.4f} px")

    print("\n--- 标定板在基坐标系下的一致性（越小越好）---")
    print(f"平移一致性 xyz std (mm): {board_consistency['translation_xyz_std_mm']}")
    print(f"平移一致性 到均值距离 (mm): "
          f"{board_consistency['translation_to_mean_mm_mean']:.3f} ± "
          f"{board_consistency['translation_to_mean_mm_std']:.3f}, "
          f"max={board_consistency['translation_to_mean_mm_max']:.3f}")
    print(f"旋转一致性 到均值角差 (deg): "
          f"{board_consistency['rotation_to_mean_deg_mean']:.4f} ± "
          f"{board_consistency['rotation_to_mean_deg_std']:.4f}, "
          f"max={board_consistency['rotation_to_mean_deg_max']:.4f}")

    print("\n--- AX=XB 运动残差一致性（越小越好）---")
    print(f"样本对数量: {axxb_consistency['pair_count']}")
    print(f"平移残差 (mm): "
          f"{axxb_consistency['translation_mm_mean']:.3f} ± "
          f"{axxb_consistency['translation_mm_std']:.3f}, "
          f"max={axxb_consistency['translation_mm_max']:.3f}")
    print(f"旋转残差 (deg): "
          f"{axxb_consistency['rotation_deg_mean']:.4f} ± "
          f"{axxb_consistency['rotation_deg_std']:.4f}, "
          f"max={axxb_consistency['rotation_deg_max']:.4f}")

    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="数据根目录，里面包含图片文件夹和 robot_poses.npz")
    parser.add_argument("--image_dir", default="calibration_images", help="图片子目录名，默认 images")
    parser.add_argument("--pose_file", default="robot_poses.npz", help="末端位姿文件名，默认 robot_poses.npz")
    parser.add_argument("--out", default="handeye_result.json", help="输出结果文件名")
    parser.add_argument("--method", default="TSAI",
                        choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
                        help="OpenCV 手眼标定方法")
    args = parser.parse_args()
    run_handeye(args.data_dir, args.image_dir, args.pose_file, args.out, args.method)
