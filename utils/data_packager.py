import os
import glob
import pickle
from tqdm import tqdm
from pathlib import Path

import cv2
import numpy as np

from algorithm.PnP import estimate_pose_from_image
from algorithm.FlangeEstimate import FlangeEstimate
from controller.Transforms import invT, T2pose
from utils.FileLoader import load_npz
from config.SystemConfig import Coordinate, Camera


def _find_image(image_dir, name):
    name = Path(str(name)).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        p = os.path.join(image_dir, name + ext)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(image_dir, name + ".*"))
    return hits[0] if hits else None


def _pose_error(T_teach, T_est):
    T_err = invT(T_teach) @ T_est
    t_mm = T_err[:3, 3] * 1000.0
    rvec, _ = cv2.Rodrigues(T_err[:3, :3])
    r_deg = float(np.linalg.norm(rvec.reshape(3)) * 180.0 / np.pi)
    return {
        "T_error": T_err,
        "translation_error_mm": t_mm,
        "translation_error_norm_mm": float(np.linalg.norm(t_mm)),
        "rotation_error_deg": r_deg,
    }


def pack_dataset(
    file_pairs,
    K,
    dist,
    board_grid,
    square_size_mm,
    T_C2T=None,
    T_P2B=None,
):
    """
    file_pairs: [(pose_npz_path, image_dir), ...]
    输出 pkl: list[dict]
    """
    flange_estimator = FlangeEstimate(mode="Vision")

    for pose_npz_path, image_dir in file_pairs:
        packed = []
        filenames, T_base_tool_teach_all, joint_all = load_npz(pose_npz_path)

        for img_name, q, T_teach in tqdm(zip(filenames, joint_all, T_base_tool_teach_all), desc=f"Processing {os.path.basename(pose_npz_path)}", total=len(filenames)):
        # for img_name, q, T_teach in zip(filenames, joint_all, T_base_tool_teach_all):
            img_path = _find_image(image_dir, img_name)
            if img_path is None:
                print(f"[skip] 找不到图片: {img_name} @ {image_dir}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"[skip] 图片读取失败: {img_path}")
                continue

            pose = estimate_pose_from_image(
                image_bgr=image,
                board_grid=board_grid,
                square_size_mm=square_size_mm,
                K=K,
                dist=dist,
            )
            if not pose.success:
                print(f"[skip] PnP失败: {img_path}, msg={pose.message}")
                continue

            T_base_tool_est = flange_estimator.estimate_T_base_flange(
                T_cam_board=pose.T_cam_board,
                T_C2T=T_C2T,
                T_P2B=T_P2B,
            )
            err = _pose_error(T_teach, T_base_tool_est)
            
            trans_thr_mm = 5.0
            rot_thr_deg = 5.0

            if err["translation_error_norm_mm"] > trans_thr_mm or err["rotation_error_deg"] > rot_thr_deg:
                print(f"[reject] {img_path} | trans={err['translation_error_norm_mm']:.2f} mm, "
                    f"rot={err['rotation_error_deg']:.2f} deg")
                continue

            sample_idx = int(img_name.split('.')[0].split('_')[-1])
            fk_pose = T2pose(T_teach)
            flange_pos_estimated = T2pose(T_base_tool_est)[:3]  # 只保留位置部分
            
            packed.append({
                "sample_idx": sample_idx,
                "joints": np.asarray(q, dtype=float),
                "cmd_pose": np.asarray(fk_pose, dtype=float),
                "end_pose": np.asarray(fk_pose, dtype=float),
                "fk_pose": np.asarray(fk_pose, dtype=float),
                "flange_pos_estimated": np.asarray(flange_pos_estimated, dtype=float),
                "payload": 0.0,  # 这里暂时没有有效载荷数据，先填0
                "speed_mode": "full",  # 这里暂时没有速度模式数据，先填full
                "image_name": os.path.basename(img_path),
                "pose_error": err,
                "pnp_reproj_mean_px": pose.reproj_mean_px,
                "pnp_reproj_rms_px": pose.reproj_rms_px,
            })
        pkl_name = os.path.basename(pose_npz_path).replace(".npz", ".pkl")
        save_pkl_path = os.path.join(root_dir, pkl_name)

        with open(save_pkl_path, "wb") as f:
            pickle.dump(packed, f)

        if packed:
            trans_err = np.array([x["pose_error"]["translation_error_norm_mm"] for x in packed], dtype=float)
            rot_err = np.array([x["pose_error"]["rotation_error_deg"] for x in packed], dtype=float)
            print(f"done: {len(packed)} samples")
            print(f"translation error [mm]: mean={trans_err.mean():.3f}, std={trans_err.std():.3f}")
            print(f"rotation error [deg]: mean={rot_err.mean():.3f}, std={rot_err.std():.3f}")
        else:
            print("done: 0 valid samples")

    return True


if __name__ == "__main__":
    root_dir = "./dataset/data_GPR_0326"
    file_pairs = [
        # (位姿npz, 对应图片文件夹)
        (os.path.join(root_dir, "robot_poses_normal.npz"), os.path.join(root_dir, "normal_images")),
        # (os.path.join(root_dir, "robot_poses_error_x05.npz"), os.path.join(root_dir, "error_images_x05")),
        # (os.path.join(root_dir, "robot_poses_error_x05y05.npz"), os.path.join(root_dir, "error_images_x05y05")),
    ]

    K = np.asarray(Camera.K, dtype=float)
    dist = np.asarray(Camera.D, dtype=float).reshape(-1, 1)

    board_grid = Camera.BOARD_GRID
    square_size_mm = Camera.SQUARE_SIZE_MM

    pack_dataset(
        file_pairs=file_pairs,
        K=K,
        dist=dist,
        board_grid=board_grid,
        square_size_mm=square_size_mm,

    )
