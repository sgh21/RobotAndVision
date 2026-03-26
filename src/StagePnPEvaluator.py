"""
调姿台 + 棋盘格 PnP 精度验证主程序。

运行方式：
    python StagePnPEvaluator.py
"""

import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from controller.MVSControl import MVSController
import config.PoseEvalConfig as cfg
from utils.pnp_eval_core import (
    build_gt_translation_mm,
    build_summary,
    draw_pose_debug,
    estimate_pose_from_image,
    relative_transform,
    rotation_angle_deg,
    translation_metrics_mm,
)


class OpenCVCamera:
    """
    为了最小修改，保留原类名 OpenCVCamera，
    但内部实现改为复用你数据采集程序中的 MVSController 采图逻辑。
    """
    def __init__(self, camera_id=0, width=None, height=None, warmup_frames=10):
        self.cam = MVSController()
        self.warmup_frames = int(max(0, warmup_frames))

    def capture(self):
        frame = None
        # 沿用“预热丢帧”的接口习惯，兼容原配置项
        for _ in range(max(1, self.warmup_frames)):
            frame = self.cam.get_image()
            if frame is None:
                continue

        if frame is None:
            raise RuntimeError("MVS 相机采图失败")
        return frame

    def close(self):
        if self.cam is not None:
            self.cam.close_device()
            self.cam = None

class FolderCamera:
    def __init__(self, image_dir, pattern="*.jpg"):
        self.paths = sorted(Path(image_dir).glob(pattern))
        self.index = 0
        if not self.paths:
            raise FileNotFoundError(f"离线图像目录为空: {image_dir}")

    def capture(self):
        if self.index >= len(self.paths):
            raise RuntimeError("离线图像已经读取完毕")
        path = self.paths[self.index]
        self.index += 1
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"读取图像失败: {path}")
        return img

    def close(self):
        return None


class StagePnPEvaluator:
    def __init__(self, config_module):
        self.cfg = config_module
        self.K = config_module.get_camera_matrix()
        self.dist = config_module.get_dist_coeffs()
        self.board_grid = tuple(config_module.BOARD_GRID)
        self.square_size_mm = float(config_module.SQUARE_SIZE)

        self.output_root = Path(config_module.OUTPUT_ROOT)
        self.raw_dir = self.output_root / "raw_images"
        self.debug_dir = self.output_root / "debug_images"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.camera = self._create_camera()
        self.initial_pose = None
        self.initial_image_name = None
        self.records = []
        self.current_stage_xy_mm = np.zeros(2, dtype=np.float64)

    def _create_camera(self):
        if self.cfg.CAPTURE_SOURCE == "opencv":
            # 为最小改动，仍沿用 "opencv" 这个配置名，
            # 但实际内部已切换为 MVSController 采图。
            return OpenCVCamera(
                camera_id=self.cfg.CAMERA_ID,
                width=self.cfg.CAMERA_WIDTH,
                height=self.cfg.CAMERA_HEIGHT,
                warmup_frames=self.cfg.CAPTURE_WARMUP_FRAMES,
            )
        if self.cfg.CAPTURE_SOURCE == "folder":
            return FolderCamera(
                image_dir=self.cfg.OFFLINE_IMAGE_DIR,
                pattern=self.cfg.OFFLINE_IMAGE_GLOB,
            )
        raise ValueError(f"未知 CAPTURE_SOURCE: {self.cfg.CAPTURE_SOURCE}")

    def _capture_and_estimate(self, image_tag, extra_text_lines=None):
        img = self.camera.capture()

        if self.cfg.SAVE_RAW_IMAGES:
            raw_path = self.raw_dir / f"{image_tag}.jpg"
            cv2.imwrite(str(raw_path), img)
        else:
            raw_path = None

        if self.cfg.SAVE_UNDISTORTED_IMAGES:
            undist = cv2.undistort(img, self.K, self.dist)
            cv2.imwrite(str(self.debug_dir / f"{image_tag}_undist.jpg"), undist)

        pose = estimate_pose_from_image(
            image_bgr=img,
            board_grid=self.board_grid,
            square_size_mm=self.square_size_mm,
            K=self.K,
            dist=self.dist,
            use_ippe=self.cfg.USE_IPPE,
            refine_lm=self.cfg.ENABLE_REFINE_LM,
            subpix_win=self.cfg.CORNER_SUBPIX_WIN,
            max_iters=self.cfg.CORNER_SUBPIX_MAX_ITERS,
            eps=self.cfg.CORNER_SUBPIX_EPS,
        )

        debug_path = None
        if pose.success and self.cfg.SAVE_DEBUG_IMAGES:
            vis = draw_pose_debug(img, pose, self.board_grid, text_lines=extra_text_lines)
            debug_path = self.debug_dir / f"{image_tag}_debug.jpg"
            cv2.imwrite(str(debug_path), vis)

        return img, raw_path, debug_path, pose

    def _prompt_init(self):
        while True:
            user_in = input(f"\n{self.cfg.PROMPT_INIT}\n>> ").strip()
            if user_in.lower() == "q":
                return False

            _, raw_path, debug_path, pose = self._capture_and_estimate("step_000_init")
            if not pose.success:
                print(f"[WARN] 初始图像 PnP 失败: {pose.message}，请调整后重试。")
                continue

            self.initial_pose = pose
            self.initial_image_name = raw_path.name if raw_path is not None else "step_000_init.jpg"
            record = {
                "step_idx": 0,
                "image_name": self.initial_image_name,
                "debug_image": str(debug_path) if debug_path is not None else None,
                "type": "initial",
                "reproj_mean_px": float(pose.reproj_mean_px),
                "reproj_rms_px": float(pose.reproj_rms_px),
                "T_cam_board": pose.T_cam_board.tolist(),
                "stage_gt_xyz_mm": [0.0, 0.0, float(self.cfg.STAGE_FIXED_Z_MM)],
            }
            self.records.append(record)
            print(f"[INFO] 初始图像采集成功，reproj_mean = {pose.reproj_mean_px:.4f} px")
            return True

    def _parse_stage_input(self, text):
        parts = text.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError("请输入两个数字: x y")
        x_mm = float(parts[0])
        y_mm = float(parts[1])
        return x_mm, y_mm

    def _update_stage_state(self, x_mm, y_mm):
        if self.cfg.STAGE_INPUT_MODE == "absolute":
            self.current_stage_xy_mm = np.array([x_mm, y_mm], dtype=np.float64)
        elif self.cfg.STAGE_INPUT_MODE == "incremental":
            self.current_stage_xy_mm += np.array([x_mm, y_mm], dtype=np.float64)
        else:
            raise ValueError(f"未知 STAGE_INPUT_MODE: {self.cfg.STAGE_INPUT_MODE}")
        return self.current_stage_xy_mm.copy()

    def _evaluate_step(self, step_idx, stage_abs_xy_mm):
        stage_gt_xyz_mm = build_gt_translation_mm(
            stage_abs_xy_mm[0],
            stage_abs_xy_mm[1],
            swap_xy=self.cfg.STAGE_SWAP_XY,
            sign_x=self.cfg.STAGE_SIGN_X,
            sign_y=self.cfg.STAGE_SIGN_Y,
            fixed_z_mm=self.cfg.STAGE_FIXED_Z_MM,
        )

        text_lines = [
            f"gt = [{stage_gt_xyz_mm[0]:.3f}, {stage_gt_xyz_mm[1]:.3f}, {stage_gt_xyz_mm[2]:.3f}] mm",
        ]
        _, raw_path, debug_path, pose = self._capture_and_estimate(f"step_{step_idx:03d}", text_lines)

        if not pose.success:
            raise RuntimeError(f"步骤 {step_idx} 的 PnP 失败: {pose.message}")

        T_b0_bi = relative_transform(self.initial_pose.T_cam_board, pose.T_cam_board)
        rel_t_m = T_b0_bi[:3, 3]
        rel_rot_deg = rotation_angle_deg(T_b0_bi[:3, :3])
        rel_metrics = translation_metrics_mm(rel_t_m, stage_gt_xyz_mm)

        record = {
            "step_idx": int(step_idx),
            "image_name": raw_path.name if raw_path is not None else f"step_{step_idx:03d}.jpg",
            "debug_image": str(debug_path) if debug_path is not None else None,
            "type": "measurement",
            "reproj_mean_px": float(pose.reproj_mean_px),
            "reproj_rms_px": float(pose.reproj_rms_px),
            "T_cam_board": pose.T_cam_board.tolist(),
            "T_board0_boardi": T_b0_bi.tolist(),
            "relative_rotation_deg": float(rel_rot_deg),
            "relative_error": {
                "est_xyz_mm": rel_metrics["est_xyz_mm"].tolist(),
                "gt_xyz_mm": rel_metrics["gt_xyz_mm"].tolist(),
                "err_xyz_mm": rel_metrics["err_xyz_mm"].tolist(),
                "est_norm_mm": float(rel_metrics["est_norm_mm"]),
                "gt_norm_mm": float(rel_metrics["gt_norm_mm"]),
                "err_norm_mm": float(rel_metrics["err_norm_mm"]),
                "vec_error_norm_mm": float(rel_metrics["vec_error_norm_mm"]),
            },
        }
        self.records.append(record)

        print(
            f"[STEP {step_idx:03d}] reproj={record['reproj_mean_px']:.4f}px | "
            f"est(mm)=[{record['relative_error']['est_xyz_mm'][0]:.3f}, "
            f"{record['relative_error']['est_xyz_mm'][1]:.3f}, "
            f"{record['relative_error']['est_xyz_mm'][2]:.3f}] | "
            f"gt(mm)=[{record['relative_error']['gt_xyz_mm'][0]:.3f}, "
            f"{record['relative_error']['gt_xyz_mm'][1]:.3f}, "
            f"{record['relative_error']['gt_xyz_mm'][2]:.3f}] | "
            f"|est|-|gt|={record['relative_error']['err_norm_mm']:.3f}mm | "
            f"vec_err={record['relative_error']['vec_error_norm_mm']:.3f}mm | "
            f"rot={record['relative_rotation_deg']:.4f}deg"
        )
        if record["relative_rotation_deg"] > float(self.cfg.WARN_ROTATION_DEG):
            print(
                f"[WARN] 相对旋转达到 {record['relative_rotation_deg']:.4f} deg，"
                f"若调姿台理论上只做平移，请检查装夹/检测稳定性。"
            )

    def _save_csv(self):
        csv_path = self.output_root / "step_metrics.csv"
        fields = [
            "step_idx",
            "image_name",
            "reproj_mean_px",
            "reproj_rms_px",
            "relative_rotation_deg",
            "gt_x_mm",
            "gt_y_mm",
            "gt_z_mm",
            "est_x_mm",
            "est_y_mm",
            "est_z_mm",
            "err_x_mm",
            "err_y_mm",
            "err_z_mm",
            "gt_norm_mm",
            "est_norm_mm",
            "err_norm_mm",
            "vec_error_norm_mm",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self.records:
                if r["type"] != "measurement":
                    continue
                rel = r["relative_error"]
                writer.writerow(
                    {
                        "step_idx": r["step_idx"],
                        "image_name": r["image_name"],
                        "reproj_mean_px": r["reproj_mean_px"],
                        "reproj_rms_px": r["reproj_rms_px"],
                        "relative_rotation_deg": r["relative_rotation_deg"],
                        "gt_x_mm": rel["gt_xyz_mm"][0],
                        "gt_y_mm": rel["gt_xyz_mm"][1],
                        "gt_z_mm": rel["gt_xyz_mm"][2],
                        "est_x_mm": rel["est_xyz_mm"][0],
                        "est_y_mm": rel["est_xyz_mm"][1],
                        "est_z_mm": rel["est_xyz_mm"][2],
                        "err_x_mm": rel["err_xyz_mm"][0],
                        "err_y_mm": rel["err_xyz_mm"][1],
                        "err_z_mm": rel["err_xyz_mm"][2],
                        "gt_norm_mm": rel["gt_norm_mm"],
                        "est_norm_mm": rel["est_norm_mm"],
                        "err_norm_mm": rel["err_norm_mm"],
                        "vec_error_norm_mm": rel["vec_error_norm_mm"],
                    }
                )
        return csv_path

    def _save_plots(self):
        steps = [r["step_idx"] for r in self.records if r["type"] == "measurement"]
        if not steps:
            return []

        outputs = []
        plot_specs = [
            (
                "reprojection_error_px.png",
                "Reprojection Error (px)",
                [r["reproj_mean_px"] for r in self.records if r["type"] == "measurement"],
                "pixels",
            ),
            (
                "translation_distance_error_mm.png",
                "Translation Distance Error (mm)",
                [r["relative_error"]["err_norm_mm"] for r in self.records if r["type"] == "measurement"],
                "mm",
            ),
            (
                "translation_vector_error_norm_mm.png",
                "Translation Vector Error Norm (mm)",
                [r["relative_error"]["vec_error_norm_mm"] for r in self.records if r["type"] == "measurement"],
                "mm",
            ),
            (
                "relative_rotation_deg.png",
                "Relative Rotation (deg)",
                [r["relative_rotation_deg"] for r in self.records if r["type"] == "measurement"],
                "deg",
            ),
        ]

        for name, title, values, ylabel in plot_specs:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, marker="o")
            plt.title(title)
            plt.xlabel("Step")
            plt.ylabel(ylabel)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            path = self.output_root / name
            plt.savefig(str(path), dpi=150)
            plt.close()
            outputs.append(path)
        return outputs

    def _save_json(self):
        step_results = [r for r in self.records if r["type"] == "measurement"]
        summary = build_summary(step_results)
        payload = {
            "config": {
                "board_grid": list(self.board_grid),
                "square_size_mm": self.square_size_mm,
                "capture_source": self.cfg.CAPTURE_SOURCE,
                "stage_input_mode": self.cfg.STAGE_INPUT_MODE,
                "stage_swap_xy": bool(self.cfg.STAGE_SWAP_XY),
                "stage_sign_x": float(self.cfg.STAGE_SIGN_X),
                "stage_sign_y": float(self.cfg.STAGE_SIGN_Y),
                "stage_fixed_z_mm": float(self.cfg.STAGE_FIXED_Z_MM),
            },
            "initial": next((r for r in self.records if r["type"] == "initial"), None),
            "steps": step_results,
            "summary": summary,
        }
        json_path = self.output_root / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return json_path, summary

    def run(self):
        t0 = time.time()
        try:
            ok = self._prompt_init()
            if not ok:
                print("[INFO] 用户终止。")
                return

            step_idx = 1
            while step_idx <= int(self.cfg.NUM_STEPS):
                user_in = input(f"\n{self.cfg.PROMPT_STEP}\n>> ").strip()
                if user_in.lower() == "q":
                    break

                try:
                    x_mm, y_mm = self._parse_stage_input(user_in)
                    stage_abs_xy_mm = self._update_stage_state(x_mm, y_mm)
                    self._evaluate_step(step_idx, stage_abs_xy_mm)
                    step_idx += 1
                except Exception as exc:
                    print(f"[WARN] 当前步骤失败：{exc}")
                    print("[INFO] 本次结果不会计入；请调整后重新输入。")
                    continue

            csv_path = self._save_csv()
            json_path, summary = self._save_json()
            plot_paths = self._save_plots()

            print("\n==================== 实验结束 ====================")
            print(f"结果 JSON : {json_path}")
            print(f"结果 CSV  : {csv_path}")
            if plot_paths:
                print("曲线图:")
                for p in plot_paths:
                    print(f"  - {p}")
            if summary:
                print("\n汇总统计:")
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"总耗时: {time.time() - t0:.2f} s")
        finally:
            self.camera.close()


if __name__ == "__main__":
    evaluator = StagePnPEvaluator(cfg)
    evaluator.run()
