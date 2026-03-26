import os
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from controller.URController import UR10Controller
from controller.MVSControl import MVSController
from controller.Transforms import pose2rtde, rtde2pose, rtde2T


class DataCollector:
    def __init__(self, config):
        self.cfg = config
        self.robot = None
        self.cam = None
        self.pose_records = []

        dataset_dir = os.path.dirname(self.cfg.IMAGE_DIR.rstrip("/\\")) or "."
        self.pose_file = getattr(
            self.cfg,
            "POSE_RECORD_FILE",
            os.path.join(dataset_dir, "robot_poses.npz")
        )

        if getattr(self.cfg, "RANDOM_SEED", None) is not None:
            np.random.seed(self.cfg.RANDOM_SEED)

        os.makedirs(self.cfg.IMAGE_DIR, exist_ok=True)

    def connect_hardware(self):
        """连接机械臂和相机"""
        print("[INFO] 正在连接硬件...")
        self.robot = UR10Controller()
        if not self.robot.connect():
            raise ConnectionError("机器人连接失败！")
        
        self.robot.set_tcp([0, 0, 0, 0, 0, 0])

        self.cam = MVSController()
        print("[INFO] 硬件连接完成。")

    def disconnect_hardware(self):
        """断开硬件连接"""
        if self.robot:
            self.robot.disconnect()
        if self.cam:
            self.cam.close_device()
        print("[INFO] 硬件已断开。")

    def _move_robot_to_pose(self, pose):
        """机器人移动到给定 Standard Pose"""
        rtde_pose = pose2rtde(pose)
        self.robot.move_l(rtde_pose, vel=self.cfg.MOVE_VEL, acc=self.cfg.MOVE_ACC)
        self.robot.wait_until_steady(1)
        time.sleep(getattr(self.cfg, "SETTLE_TIME", 0.5))


    def _save_pose_records(self):
        """保存图片名 - 末端位姿数据对，统一为手眼标定可直接读取的格式"""
        if len(self.pose_records) == 0:
            print("[WARN] 没有可保存的位姿记录。")
            return

        pose_dir = os.path.dirname(self.pose_file)
        if pose_dir:
            os.makedirs(pose_dir, exist_ok=True)

        np.savez(
            self.pose_file,
            filenames=np.asarray([r["filename"] for r in self.pose_records]),
            tcp_pose_rtde=np.asarray([r["tcp_pose_rtde"] for r in self.pose_records], dtype=float),
            tool_pose_std=np.asarray([r["tool_pose_std"] for r in self.pose_records], dtype=float),
            T_base_tool=np.asarray([r["T_base_tool"] for r in self.pose_records], dtype=float),
        )
        print(f"[OK] 末端位姿已保存: {self.pose_file}")

    def _capture_and_save(self, img_idx):
        """采图并保存，同时记录当前实际TCP位姿"""
        # 机器人已稳定后读取实际TCP位姿（RTDE原始格式）

        tcp_pose_rtde = np.asarray(
            self.robot.get_tcp_pose(filtered=False),
            dtype=float
        )
        # print(f"[INFO] 当前TCP位姿 (RTDE格式): {tcp_pose_rtde.tolist()}")
        image = self.cam.get_image()
        if image is None:
            print(f"[WARN] 第 {img_idx} 张图采集失败！")
            return False

        img_name = f"calib_{img_idx:02d}.jpg"
        img_path = os.path.join(self.cfg.IMAGE_DIR, img_name)
        cv2.imwrite(img_path, image)

        self.pose_records.append({
            "filename": img_name,
            "tcp_pose_rtde": tcp_pose_rtde,
            "tool_pose_std": np.asarray(rtde2pose(tcp_pose_rtde.tolist()), dtype=float),
            "T_base_tool": np.asarray(rtde2T(tcp_pose_rtde)),
        })

        print(f"[OK] 图像已保存: {img_path}")
        return True

    # def _capture_and_save(self, img_idx):
    #     """采图并保存"""
    #     image = self.cam.get_image()
    #     if image is None:
    #         print(f"[WARN] 第 {img_idx} 张图采集失败！")
    #         return False

    #     img_path = os.path.join(self.cfg.IMAGE_DIR, f"calib_{img_idx:02d}.jpg")
    #     cv2.imwrite(img_path, image)
    #     print(f"[OK] 图像已保存: {img_path}")
    #     return True

    def _get_init_pose_info(self):
        """
        从初始位姿中提取关键几何信息。

        说明：
        - INIT_POSE 表示机器人末端位姿，而不是相机光心位姿。
        - CAM_OFFSET 表示相机光心相对末端坐标系原点的固定偏置。
        - 程序默认 INIT_POSE 已经满足：标定板位于视野中心、相机光轴基本垂直标定板平面。
        - 后续所有随机姿态均围绕该参考姿态做局部角度扰动。
        """
        init_pose = np.asarray(self.cfg.INIT_POSE, dtype=float)
        board_pos = np.asarray(self.cfg.BOARD_POS, dtype=float)
        cam_offset = np.asarray(getattr(self.cfg, "CAM_OFFSET", [0.0, 0.0, 0.0]), dtype=float)

        init_tool_pos = init_pose[:3]
        init_rpy = init_pose[3:]
        R_init = R.from_euler('xyz', init_rpy).as_matrix()

        # 初始相机光心位置 = 末端位置 + 旋转后的固定偏置
        init_cam_pos = init_tool_pos + R_init @ cam_offset

        cam_to_board = board_pos - init_cam_pos
        work_dist = np.linalg.norm(cam_to_board)
        if work_dist < 1e-9:
            raise ValueError("初始相机光心与 BOARD_POS 重合，无法定义观察方向！")

        view_dir_world = cam_to_board / work_dist

        # 光轴在工具坐标系下的方向：由初始姿态反推得到
        optical_axis_local = R_init.T @ view_dir_world
        optical_axis_local = optical_axis_local / np.linalg.norm(optical_axis_local)

        return {
            "init_pose": init_pose,
            "init_tool_pos": init_tool_pos,
            "init_cam_pos": init_cam_pos,
            "init_rpy": init_rpy,
            "R_init": R_init,
            "board_pos": board_pos,
            "cam_offset": cam_offset,
            "work_dist": work_dist,
            "optical_axis_local": optical_axis_local,
        }

    def move_to_initial_pose(self):
        """先移动到配置文件中的初始位姿"""
        print("[INFO] 正在移动到初始位姿 ...")
        self._move_robot_to_pose(self.cfg.INIT_POSE)
        print("[OK] 已到达初始位姿。")

    def generate_random_poses(self):
        """
        生成满足以下约束的随机位姿：
        1. 以 INIT_POSE 作为参考中心姿态；
        2. 在当前姿态基础上，对末端局部 rpy 做随机扰动；
        3. 使用“相机光轴继续看向标定板中心 + 相机工作距离保持初始值不变”反推相机位置；
        4. 再由相机固定偏置反推机器人末端位置；
        5. 最终使用 MAX_DEV[:3] 对末端位置做安全边界检查。
        """
        info = self._get_init_pose_info()
        init_tool_pos = info["init_tool_pos"]
        R_init = info["R_init"]
        board_pos = info["board_pos"]
        cam_offset = info["cam_offset"]
        work_dist = info["work_dist"]
        optical_axis_local = info["optical_axis_local"]

        max_pos_dev = np.asarray(self.cfg.MAX_DEV[:3], dtype=float)
        max_ang_dev = np.asarray(self.cfg.MAX_DEV[3:], dtype=float)
        max_attempts = getattr(self.cfg, "MAX_SAMPLE_ATTEMPTS", 2000)

        poses = []
        attempts = 0

        print(f"[INFO] 初始相机工作距离: {work_dist:.4f} m")
        print(f"[INFO] 相机固定偏置 CAM_OFFSET: {cam_offset.tolist()} m")
        print("[INFO] 开始生成随机位姿 ...")

        while len(poses) < self.cfg.NUM_POSES and attempts < max_attempts:
            attempts += 1

            # 在初始姿态基础上施加局部 RPY 扰动
            d_rpy = np.random.uniform(-max_ang_dev, max_ang_dev)
            R_delta = R.from_euler('xyz', d_rpy).as_matrix()
            R_new = R_init @ R_delta

            # 新姿态下，相机光轴在世界坐标系中的方向
            view_dir_world = R_new @ optical_axis_local
            view_dir_world = view_dir_world / np.linalg.norm(view_dir_world)

            # 先根据“相机看向标定板中心 + 工作距离不变”反推出相机光心位置
            cam_pos = board_pos - work_dist * view_dir_world

            # 再由相机固定偏置反推出机器人末端位置
            tool_pos = cam_pos - R_new @ cam_offset

            # 用末端平移阈值做安全边界约束
            if np.any(np.abs(tool_pos - init_tool_pos) > max_pos_dev):
                continue

            pose = tool_pos.tolist() + R.from_matrix(R_new).as_euler('xyz').tolist()
            poses.append(pose)

        if len(poses) < self.cfg.NUM_POSES:
            print(f"[WARN] 尝试 {attempts} 次，仅生成 {len(poses)} 个有效位姿。")
            print("[WARN] 可适当增大 MAX_DEV 的角度范围，或放宽 MAX_DEV[:3] 的末端平移安全边界。")
        else:
            print(f"[OK] 成功生成 {len(poses)} 个随机位姿。")

        return poses

    def execute_collection(self):
        """执行全自动采图流程"""
        # 1) 先回到初始位姿
        self.move_to_initial_pose()

        img_idx = 0

        # 2) 是否先采集一张初始位姿图像
        if getattr(self.cfg, "CAPTURE_INIT_IMAGE", True):
            print("[INFO] 采集初始位姿图像 ...")
            self._capture_and_save(img_idx)
            img_idx += 1

        # 3) 生成随机位姿并依次采集
        poses = self.generate_random_poses()
        for pose_id, pose in enumerate(poses, start=1):
            print(f"\n[INFO] 正在移动至随机采集点 {pose_id}/{len(poses)} ...")
            self._move_robot_to_pose(pose)
            self._capture_and_save(img_idx)
            img_idx += 1
        
        self._save_pose_records()

        print("[OK] 采集流程结束。")
