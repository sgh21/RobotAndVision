import os
import time
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from controller.URController import UR10Controller
from controller.MVSControl import MVSController
from controller.Transforms import pose2rtde


class DataCollector:
    def __init__(self, config):
        self.cfg = config
        self.robot = None
        self.cam = None
        
        # 确保图片保存目录存在
        os.makedirs(self.cfg.IMAGE_DIR, exist_ok=True)

    def connect_hardware(self):
        """连接机械臂和相机"""
        print("[INFO] 正在连接硬件...")
        self.robot = UR10Controller()
        if not self.robot.connect():
            raise ConnectionError("机器人连接失败！")
            
        self.cam = MVSController()
        print("[INFO] 硬件连接完成。")

    def disconnect_hardware(self):
        """断开硬件连接"""
        if self.robot:
            self.robot.disconnect()
        if self.cam:
            self.cam.close_device()
        print("[INFO] 硬件已断开。")

    def _spherical_to_cartesian(self, r, az, el):
        """
        将球面坐标转换为笛卡尔坐标的相对偏移量
        az: 方位角 (Azimuth)
        el: 俯仰角 (Elevation)
        """
        x = r * math.cos(el) * math.cos(az)
        y = r * math.cos(el) * math.sin(az)
        z = r * math.sin(el)
        return np.array([x, y, z])

    def _look_at(self, cam_pos, target_pos, roll_rad):
        """
        计算 Look-At 位姿，使得工具 Z 轴正对 target_pos，并应用光轴的 Roll 旋转
        """
        # 计算 Z 轴向量 (从相机指向目标)
        z_vec = target_pos - cam_pos
        z_norm = np.linalg.norm(z_vec)
        if z_norm < 1e-6:
            z_vec = np.array([0.0, 0.0, -1.0])
        else:
            z_vec = z_vec / z_norm
            
        # 确定参考 X 轴 (假设基坐标系 X 轴为水平参考)
        ref_x = np.array([1.0, 0.0, 0.0])
        y_vec = np.cross(z_vec, ref_x)
        y_norm = np.linalg.norm(y_vec)
        
        # 奇点处理：如果相机正下视，Z 向量与 X 向量平行
        if y_norm < 1e-6:
            ref_y = np.array([0.0, -1.0, 0.0])
            x_vec = np.cross(ref_y, z_vec)
            x_vec = x_vec / np.linalg.norm(x_vec)
            y_vec = np.cross(z_vec, x_vec)
        else:
            y_vec = y_vec / y_norm
            x_vec = np.cross(y_vec, z_vec)
            
        # 构建基础旋转矩阵 R = [X, Y, Z]
        R_base = np.column_stack((x_vec, y_vec, z_vec))
        
        # 应用绕相机局部 Z 轴的 roll 旋转
        r_roll = R.from_euler('z', roll_rad).as_matrix()
        R_final = R_base @ r_roll
        
        # 转换回欧拉角 RPY (符合 Transforms.py 的标准格式约定)
        rpy = R.from_matrix(R_final).as_euler('xyz').tolist()
        
        return cam_pos.tolist() + rpy

    def generate_random_poses(self):
        """基于球面坐标采样和 Look-At 逻辑生成安全且正对标定板的随机位姿"""
        poses = []
        target_world = np.array(self.cfg.BOARD_POS)
        init_pos = np.array(self.cfg.INIT_POSE[:3])
        
        # 1. 提取初始状态作为基准
        V0 = init_pos - target_world
        r0 = np.linalg.norm(V0)
        az0 = math.atan2(V0[1], V0[0])
        el0 = math.asin(V0[2] / r0) if r0 > 0 else 0
        
        # 2. 安全限制：以 MAX_DEV 为硬性边界
        cart_limit = np.array(self.cfg.MAX_DEV[:3])
        roll_limit = self.cfg.MAX_DEV[3]
        
        print(f"[INFO] 开始生成位姿，标定板距离基准: {r0:.3f}m ...")
        
        max_attempts = 1000
        attempts = 0
        
        while len(poses) < self.cfg.NUM_POSES and attempts < max_attempts:
            attempts += 1
            
            # --- 步骤 A：球面坐标域内的随机游走 ---
            # 距离扰动 ±0.05m
            r = r0 + np.random.uniform(-0.05, 0.05)
            # 方位角 ±30° (提供左右偏角)
            az = az0 + np.random.uniform(-math.pi/6, math.pi/6)
            # 俯仰角 ±15° (提供高低偏角)
            el = el0 + np.random.uniform(-math.pi/12, math.pi/12)
            
            cam_pos = target_world + self._spherical_to_cartesian(r, az, el)
            
            # --- 步骤 B：安全墙拦截 (Cartesian 边界) ---
            # 任何超出了 MAX_DEV 平移允许范围的点，直接丢弃重新生成
            if np.any(np.abs(cam_pos - init_pos) > cart_limit):
                continue
                
            # --- 步骤 C：目标点 Jitter 抖动 ---
            # 避免相机永远机械地死盯着中心点，给目标点加一点 2cm 级别的噪音
            jitter = np.random.uniform(-0.02, 0.02, size=3)
            look_pt = target_world + jitter
            
            # --- 步骤 D：绕光轴自转 (Roll) ---
            roll_rad = np.random.uniform(-roll_limit, roll_limit)
            
            # 最终合成位姿
            pose = self._look_at(cam_pos, look_pt, roll_rad)
            poses.append(pose)
            
        if len(poses) < self.cfg.NUM_POSES:
            print(f"[WARN] 安全限制触发！尝试了 {max_attempts} 次，仅成功生成 {len(poses)} 个安全位姿。")
            print("[WARN] 如果数量太少，请尝试在 IntrinsicConfig.py 中适当放宽 MAX_DEV 范围。")
        else:
            print(f"[OK] 成功生成 {len(poses)} 个满足安全限制的正对姿态！")
            
        return poses

    def execute_collection(self):
        """执行全自动采图流程"""
        poses = self.generate_random_poses()
        
        for idx, pose in enumerate(poses):
            print(f"[INFO] 正在移动至采集点 {idx+1}/{len(poses)} ...")
            
            # 将 Standard Pose 转换为 RTDE 格式
            rtde_pose = pose2rtde(pose)
            
            # 机器人运动
            self.robot.move_l(rtde_pose, vel=self.cfg.MOVE_VEL, acc=self.cfg.MOVE_ACC)
            self.robot.wait_until_steady()
            
            # 增加短暂延时，确保机械臂完全静止且相机画面无残影
            time.sleep(0.5)
            
            # 采集图像并保存
            image = self.cam.get_image()
            if image is not None:
                img_path = os.path.join(self.cfg.IMAGE_DIR, f"calib_{idx:02d}.jpg")
                cv2.imwrite(img_path, image)
                print(f"[OK] 图像已保存: {img_path}")
            else:
                print(f"[WARN] 第 {idx+1} 个位姿采图失败！")