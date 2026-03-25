"""
    内参标定配置文件，主要用于控制标定板参数、机器人运动范围、采图数量等关键参数的设置。
"""


import numpy as np
import math
PI = math.pi

# ================= 标定板物理参数 =================
BOARD_GRID = (9, 6)           # 棋盘格内角点数量 (列数, 行数)
SQUARE_SIZE = 15.0            # 每个格子的物理边长 (mm)
BOARD_POS = [0.5, 0.0, 0.0]   # 标定板在机器人基坐标系下的物理位置 (x, y, z) 单位: m

# ================= 机器人运动参数 =================
# 初始中心采图位姿 (Standard Pose: x, y, z, roll, pitch, yaw)
# 单位: x,y,z(m), r,p,y(rad)
INIT_POSE = [0.4, 0.0, 0.3, PI, 0.0, 0.0]

# 随机采样的最大偏移量 (dx, dy, dz, droll, dpitch, dyaw)
# 平移范围: ±0.05m (50mm)，旋转范围: ±0.26rad (约15度)
MAX_DEV = [0.05, 0.05, 0.05, PI/12, PI/12, PI/12]

NUM_POSES = 20                # 采图数量
MOVE_VEL = 0.25               # 机器人移动速度 (m/s)
MOVE_ACC = 0.5                # 机器人移动加速度 (m/s^2)

# ================= 文件路径配置 =================
IMAGE_DIR = "./dataset/calibration_images"
OUTPUT_DIR = "./dataset/calibration_results"