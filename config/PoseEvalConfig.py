"""
PnP 位姿精度验证配置。

说明：
1) 直接复用 IntrinsicConfig.py 中的棋盘格参数、相机内参和畸变系数。
2) 本文件只补充“验证流程”相关参数。
3) 程序默认采用控制台交互：你输入调姿台相对初始位置的 x/y 位移（mm），程序拍照并评估。
"""

from config.IntrinsicConfig import BOARD_GRID, SQUARE_SIZE, K, D

# ============================================================
# 1) 数据源配置
# ============================================================
# 可选: "opencv" | "folder"
# - opencv: 直接通过 cv2.VideoCapture 采集图像
# - folder: 从目录中顺序读取图像，便于离线调试
CAPTURE_SOURCE = "opencv"

# 当 CAPTURE_SOURCE = "opencv" 时使用
CAMERA_ID = 0
CAMERA_WIDTH = None          # 例如 2448；None 表示不主动设置
CAMERA_HEIGHT = None         # 例如 2048；None 表示不主动设置
CAPTURE_WARMUP_FRAMES = 10   # 正式拍照前丢弃若干帧，使曝光更稳定

# 当 CAPTURE_SOURCE = "folder" 时使用
OFFLINE_IMAGE_DIR = "./dataset/pnp_eval_input"
OFFLINE_IMAGE_GLOB = "*.jpg"

# ============================================================
# 2) 实验流程配置
# ============================================================
NUM_STEPS = 10               # 最多评估多少次（不含初始图像）

# 用户输入模式：
# - "absolute"   : 每次输入“相对初始位置”的绝对位移 dx dy（mm）
# - "incremental": 每次输入“相对上一步”的增量位移 dx dy（mm），程序内部自动累加
STAGE_INPUT_MODE = "absolute"

# 如果调姿台的轴线方向与你定义的棋盘格坐标方向不一致，可在这里做简单映射。
# 棋盘格坐标系采用 OpenCV 常见定义：
# x 沿棋盘格列方向，y 沿棋盘格行方向，z 垂直棋盘格平面。
STAGE_SWAP_XY = False
STAGE_SIGN_X = 1.0
STAGE_SIGN_Y = 1.0
STAGE_FIXED_Z_MM = 0.0       # 若理论上纯平移且无 z 向移动，保持 0 即可

# 若你的调姿台运动理论上不引入转动，则相对旋转应接近单位阵。
# 该项只用于结果提示，不参与计算。
WARN_ROTATION_DEG = 1.0

# ============================================================
# 3) PnP / 角点检测参数
# ============================================================
USE_IPPE = True
ENABLE_REFINE_LM = True
CORNER_SUBPIX_WIN = (11, 11)
CORNER_SUBPIX_MAX_ITERS = 50
CORNER_SUBPIX_EPS = 1e-3

# ============================================================
# 4) 输出配置
# ============================================================
OUTPUT_ROOT = "./dataset/pnp_eval_results"
SAVE_RAW_IMAGES = True
SAVE_DEBUG_IMAGES = True
SAVE_UNDISTORTED_IMAGES = False

# ============================================================
# 5) 交互文案
# ============================================================
PROMPT_INIT = (
    "请保持机器人位姿不动，将标定板放到初始位置后按回车采集初始图像；"
    "输入 q 退出。"
)
PROMPT_STEP = (
    "请输入当前标定板的 stage x y 位移（单位 mm）。\n"
    "- 若 STAGE_INPUT_MODE='absolute'：输入相对初始位置的绝对位移\n"
    "- 若 STAGE_INPUT_MODE='incremental'：输入相对上一步的增量位移\n"
    "示例：2.0 -1.5\n"
    "输入 q 结束实验。"
)

# ============================================================
# 6) 内参/畸变转换
# ============================================================
def get_camera_matrix():
    import numpy as np
    return np.asarray(K, dtype=np.float64)


def get_dist_coeffs():
    import numpy as np
    return np.asarray(D, dtype=np.float64).reshape(-1, 1)
