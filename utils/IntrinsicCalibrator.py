import os
import glob
import cv2
import numpy as np

class IntrinsicCalibrator:
    def __init__(self, config):
        self.cfg = config
        self.objpoints = [] # 3D 物理点集合
        self.imgpoints = [] # 2D 像素点集合
        self.img_size = None
        
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        
        # 生成标准棋盘格的物理坐标 (Z=0)
        w, h = self.cfg.BOARD_GRID
        self.objp = np.zeros((w * h, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * self.cfg.SQUARE_SIZE

        # 亚像素角点搜索的终止条件: 最大迭代30次或精度达到0.001
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def extract_corners(self):
        """读取图像并提取棋盘格角点"""
        images = glob.glob(os.path.join(self.cfg.IMAGE_DIR, '*.jpg'))
        if not images:
            raise FileNotFoundError(f"在 {self.cfg.IMAGE_DIR} 中未找到图像！")

        print(f"[INFO] 开始提取角点，共找到 {len(images)} 张图像...")
        
        valid_images = 0
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.img_size is None:
                self.img_size = gray.shape[::-1] # (width, height)

            # 寻找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, self.cfg.BOARD_GRID, None)

            if ret:
                valid_images += 1
                self.objpoints.append(self.objp)
                
                # 亚像素级精确化
                corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners_subpix)

                # 绘制角点并保存到输出目录
                cv2.drawChessboardCorners(img, self.cfg.BOARD_GRID, corners_subpix, ret)
                base_name = os.path.basename(fname)
                cv2.imwrite(os.path.join(self.cfg.OUTPUT_DIR, f"reproj_{base_name}"), img)
            else:
                print(f"[WARN] 图像 {fname} 未检测到完整棋盘格。")

        print(f"[INFO] 角点提取完成。有效图像数量: {valid_images}/{len(images)}")
        return valid_images > 0

    def calibrate_and_evaluate(self):
        """执行相机标定并计算重投影误差"""
        if not self.objpoints:
            print("[ERROR] 没有足够的有效角点数据进行标定。")
            return

        print("[INFO] 正在计算相机内参...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.img_size, None, None
        )

        print("\n" + "="*50)
        print("                 内参标定结果")
        print("="*50)
        print(f"RMS 误差 (整体): {ret:.4f} pixels")
        print("相机内参矩阵 (Camera Matrix):")
        print(np.round(mtx, 4))
        print("畸变系数 (Distortion Coefficients):")
        print(np.round(dist, 4))
        print("="*50)

        self._calculate_reprojection_error(mtx, dist, rvecs, tvecs)

    def _calculate_reprojection_error(self, mtx, dist, rvecs, tvecs):
        """计算并打印每张图像的重投影误差"""
        total_error = 0
        total_points = 0
        
        print("\n[INFO] 各图像重投影误差分析:")
        for i in range(len(self.objpoints)):
            # 将物理 3D 点重新投影到 2D 像素平面
            imgpoints_projected, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            
            # 计算 L2 范数误差
            error = cv2.norm(self.imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
            total_error += error
            total_points += 1
            print(f"  - 图像 {i+1} 误差: {error:.4f} pixels")
            
        mean_error = total_error / total_points
        print(f"==> 平均重投影误差: {mean_error:.4f} pixels")