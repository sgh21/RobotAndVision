import numpy as np
from controller.Transforms import invT
from config.SystemConfig import Coordinate

class FlangeEstimate:
    def __init__(self, mode = "Vision"):
        self.mode = mode
        if self.mode == "Vision":
            self.T_C2T = Coordinate.T_C2T
            self.T_P2B = Coordinate.T_P2B
    
    def estimate_T_base_flange(self, T_cam_board, T_C2T = None, T_P2B = None):
        """
        根据相机观测到的标定板位姿，估计法兰在机器人基坐标系下的位姿。

        约定：
        - T_cam_board  = ^C T_P  ，PnP 输出，表示标定板在相机坐标系下
        - T_tool_cam   = ^F T_C  ，相机外参，表示相机在法兰坐标系下
        - T_base_board = ^B T_P  ，标定板标定结果，表示标定板在基坐标系下

        则有：
        ^B T_F = ^B T_P * (^C T_P)^(-1) * (^F T_C)^(-1)
        """
        T_cam_board = np.asarray(T_cam_board, dtype=float)
        T_tool_cam = np.asarray(T_C2T if T_C2T is not None else self.T_C2T, dtype=float)
        T_base_board = np.asarray(T_P2B if T_P2B is not None else self.T_P2B, dtype=float)
        return T_base_board @ invT(T_cam_board) @ invT(T_tool_cam)

