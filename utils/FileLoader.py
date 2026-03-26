import os
import numpy as np
from controller.Transforms import rtde2T, pose2T


def load_pose_file(file_path):
    """
    优先读取 T_base_tool；
    其次兼容 tcp_pose_rtde；
    再兼容旧格式 tool_pose_std / tool_poses。
    """
    data = np.load(file_path, allow_pickle=True)
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

def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    filenames, T_base_tool = load_pose_file(npz_path)
    if "robot_joint" in  data:
        joints = np.asarray(data["robot_joint"], dtype=float) 
    return filenames, T_base_tool, joints