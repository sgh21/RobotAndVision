from scipy.spatial.transform import Rotation as R
import numpy as np

# ==============================================================================
# 基础旋转转换函数
# ==============================================================================

def rpy2rot(rpy):
    """
    Convert roll-pitch-yaw angles to rotation matrix.
    
    Args:
        rpy: roll-pitch-yaw angles [rad]
    """
    r = R.from_euler('xyz', rpy)
    return r.as_matrix()

def rot2rpy(rot):
    """
    Convert rotation matrix to roll-pitch-yaw angles.
    
    Args:
        rot: rotation matrix
    """
    r = R.from_matrix(rot)
    return r.as_euler('xyz')

def rot2quat(rot):
    """
    Convert rotation matrix to quaternion.
    
    Args:
        rot: rotation matrix
    """
    r = R.from_matrix(rot)
    return r.as_quat()

def quat2rot(quat):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quat: quaternion
    """
    r = R.from_quat(quat)
    return r.as_matrix()

def rpy2quat(rpy):
    """
    Convert roll-pitch-yaw angles to quaternion.
    
    Args:
        rpy: roll-pitch-yaw angles [rad]
    """
    r = R.from_euler('xyz', rpy)
    return r.as_quat()

def quat2rpy(quat):
    """
    Convert quaternion to roll-pitch-yaw angles.
    
    Args:
        quat: quaternion
    """
    r = R.from_quat(quat)
    return r.as_euler('xyz')

def rot2axisangle(rot):
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        rot: rotation matrix
    """
    r = R.from_matrix(rot)
    return r.as_rotvec()

def axisangle2rot(axisangle):
    """
    Convert axis-angle representation to rotation matrix.
    
    Args:
        axisangle: axis-angle representation
    """
    r = R.from_rotvec(axisangle)
    return r.as_matrix()

def quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
    
    Args:
        quat: quaternion
    """
    r = R.from_quat(quat)
    return r.as_rotvec()

def axisangle2quat(axisangle):
    """
    Convert axis-angle representation to quaternion.
    
    Args:
        axisangle: axis-angle representation
    """
    r = R.from_rotvec(axisangle)
    return r.as_quat()

def rpy2axisangle(rpy):
    """
    Convert roll-pitch-yaw angles to axis-angle representation.
    
    Args:
        rpy: roll-pitch-yaw angles [rad]
    """
    r = R.from_euler('xyz', rpy)
    return r.as_rotvec()

def axisangle2rpy(axisangle):
    """
    Convert axis-angle representation to roll-pitch-yaw angles.
    
    Args:
        axisangle: axis-angle representation
    """
    r = R.from_rotvec(axisangle)
    return r.as_euler('xyz')

def rtde2pose(rtde_pose):
    """
    Convert RTDE pose format to position and rpy.
    
    Args:
        rtde_pose: [x, y, z, rx, ry, rz] where rotation is in axis-angle (rad)
    """
    position = rtde_pose[:3]
    rotvec = rtde_pose[3:]
    rpy = axisangle2rpy(rotvec)
    pose = position + rpy.tolist()
    return pose

def pose2rtde(pose):
    """
    Convert position and rpy to RTDE pose format.
    
    Args:
        pose: [x, y, z, roll, pitch, yaw] where rotation is in rpy (rad)
    """
    position = pose[:3]
    rpy = pose[3:]
    rotvec = rpy2axisangle(rpy)
    rtde_pose = position + rotvec.tolist()
    return rtde_pose

def rtde2T(tcp_pose_rtde):
    """
    RTDE TCP位姿 [x, y, z, rx, ry, rz] -> 4x4 齐次矩阵 T_base_tool
    其中 rx,ry,rz 为旋转向量（轴角）
    """
    tcp_pose_rtde = np.asarray(tcp_pose_rtde, dtype=float).reshape(6)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_rotvec(tcp_pose_rtde[3:]).as_matrix()
    T[:3, 3] = tcp_pose_rtde[:3]
    return T

def pose2T(pose):
    """
    标准位姿 [x, y, z, roll, pitch, yaw] -> 4x4 齐次矩阵 T_base_tool
    其中 roll,pitch,yaw 为欧拉角（弧度）
    """
    pose = np.asarray(pose, dtype=float).reshape(6)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_euler("xyz", pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T

def T2pose(T):
    """
    4x4 齐次矩阵 T_base_tool -> 标准位姿 [x, y, z, roll, pitch, yaw]
    其中 roll,pitch,yaw 为欧拉角（弧度）
    """
    T = np.asarray(T, dtype=float)
    position = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz")
    pose = position.tolist() + rpy.tolist()
    return pose

def T2rtde(T):
    """
    4x4 齐次矩阵 T_base_tool -> RTDE TCP位姿 [x, y, z, rx, ry, rz]
    其中 rx,ry,rz 为旋转向量（轴角）
    """
    T = np.asarray(T, dtype=float)
    position = T[:3, 3]
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    rtde_pose = position.tolist() + rotvec.tolist()
    return rtde_pose


def invT(T):
    """4x4 齐次变换求逆。"""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out

# ==============================================================================
# 测试示例
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RTDE位姿 <-> 标准位姿 转换测试")
    print("=" * 60)
    
    # 示例1: RTDE位姿 -> 标准位姿
    print("\n--- RTDE -> 标准位姿 ---")
    pose_rtde = [0.5, 0.2, 0.3, 2.222, 2.222, 0]  # m, rad
    pose_std = rtde2pose(pose_rtde)
    print(f"RTDE格式:  [{', '.join([f'{p:.4f}' for p in pose_rtde])}]")
    print(f"标准位姿:  x={pose_std[0]:.1f}mm, y={pose_std[1]:.1f}mm, z={pose_std[2]:.1f}mm")
    print(f"           roll={pose_std[3]:.2f}°, pitch={pose_std[4]:.2f}°, yaw={pose_std[5]:.2f}°")
    
    # 示例2: 标准位姿 -> RTDE位姿
    print("\n--- 标准位姿 -> RTDE ---")
    pose_std = [500, 200, 300, 0, 0, 180]  # mm, deg
    pose_rtde = pose2rtde(pose_std)
    print(f"标准位姿:  x={pose_std[0]}mm, y={pose_std[1]}mm, z={pose_std[2]}mm")
    print(f"           roll={pose_std[3]}°, pitch={pose_std[4]}°, yaw={pose_std[5]}°")
    print(f"RTDE格式:  [{', '.join([f'{p:.4f}' for p in pose_rtde])}]")
    
    # 示例3: 往返转换验证
    print("\n--- 往返转换验证 ---")
    original = [400, 150, 250, 45/180*np.pi, 30/180*np.pi, 90/180*np.pi]  # mm, rad
    rtde = pose2rtde(original)
    recovered = rtde2pose(rtde)
    print(f"原始:  {[f'{v:.2f}' for v in original]}")
    print(f"恢复:  {[f'{v:.2f}' for v in recovered]}")  