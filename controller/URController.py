"""
UR10机器人控制器
基于RTDE协议的规范化封装

依赖:
    - ur_rtde: pip install ur_rtde
    - numpy: pip install numpy

使用示例:
    from UR10Controller.URController import UR10Controller
    
    robot = UR10Controller()
    robot.connect()
    robot.move_j([0, -1.57, 0, -1.57, 0, 0])
    robot.disconnect()
"""

import time
import math
import numpy as np
from typing import List, Optional, Tuple, Union
from enum import Enum

import rtde_receive
import rtde_control
from config.RobotConfig import *
from controller.Transforms import pose2rtde, rtde2pose

class RobotMode(Enum):
    """机器人模式枚举"""
    DISCONNECTED = -1
    CONFIRM_SAFETY = 1
    BOOTING = 2
    POWER_OFF = 3
    POWER_ON = 4
    IDLE = 5
    BACKDRIVE = 6
    RUNNING = 7


class SafetyMode(Enum):
    """安全模式枚举"""
    NORMAL = 1
    REDUCED = 2
    PROTECTIVE_STOP = 3
    RECOVERY = 4
    SAFEGUARD_STOP = 5
    SYSTEM_EMERGENCY_STOP = 6
    ROBOT_EMERGENCY_STOP = 7
    VIOLATION = 8
    FAULT = 9


class UR10Controller:
    """
    UR10机器人控制器类
    
    基于RTDE协议实现对UR10机器人的控制，提供:
    - 关节空间运动控制 (MoveJ)
    - 笛卡尔空间运动控制 (MoveL, MoveP)
    - 实时伺服控制 (ServoJ)
    - 机器人状态读取
    - 速度控制模式
    
    Attributes:
        ip (str): 机器人IP地址
        rtde_frequency (int): RTDE通信频率
        is_connected (bool): 连接状态
    """
    
    def __init__(self, 
                 ip: str = None,
                 rtde_frequency: int = None):
        """
        初始化UR10控制器
        
        Args:
            ip: 机器人IP地址，默认从配置文件读取
            rtde_frequency: RTDE通信频率(Hz)，可选125/250/500，默认从配置读取
        """
        # 连接参数
        self.ip = ip or ROBOT_IP
        self.rtde_frequency = rtde_frequency or RTDE_FREQUENCY
        
        # RTDE接口
        self._rtde_r: Optional[rtde_receive.RTDEReceiveInterface] = None
        self._rtde_c: Optional[rtde_control.RTDEControlInterface] = None
        
        # 连接状态
        self._connected = False
        
        # 默认运动参数
        self._default_joint_vel = DEFAULT_JOINT_VEL
        self._default_joint_acc = DEFAULT_JOINT_ACC
        self._default_linear_vel = DEFAULT_LINEAR_VEL
        self._default_linear_acc = DEFAULT_LINEAR_ACC
        self._default_blend = DEFAULT_BLEND_RADIUS
    
    # ==========================================================================
    # 连接管理
    # ==========================================================================
    
    def connect(self) -> bool:
        """
        连接到UR10机器人
        
        Returns:
            bool: 连接成功返回True，失败返回False
            
        Raises:
            RuntimeError: 连接过程中发生错误
        """
        if self._connected:
            print(f"[INFO] 已经连接到机器人: {self.ip}")
            return True
        
        try:
            print(f"[INFO] 正在连接机器人: {self.ip} ...")
            
            # 初始化接收接口
            self._rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
            print("[INFO] RTDE Receive 接口已连接")
            
            # 初始化控制接口 - 修改端口号为30004，与ur_control.py一致
            self._rtde_c = rtde_control.RTDEControlInterface(
                self.ip
                # self.rtde_frequency,
                # rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP,
                # 50002  
            )
            print("[INFO] RTDE Control 接口已连接")
            
            # 验证连接状态
            if not (self._rtde_r.isConnected() and self._rtde_c.isConnected()):
                raise ConnectionError("RTDE接口未成功连接")
            
            self._connected = True
            print(f"[OK] 成功连接到UR10机器人: {self.ip}")
            
            # 打印机器人状态
            self._print_robot_info()
            
            return True
            
        except Exception as e:
            self._connected = False
            self._rtde_r = None
            self._rtde_c = None
            print(f"[ERROR] 连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开与机器人的连接"""
        try:
            if self._rtde_c:
                self._rtde_c.disconnect()
            if self._rtde_r:
                self._rtde_r.disconnect()
            self._connected = False
            print("[INFO] 已断开机器人连接")
        except Exception as e:
            print(f"[WARNING] 断开连接时发生错误: {e}")
    
    def reconnect(self) -> bool:
        """
        重新连接机器人
        
        Returns:
            bool: 重连成功返回True
        """
        try:
            if self._rtde_r:
                self._rtde_r.reconnect()
            if self._rtde_c:
                self._rtde_c.reconnect()
            self._connected = True
            print("[INFO] 重新连接成功")
            return True
        except Exception as e:
            print(f"[ERROR] 重连失败: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        if not self._connected:
            return False
        try:
            return self._rtde_c.isConnected() and self._rtde_r.isConnected()
        except:
            return False
    
    def _check_connection(self):
        """检查连接状态，未连接则抛出异常"""
        if not self.is_connected:
            raise ConnectionError("机器人未连接，请先调用connect()方法")
    
    def _print_robot_info(self):
        """打印机器人基本信息"""
        try:
            mode = self.get_robot_mode()
            safety = self.get_safety_mode()
            joints = self.get_joint_positions()
            joints_deg = [math.degrees(j) for j in joints]
            
            print("\n" + "=" * 50)
            print("           UR10 机器人状态")
            print("=" * 50)
            print(f"  IP地址:     {self.ip}")
            print(f"  RTDE频率:   {self.rtde_frequency} Hz")
            print(f"  机器人模式: {mode}")
            print(f"  安全模式:   {safety}")
            print(f"  关节角度(°): [{', '.join([f'{j:.1f}' for j in joints_deg])}]")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"[WARNING] 无法获取机器人信息: {e}")
    
    # ==========================================================================
    # 状态读取
    # ==========================================================================
    
    def _sample_with_filter(self, getter, samples: int = 5, interval: float = 0.002) -> List[float]:
        """
        多次采样并取均值的辅助函数（均值滤波）
        
        Args:
            getter: 数据获取函数
            samples: 采样次数，默认5次
            interval: 采样间隔，单位: s，默认2ms
            
        Returns:
            List[float]: 滤波后的数据
        """
        samples = max(1, int(samples))
        data = []
        for i in range(samples):
            data.append(getter())
            if i < samples - 1 and interval > 0:
                time.sleep(interval)
        arr = np.array(data, dtype=float)
        return arr.mean(axis=0).tolist()
    def get_joint_positions(self, filtered: bool = True, samples: int = 5) -> List[float]:
        """
        获取当前关节角度
        
        Args:
            filtered: 是否进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            List[float]: 6个关节角度 [j1, j2, j3, j4, j5, j6]，单位: rad
        """
        self._check_connection()
        if filtered:
            return self._sample_with_filter(lambda: list(self._rtde_r.getActualQ()), samples)
        return list(self._rtde_r.getActualQ())
    
    def get_joint_positions_deg(self, filtered: bool = True, samples: int = 5) -> List[float]:
        """
        获取当前关节角度（角度制）
        
        Args:
            filtered: 是否进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            List[float]: 6个关节角度 [j1, j2, j3, j4, j5, j6]，单位: 度(°)
        """
        return [math.degrees(j) for j in self.get_joint_positions(filtered=filtered, samples=samples)]
    
    def get_joint_velocities(self, filtered: bool = True, samples: int = 5) -> List[float]:
        """
        获取当前关节速度
        
        Args:
            filtered: 是否进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            List[float]: 6个关节速度，单位: rad/s
        """
        self._check_connection()
        if filtered:
            return self._sample_with_filter(lambda: list(self._rtde_r.getActualQd()), samples)
        return list(self._rtde_r.getActualQd())
    
    def get_joint_currents(self, filtered: bool = True, samples: int = 5) -> List[float]:
        """
        获取当前关节电流
        
        Args:
            filtered: 是否进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            List[float]: 6个关节电流，单位: A
        """
        self._check_connection()
        if filtered:
            return self._sample_with_filter(lambda: list(self._rtde_r.getActualCurrent()), samples)
        return list(self._rtde_r.getActualCurrent())
    
    def get_joint_torques(self, filtered: bool = True, samples: int = 5) -> List[float]:
        """
        获取当前关节力矩
        
        Args:
            filtered: 是否进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            List[float]: 6个关节力矩，单位: Nm
        """
        self._check_connection()
        if filtered:
            return self._sample_with_filter(lambda: list(self._rtde_r.getTargetMoment()), samples)
        return list(self._rtde_r.getTargetMoment())
    
    def get_tcp_pose(self, filtered: bool = True, samples: int = 5) -> List[float]:
        """
        获取当前TCP位姿（基坐标系下）
        
        Args:
            filtered: 是否进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            List[float]: TCP位姿 [x, y, z, rx, ry, rz]
                - x, y, z: 位置，单位: m
                - rx, ry, rz: 旋转向量(轴角表示)，单位: rad
        """
        self._check_connection()
        if filtered:
            return self._sample_with_filter(lambda: list(self._rtde_r.getActualTCPPose()), samples)
        return list(self._rtde_r.getActualTCPPose())
    
    def get_tcp_position(self, filtered: bool = True, samples: int = 5) -> List[float]:
        """
        获取当前TCP位置
        
        Args:
            filtered: 是否进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            List[float]: [x, y, z]，单位: m
        """
        return self.get_tcp_pose(filtered=filtered, samples=samples)[:3]
    
    def get_tcp_orientation(self, filtered: bool = True, samples: int = 5) -> List[float]:
        """
        获取当前TCP姿态（旋转向量表示）
        
        Args:
            filtered: 是否进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            List[float]: [rx, ry, rz]，单位: rad
        """
        return self.get_tcp_pose(filtered=filtered, samples=samples)[3:]
    
    def get_tcp_speed(self, filtered: bool = True, samples: int = 5) -> List[float]:
        """
        获取当前TCP速度
        
        Args:
            filtered: 是否进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            List[float]: TCP速度 [vx, vy, vz, wx, wy, wz]
                - vx, vy, vz: 线速度，单位: m/s
                - wx, wy, wz: 角速度，单位: rad/s
        """
        self._check_connection()
        if filtered:
            return self._sample_with_filter(lambda: list(self._rtde_r.getActualTCPSpeed()), samples)
        return list(self._rtde_r.getActualTCPSpeed())
    
    def get_tcp_force(self, filtered: bool = True, samples: int = 5) -> List[float]:
        """
        获取当前TCP力/力矩
        
        Args:
            filtered: 是否进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            List[float]: [Fx, Fy, Fz, Tx, Ty, Tz]
                - Fx, Fy, Fz: 力，单位: N
                - Tx, Ty, Tz: 力矩，单位: Nm
        """
        self._check_connection()
        if filtered:
            return self._sample_with_filter(lambda: list(self._rtde_r.getActualTCPForce()), samples)
        return list(self._rtde_r.getActualTCPForce())
    
    def get_robot_status(self, filtered: bool = True, samples: int = 5) -> dict:
        """
        获取机器人完整状态信息
        
        Args:
            filtered: 是否对数值型数据进行均值滤波，默认True
            samples: 滤波采样次数，默认5次
        
        Returns:
            dict: 包含所有状态信息的字典
        """
        self._check_connection()
        return {
            "timestamp": time.time(),
            "robot_mode": self.get_robot_mode(),
            "safety_mode": self.get_safety_mode(),
            "joint_positions_rad": self.get_joint_positions(filtered=filtered, samples=samples),
            "joint_positions_deg": self.get_joint_positions_deg(filtered=filtered, samples=samples),
            "joint_velocities": self.get_joint_velocities(filtered=filtered, samples=samples),
            "joint_currents": self.get_joint_currents(filtered=filtered, samples=samples),
            "tcp_pose": self.get_tcp_pose(filtered=filtered, samples=samples),
            "tcp_speed": self.get_tcp_speed(filtered=filtered, samples=samples),
            "tcp_force": self.get_tcp_force(filtered=filtered, samples=samples),
        }
    # def get_joint_positions(self) -> List[float]:
    #     """
    #     获取当前关节角度
        
    #     Returns:
    #         List[float]: 6个关节角度 [j1, j2, j3, j4, j5, j6]，单位: rad
    #     """
    #     self._check_connection()
    #     return list(self._rtde_r.getActualQ())
    
    # def get_joint_positions_deg(self) -> List[float]:
    #     """
    #     获取当前关节角度（角度制）
        
    #     Returns:
    #         List[float]: 6个关节角度 [j1, j2, j3, j4, j5, j6]，单位: 度(°)
    #     """
    #     return [math.degrees(j) for j in self.get_joint_positions()]
    
    # def get_joint_velocities(self) -> List[float]:
    #     """
    #     获取当前关节速度
        
    #     Returns:
    #         List[float]: 6个关节速度，单位: rad/s
    #     """
    #     self._check_connection()
    #     return list(self._rtde_r.getActualQd())
    
    # def get_joint_currents(self) -> List[float]:
    #     """
    #     获取当前关节电流
        
    #     Returns:
    #         List[float]: 6个关节电流，单位: A
    #     """
    #     self._check_connection()
    #     return list(self._rtde_r.getActualCurrent())
    
    # def get_joint_torques(self) -> List[float]:
    #     """
    #     获取当前关节力矩
        
    #     Returns:
    #         List[float]: 6个关节力矩，单位: Nm
    #     """
    #     self._check_connection()
    #     return list(self._rtde_r.getTargetMoment())
    
    # def get_tcp_pose(self) -> List[float]:
    #     """
    #     获取当前TCP位姿（基坐标系下）
        
    #     Returns:
    #         List[float]: TCP位姿 [x, y, z, rx, ry, rz]
    #             - x, y, z: 位置，单位: m
    #             - rx, ry, rz: 旋转向量(轴角表示)，单位: rad
    #     """
    #     self._check_connection()
    #     return list(self._rtde_r.getActualTCPPose())
    
    # def get_tcp_position(self) -> List[float]:
    #     """
    #     获取当前TCP位置
        
    #     Returns:
    #         List[float]: [x, y, z]，单位: m
    #     """
    #     return self.get_tcp_pose()[:3]
    
    # def get_tcp_orientation(self) -> List[float]:
    #     """
    #     获取当前TCP姿态（旋转向量表示）
        
    #     Returns:
    #         List[float]: [rx, ry, rz]，单位: rad
    #     """
    #     return self.get_tcp_pose()[3:]
    
    # def get_tcp_speed(self) -> List[float]:
    #     """
    #     获取当前TCP速度
        
    #     Returns:
    #         List[float]: TCP速度 [vx, vy, vz, wx, wy, wz]
    #             - vx, vy, vz: 线速度，单位: m/s
    #             - wx, wy, wz: 角速度，单位: rad/s
    #     """
    #     self._check_connection()
    #     return list(self._rtde_r.getActualTCPSpeed())
    
    # def get_tcp_force(self) -> List[float]:
    #     """
    #     获取当前TCP力/力矩
        
    #     Returns:
    #         List[float]: [Fx, Fy, Fz, Tx, Ty, Tz]
    #             - Fx, Fy, Fz: 力，单位: N
    #             - Tx, Ty, Tz: 力矩，单位: Nm
    #     """
    #     self._check_connection()
    #     return list(self._rtde_r.getActualTCPForce())
    
    def get_robot_mode(self) -> int:
        """
        获取机器人模式
        
        Returns:
            int: 机器人模式代码
                - -1: DISCONNECTED
                - 1: CONFIRM_SAFETY
                - 2: BOOTING  
                - 3: POWER_OFF
                - 4: POWER_ON
                - 5: IDLE
                - 6: BACKDRIVE
                - 7: RUNNING
        """
        self._check_connection()
        return self._rtde_r.getRobotMode()
    
    def get_safety_mode(self) -> int:
        """
        获取安全模式
        
        Returns:
            int: 安全模式代码
                - 1: NORMAL
                - 2: REDUCED
                - 3: PROTECTIVE_STOP
                - 4: RECOVERY
                - 5: SAFEGUARD_STOP
                - 6: SYSTEM_EMERGENCY_STOP
                - 7: ROBOT_EMERGENCY_STOP
                - 8: VIOLATION
                - 9: FAULT
        """
        self._check_connection()
        return self._rtde_r.getSafetyMode()
    
    def get_robot_status(self) -> dict:
        """
        获取机器人完整状态信息
        
        Returns:
            dict: 包含所有状态信息的字典
        """
        self._check_connection()
        return {
            "timestamp": time.time(),
            "robot_mode": self.get_robot_mode(),
            "safety_mode": self.get_safety_mode(),
            "joint_positions_rad": self.get_joint_positions(),
            "joint_positions_deg": self.get_joint_positions_deg(),
            "joint_velocities": self.get_joint_velocities(),
            "joint_currents": self.get_joint_currents(),
            "tcp_pose": self.get_tcp_pose(),
            "tcp_speed": self.get_tcp_speed(),
            "tcp_force": self.get_tcp_force(),
        }
    
    def get_inverse_kinematics(self, 
                                target_pose: List[float],
                                qnear: List[float] = None,
                                max_position_error: float = 1e-10,
                                max_orientation_error: float = 1e-10) -> Optional[List[float]]:
        """
        求解逆运动学
        
        根据目标TCP位姿计算对应的关节角度。
        
        Args:
            target_pose: 目标TCP位姿 [x, y, z, rx, ry, rz]
                - x, y, z: 位置，单位: m
                - rx, ry, rz: 旋转向量(轴角表示)，单位: rad
            qnear: 参考关节角度 [j1,...,j6]，单位: rad
                   用于选择最接近的逆解，默认使用当前关节角度
            max_position_error: 最大位置误差，单位: m
            max_orientation_error: 最大姿态误差，单位: rad
            
        Returns:
            List[float]: 关节角度 [j1, j2, j3, j4, j5, j6]，单位: rad
                         求解失败返回 None
        
        Example:
            >>> pose = [0.5, 0.2, 0.3, 2.22, 2.22, 0]
            >>> joints = robot.get_inverse_kinematics(pose)
            >>> if joints:
            ...     print(f"关节角度: {joints}")
        """
        self._check_connection()
        
        # 如果未指定参考关节角，使用当前关节角
        if qnear is None:
            qnear = self.get_joint_positions()
        
        try:
            result = self._rtde_c.getInverseKinematics(
                target_pose, 
                qnear, 
                max_position_error, 
                max_orientation_error
            )
            
            # 检查结果是否有效 (ur_rtde 返回空列表或全零表示失败)
            if result is None or len(result) != 6:
                return None
            
            # 检查是否为全零解（可能表示求解失败）
            if all(abs(j) < 1e-10 for j in result):
                # 再次验证：计算正运动学检查是否匹配
                # 如果目标位置也在原点附近，则可能是有效解
                if not all(abs(p) < 0.01 for p in target_pose[:3]):
                    return None
            
            return list(result)
            
        except Exception as e:
            print(f"[ERROR] 逆运动学求解失败: {e}")
            return None
    
    def get_forward_kinematics(self, 
                                joint_positions: List[float] = None,
                                tcp_offset: List[float] = None) -> Optional[List[float]]:
        """
        求解正运动学
        
        根据关节角度计算对应的TCP位姿。
        
        Args:
            joint_positions: 关节角度 [j1,...,j6]，单位: rad
                            默认使用当前关节角度
            tcp_offset: TCP偏移 [x, y, z, rx, ry, rz]，默认使用当前设置
            
        Returns:
            List[float]: TCP位姿 [x, y, z, rx, ry, rz]
                         求解失败返回 None
        """
        self._check_connection()
        
        if joint_positions is None:
            joint_positions = self.get_joint_positions()
        
        if tcp_offset is None:
            tcp_offset = [0, 0, 0, 0, 0, 0]
        
        try:
            result = self._rtde_c.getForwardKinematics(joint_positions, tcp_offset)
            return list(result) if result else None
        except Exception as e:
            print(f"[ERROR] 正运动学求解失败: {e}")
            return None
    
    def print_status(self):
        """打印当前机器人状态到控制台"""
        status = self.get_robot_status()
        
        print("\n" + "=" * 60)
        print("                    UR10 当前状态")
        print("=" * 60)
        print(f"  时间戳:       {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  机器人模式:   {status['robot_mode']}")
        print(f"  安全模式:     {status['safety_mode']}")
        print("-" * 60)
        print(f"  关节角度(°):  [{', '.join([f'{j:8.2f}' for j in status['joint_positions_deg']])}]")
        print(f"  关节速度:     [{', '.join([f'{v:8.4f}' for v in status['joint_velocities']])}] rad/s")
        print(f"  关节电流:     [{', '.join([f'{c:8.4f}' for c in status['joint_currents']])}] A")
        print("-" * 60)
        print(f"  TCP位置(m):   [{', '.join([f'{p:8.4f}' for p in status['tcp_pose'][:3]])}]")
        print(f"  TCP姿态(rad): [{', '.join([f'{r:8.4f}' for r in status['tcp_pose'][3:]])}]")
        print(f"  TCP力(N/Nm):  [{', '.join([f'{f:8.2f}' for f in status['tcp_force']])}]")
        print("=" * 60 + "\n")
    
    # ==========================================================================
    # 关节空间运动控制
    # ==========================================================================
    
    def move_j(self, 
               target_joints: List[float],
               vel: float = None,
               acc: float = None,
               asynchronous: bool = False) -> bool:
        """
        关节空间运动 (MoveJ)
        
        机器人各关节同时运动到目标角度，路径为关节空间中的直线，
        TCP轨迹通常为曲线。适用于快速到达目标位置，无需精确控制路径。
        
        Args:
            target_joints: 目标关节角度 [j1, j2, j3, j4, j5, j6]，单位: rad
            vel: 关节速度，单位: rad/s，默认使用配置值
            acc: 关节加速度，单位: rad/s²，默认使用配置值
            asynchronous: 是否异步执行
                - False: 阻塞直到运动完成
                - True: 立即返回，运动在后台执行
        
        Returns:
            bool: 命令发送成功返回True
            
        Example:
            >>> robot.move_j([0, -1.57, 1.57, -1.57, -1.57, 0])
        """
        self._check_connection()
        
        vel = vel if vel is not None else self._default_joint_vel
        acc = acc if acc is not None else self._default_joint_acc
        
        # 限制速度和加速度
        vel = min(vel, MAX_JOINT_VEL)
        acc = min(acc, MAX_JOINT_ACC)
        
        try:
            self._rtde_c.moveJ(target_joints, vel, acc, asynchronous)
            return True
        except Exception as e:
            print(f"[ERROR] MoveJ执行失败: {e}")
            return False
    
    def move_j_deg(self,
                   target_joints_deg: List[float],
                   vel: float = None,
                   acc: float = None,
                   asynchronous: bool = False) -> bool:
        """
        关节空间运动（角度制）
        
        Args:
            target_joints_deg: 目标关节角度 [j1, j2, j3, j4, j5, j6]，单位: 度(°)
            vel: 关节速度，单位: rad/s
            acc: 关节加速度，单位: rad/s²
            asynchronous: 是否异步执行
            
        Returns:
            bool: 命令发送成功返回True
            
        Example:
            >>> robot.move_j_deg([0, -90, 90, -90, -90, 0])
        """
        target_joints_rad = [math.radians(j) for j in target_joints_deg]
        return self.move_j(target_joints_rad, vel, acc, asynchronous)
    
    def move_j_ik(self,
                  target_pose: List[float],
                  vel: float = None,
                  acc: float = None,
                  asynchronous: bool = False) -> bool:
        """
        关节空间运动到目标TCP位姿 (MoveJ + 逆运动学)
        
        使用机器人内置逆运动学求解，以关节空间运动方式到达指定TCP位姿。
        
        Args:
            target_pose: 目标TCP位姿 [x, y, z, rx, ry, rz]
                - x, y, z: 位置，单位: m
                - rx, ry, rz: 旋转向量，单位: rad
            vel: 关节速度，单位: rad/s
            acc: 关节加速度，单位: rad/s²
            asynchronous: 是否异步执行
            
        Returns:
            bool: 命令发送成功返回True
        """
        self._check_connection()
        
        vel = vel if vel is not None else self._default_joint_vel
        acc = acc if acc is not None else self._default_joint_acc
        
        try:
            self._rtde_c.moveJ_IK(target_pose, vel, acc, asynchronous)
            return True
        except Exception as e:
            print(f"[ERROR] MoveJ_IK执行失败: {e}")
            return False
    
    # ==========================================================================
    # 笛卡尔空间运动控制
    # ==========================================================================
    
    def move_l(self,
               target_pose: List[float],
               vel: float = None,
               acc: float = None,
               asynchronous: bool = False) -> bool:
        """
        笛卡尔空间直线运动 (MoveL)
        
        TCP沿直线轨迹运动到目标位姿，适用于需要精确控制TCP轨迹的场景。
        
        Args:
            target_pose: 目标TCP位姿 [x, y, z, rx, ry, rz]
                - x, y, z: 位置，单位: m
                - rx, ry, rz: 旋转向量(轴角表示)，单位: rad
            vel: TCP线速度，单位: m/s，默认使用配置值
            acc: TCP线加速度，单位: m/s²，默认使用配置值
            asynchronous: 是否异步执行
            
        Returns:
            bool: 命令发送成功返回True
            
        Example:
            >>> robot.move_l([0.5, 0.2, 0.3, 0, 3.14, 0])
        """
        self._check_connection()
        
        vel = vel if vel is not None else self._default_linear_vel
        acc = acc if acc is not None else self._default_linear_acc
        
        # 限制速度和加速度
        vel = min(vel, MAX_LINEAR_VEL)
        acc = min(acc, MAX_LINEAR_ACC)
        
        try:
            self._rtde_c.moveL(target_pose, vel, acc, asynchronous)
            return True
        except Exception as e:
            print(f"[ERROR] MoveL执行失败: {e}")
            return False
    
    def move_l_fk(self,
                  target_joints: List[float],
                  vel: float = None,
                  acc: float = None,
                  asynchronous: bool = False) -> bool:
        """
        笛卡尔直线运动到目标关节位置 (MoveL + 正运动学)
        
        使用正运动学计算目标位姿，然后以直线运动方式到达。
        
        Args:
            target_joints: 目标关节角度 [j1,...,j6]，单位: rad
            vel: TCP线速度，单位: m/s
            acc: TCP线加速度，单位: m/s²
            asynchronous: 是否异步执行
            
        Returns:
            bool: 命令发送成功返回True
        """
        self._check_connection()
        
        vel = vel if vel is not None else self._default_linear_vel
        acc = acc if acc is not None else self._default_linear_acc
        
        try:
            self._rtde_c.moveL_FK(target_joints, vel, acc, asynchronous)
            return True
        except Exception as e:
            print(f"[ERROR] MoveL_FK执行失败: {e}")
            return False
    
    def move_p(self,
               target_pose: List[float],
               vel: float = None,
               acc: float = None,
               blend: float = None,
               asynchronous: bool = False) -> bool:
        """
        笛卡尔空间点到点运动 (MoveP)
        
        类似MoveL，但支持路径混合(blend)，可实现平滑过渡。
        适用于连续轨迹运动，如焊接、涂胶等。
        
        Args:
            target_pose: 目标TCP位姿 [x, y, z, rx, ry, rz]
            vel: TCP线速度，单位: m/s
            acc: TCP线加速度，单位: m/s²
            blend: 混合半径，单位: m，用于路径平滑过渡
            asynchronous: 是否异步执行
            
        Returns:
            bool: 命令发送成功返回True
        """
        self._check_connection()
        
        vel = vel if vel is not None else self._default_linear_vel
        acc = acc if acc is not None else self._default_linear_acc
        blend = blend if blend is not None else self._default_blend
        
        try:
            self._rtde_c.moveP(target_pose, vel, acc, blend, asynchronous)
            return True
        except Exception as e:
            print(f"[ERROR] MoveP执行失败: {e}")
            return False
    
    # ==========================================================================
    # 相对运动
    # ==========================================================================
    
    def move_l_relative(self,
                        delta_pose: List[float],
                        vel: float = None,
                        acc: float = None,
                        asynchronous: bool = False) -> bool:
        """
        相对当前位姿的直线运动（基坐标系下）
        
        Args:
            delta_pose: 相对位移 [dx, dy, dz, drx, dry, drz]
                - dx, dy, dz: 位置增量，单位: m
                - drx, dry, drz: 姿态增量，单位: rad
            vel: TCP线速度，单位: m/s
            acc: TCP线加速度，单位: m/s²
            asynchronous: 是否异步执行
            
        Returns:
            bool: 命令发送成功返回True
            
        Example:
            >>> robot.move_l_relative([0, 0, 0.1, 0, 0, 0])  # 向上移动10cm
        """
        current_pose = self.get_tcp_pose()
        target_pose = [current_pose[i] + delta_pose[i] for i in range(6)]
        return self.move_l(target_pose, vel, acc, asynchronous)
    
    def translate(self,
                  dx: float = 0,
                  dy: float = 0,
                  dz: float = 0,
                  vel: float = None,
                  acc: float = None,
                  asynchronous: bool = False) -> bool:
        """
        TCP平移运动（基坐标系下）
        
        Args:
            dx: X轴方向位移，单位: m
            dy: Y轴方向位移，单位: m  
            dz: Z轴方向位移，单位: m
            vel: TCP线速度，单位: m/s
            acc: TCP线加速度，单位: m/s²
            asynchronous: 是否异步执行
            
        Returns:
            bool: 命令发送成功返回True
            
        Example:
            >>> robot.translate(dz=0.1)  # 向上移动10cm
        """
        return self.move_l_relative([dx, dy, dz, 0, 0, 0], vel, acc, asynchronous)
    
    # ==========================================================================
    # 速度控制模式
    # ==========================================================================
    
    def speed_j(self,
                joint_speeds: List[float],
                acc: float = None,
                time: float = 0) -> bool:
        """
        关节速度控制
        
        以指定速度运动各关节，直到调用stop()或发送新的速度指令。
        
        Args:
            joint_speeds: 各关节速度 [v1, v2, v3, v4, v5, v6]，单位: rad/s
            acc: 加速度，单位: rad/s²
            time: 运动时间，单位: s，0表示持续运动直到收到新指令
            
        Returns:
            bool: 命令发送成功返回True
        """
        self._check_connection()
        acc = acc if acc is not None else self._default_joint_acc
        
        try:
            self._rtde_c.speedJ(joint_speeds, acc, time)
            return True
        except Exception as e:
            print(f"[ERROR] SpeedJ执行失败: {e}")
            return False
    
    def speed_l(self,
                tcp_speed: List[float],
                acc: float = None,
                time: float = 0) -> bool:
        """
        TCP速度控制
        
        以指定速度运动TCP，直到调用stop()或发送新的速度指令。
        
        Args:
            tcp_speed: TCP速度 [vx, vy, vz, wx, wy, wz]
                - vx, vy, vz: 线速度，单位: m/s
                - wx, wy, wz: 角速度，单位: rad/s
            acc: 加速度，单位: m/s²
            time: 运动时间，单位: s，0表示持续运动
            
        Returns:
            bool: 命令发送成功返回True
        """
        self._check_connection()
        acc = acc if acc is not None else self._default_linear_acc
        
        try:
            self._rtde_c.speedL(tcp_speed, acc, time)
            return True
        except Exception as e:
            print(f"[ERROR] SpeedL执行失败: {e}")
            return False
    
    # ==========================================================================
    # 实时伺服控制
    # ==========================================================================
    
    def servo_j(self,
                target_joints: List[float],
                vel: float = None,
                acc: float = None,
                dt: float = None,
                lookahead_time: float = None,
                gain: float = None) -> bool:
        """
        关节伺服控制 (ServoJ)
        
        用于实时控制场景（如视觉伺服、力控），以高频率发送关节目标位置。
        
        注意:
            - 必须以固定频率(如500Hz)持续调用
            - 目标位置应与当前位置接近，否则可能导致不平滑运动
            - 使用前建议调用init_period()初始化周期
        
        Args:
            target_joints: 目标关节角度 [j1,...,j6]，单位: rad
            vel: 关节速度，单位: rad/s（安全限制用）
            acc: 关节加速度，单位: rad/s²（安全限制用）
            dt: 控制周期，单位: s，默认为1/rtde_frequency
            lookahead_time: 前瞻时间，单位: s，范围0.03-0.2
            gain: 比例增益，范围100-2000
            
        Returns:
            bool: 命令发送成功返回True
        """
        self._check_connection()
        
        vel = vel if vel is not None else self._default_joint_vel
        acc = acc if acc is not None else self._default_joint_acc
        dt = dt if dt is not None else 1.0 / self.rtde_frequency
        lookahead_time = lookahead_time if lookahead_time is not None else SERVO_LOOKAHEAD_TIME
        gain = gain if gain is not None else SERVO_GAIN
        
        try:
            self._rtde_c.servoJ(target_joints, vel, acc, dt, lookahead_time, gain)
            return True
        except Exception as e:
            print(f"[ERROR] ServoJ执行失败: {e}")
            return False
    
    def servo_l(self,
                target_pose: List[float],
                vel: float = None,
                acc: float = None,
                dt: float = None,
                lookahead_time: float = None,
                gain: float = None) -> bool:
        """
        笛卡尔伺服控制 (ServoL)
        
        用于实时控制场景，以高频率发送TCP目标位姿。
        
        Args:
            target_pose: 目标TCP位姿 [x, y, z, rx, ry, rz]
            vel: 速度限制，单位: m/s
            acc: 加速度限制，单位: m/s²
            dt: 控制周期，单位: s
            lookahead_time: 前瞻时间，单位: s
            gain: 比例增益
            
        Returns:
            bool: 命令发送成功返回True
        """
        self._check_connection()
        
        vel = vel if vel is not None else self._default_linear_vel
        acc = acc if acc is not None else self._default_linear_acc
        dt = dt if dt is not None else 1.0 / self.rtde_frequency
        lookahead_time = lookahead_time if lookahead_time is not None else SERVO_LOOKAHEAD_TIME
        gain = gain if gain is not None else SERVO_GAIN
        
        try:
            self._rtde_c.servoL(target_pose, vel, acc, dt, lookahead_time, gain)
            return True
        except Exception as e:
            print(f"[ERROR] ServoL执行失败: {e}")
            return False
    
    def init_period(self) -> float:
        """
        初始化伺服周期
        
        在伺服控制循环开始时调用，返回周期起始时间戳。
        
        Returns:
            float: 周期起始时间戳
            
        Example:
            >>> t_start = robot.init_period()
            >>> robot.servo_j(target_joints)
            >>> robot.wait_period(t_start)
        """
        self._check_connection()
        return self._rtde_c.initPeriod()
    
    def wait_period(self, t_start: float):
        """
        等待伺服周期结束
        
        确保以固定频率执行控制循环。
        
        Args:
            t_start: 周期起始时间戳（由init_period()返回）
        """
        self._check_connection()
        self._rtde_c.waitPeriod(t_start)
    
    def servo_stop(self):
        """停止伺服运动"""
        self._check_connection()
        self._rtde_c.servoStop()
    
    # ==========================================================================
    # 停止控制
    # ==========================================================================
    
    def stop(self, acc: float = 2.0):
        """
        停止当前运动
        
        Args:
            acc: 减速度，单位: rad/s² 或 m/s²
        """
        self._check_connection()
        try:
            self._rtde_c.stopJ(acc)
        except Exception as e:
            print(f"[WARNING] 停止命令执行异常: {e}")
    
    def stop_l(self, acc: float = 2.0):
        """
        停止当前直线运动
        
        Args:
            acc: 减速度，单位: m/s²
        """
        self._check_connection()
        self._rtde_c.stopL(acc)
    
    def stop_script(self):
        """停止当前运行的脚本"""
        self._check_connection()
        self._rtde_c.stopScript()
    
    # ==========================================================================
    # 预设位姿
    # ==========================================================================
    
    def go_home(self, joint = True, vel: float = None, acc: float = None) -> bool:
        """
        移动到Home位置
        
        Args:
            vel: 关节速度，单位: rad/s
            acc: 关节加速度，单位: rad/s²
            
        Returns:
            bool: 命令发送成功返回True
        """
        if joint:
            print(f"Moving to Home position {HOME_JOINTS_DEG}...")
            input("Press Enter to continue...")
            return self.move_j_deg(HOME_JOINTS_DEG, vel, acc)
        else:
            home_pose = HOME_POSE_XYZRPY
            rtde_pose = pose2rtde(home_pose)
            # 打印目标位姿，确认安全
            print(f"Moving to Home pose (RTDE): {rtde_pose}")
            input("Press Enter to continue...")
            return self.move_l(rtde_pose, vel, acc)
    
    def go_ready(self, joint = True, vel: float = None, acc: float = None) -> bool:
        """
        移动到Ready/待机位置
        
        Args:
            vel: 关节速度，单位: rad/s
            acc: 关节加速度，单位: rad/s²
            
        Returns:
            bool: 命令发送成功返回True
        """
        if joint:
            print(f"Moving to Ready position {READY_JOINTS_DEG}...")
            input("Press Enter to continue...")
            return self.move_j_deg(READY_JOINTS_DEG, vel, acc)
        else:
            ready_pose = READY_POSE_XYZRPY
            rtde_pose = pose2rtde(ready_pose)
            # 打印目标位姿，确认安全
            print(f"Moving to Ready pose (RTDE): {rtde_pose}")
            input("Press Enter to continue...")
            return self.move_l(rtde_pose, vel, acc)
    
    # ==========================================================================
    # 设置参数
    # ==========================================================================
    
    def set_default_joint_params(self, vel: float = None, acc: float = None):
        """
        设置默认关节运动参数
        
        Args:
            vel: 默认关节速度，单位: rad/s
            acc: 默认关节加速度，单位: rad/s²
        """
        if vel is not None:
            self._default_joint_vel = min(vel, MAX_JOINT_VEL)
        if acc is not None:
            self._default_joint_acc = min(acc, MAX_JOINT_ACC)
    
    def set_default_linear_params(self, vel: float = None, acc: float = None):
        """
        设置默认笛卡尔运动参数
        
        Args:
            vel: 默认TCP线速度，单位: m/s
            acc: 默认TCP线加速度，单位: m/s²
        """
        if vel is not None:
            self._default_linear_vel = min(vel, MAX_LINEAR_VEL)
        if acc is not None:
            self._default_linear_acc = min(acc, MAX_LINEAR_ACC)
    
    def set_tcp(self, tcp_offset: List[float]):
        """
        设置工具中心点偏移
        
        Args:
            tcp_offset: TCP偏移 [x, y, z, rx, ry, rz]
                - x, y, z: 位置偏移，单位: m
                - rx, ry, rz: 旋转偏移，单位: rad
        """
        self._check_connection()
        self._rtde_c.setTcp(tcp_offset)
    
    def set_payload(self, mass: float, cog: List[float] = None):
        """
        设置负载参数
        
        Args:
            mass: 负载质量，单位: kg
            cog: 负载重心 [x, y, z]，单位: m，默认为[0,0,0]
        """
        self._check_connection()
        if cog is None:
            cog = [0, 0, 0]
        self._rtde_c.setPayload(mass, cog)
    
    # ==========================================================================
    # 运动状态检查
    # ==========================================================================
    
    def is_steady(self) -> bool:
        """
        检查机器人是否静止
        
        Returns:
            bool: 静止返回True
        """
        self._check_connection()
        return self._rtde_c.isSteady()
    
    def wait_until_steady(self, timeout: float = 10.0) -> bool:
        """
        等待机器人静止
        
        Args:
            timeout: 超时时间，单位: s
            
        Returns:
            bool: 在超时前静止返回True，超时返回False
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_steady():
                return True
            time.sleep(0.01)
        return False
    
    # ==========================================================================
    # 上下文管理
    # ==========================================================================
    
    def __enter__(self):
        """支持with语句"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动断开连接"""
        self.disconnect()


# ==============================================================================
# 使用示例
# ==============================================================================

if __name__ == "__main__":
    from controller.Transforms import *
    robot = UR10Controller(ip="192.168.70.10")
    if robot.connect():
        robot.print_status()
        joint = robot.get_joint_positions_deg()
        print(f"当前关节角度(°): {joint}")
        rtde_pose = robot.get_tcp_pose()
        print(f"当前TCP位姿: {rtde_pose}")
        pose = rtde2pose(rtde_pose)
        print(f"当前TCP位姿:\n{pose}")
        robot.go_home(joint=True)
        # robot.wait_until_steady()
        # robot.go_ready()
        # robot.wait_until_steady()
        robot.disconnect()