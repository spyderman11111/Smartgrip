#!/usr/bin/env python3
"""
ROS 2 Node: Virtual Top-Down Locator (RGB + Virtual Horizontal Plane)

目的：
- 读取目标像素中心 (u,v)，结合相机内/外参与 TF，
  在 `base_link` 下用“虚拟水平面 z = z_virt”求交，得到物体中心 C。
- 计算物体正上方 h（默认 0.30 m）的定位点 P_top，并保证末端执行器到达时与桌面平行（-Z_tool // +Z_base）。
- 规划：当前姿态 -> 预靠近点 P_pre (在 P_top 上方 `approach_clearance`) -> 笛卡尔直线下探到 P_top。
- 可选执行（MoveIt），或仅发布姿态用于可视化/调试。

=========================== 参数（请在程序前面配置） ===========================
# 相机内参（若 use_camera_info=True，则从 /camera_info 自动读取并覆盖 fx,fy,cx,cy）
- use_camera_info        [bool]   默认 True
- camera_info_topic      [str]    默认 '/camera_info'
- fx, fy, cx, cy         [float]  相机内参（去畸变后的主点与焦距）

# 相机外参（tool -> cam），单位：米 + 弧度（RPY，XYZ固定轴顺序）
- t_tool_cam_xyz         [list[float;3]]  默认 [0.0, 0.0, 0.10]
- t_tool_cam_rpy         [list[float;3]]  默认 [0.0, 0.0, 0.0]

# 坐标系 & MoveIt
- base_frame             [str]    默认 'base_link'
- tool_frame             [str]    默认 'tool0' 或你的 TCP 帧名
- group_name             [str]    默认 'ur_manipulator'
- eef_link               [str]    默认 'tool0'（你的末端执行器链接名）

# 虚拟水平面 & 目标高度
- z_virt                 [float]  虚拟桌面高度（m），默认 0.0
- h_above                [float]  物体上方高度 h（m），默认 0.30
- approach_clearance     [float]  预靠近点相对 P_top 的额外高度（m），默认 0.10

# 输入与执行
- target_pixel_topic     [str]    默认 '/target_pixel'（geometry_msgs/PointStamped，x=u, y=v）
- execute                [bool]   默认 False（仅发布姿态）。True 时调用 MoveIt 执行。
- max_vel_scale          [float]  默认 0.2
- max_acc_scale          [float]  默认 0.2

# 输出话题
- pub_p_top              [str]    默认 '/p_top'
- pub_p_pre              [str]    默认 '/p_pre'
- pub_path_markers       [str]    默认 '/path_markers'（可选：略）
===============================================================================

使用：
  ros2 run <your_pkg> virtual_topdown_locator_node.py --ros-args \
    -p use_camera_info:=true -p camera_info_topic:=/camera_info \
    -p t_tool_cam_xyz:='[0.0, 0.0, 0.10]' -p t_tool_cam_rpy:='[0.0, 0.0, 0.0]' \
    -p base_frame:=base_link -p tool_frame:=tool0 -p group_name:=ur_manipulator -p eef_link:=tool0 \
    -p z_virt:=0.0 -p h_above:=0.30 -p approach_clearance:=0.10 \
    -p target_pixel_topic:=/target_pixel -p execute:=true

输入消息格式：
- /target_pixel : geometry_msgs/PointStamped
    point.x = u（像素列），point.y = v（像素行），point.z 可忽略；header.stamp 用于 TF 同步。

注意：
- 像素应为去畸变后的坐标，且与 fx,fy,cx,cy 对应。
- 末端姿态到达点时会保持“与桌面平行”：-Z_tool 对齐 +Z_base。
- 若不执行，仅发布 P_pre / P_top 供 RViz 查看。
"""

import math
from typing import Optional, Tuple, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PointStamped, PoseStamped, Pose, PoseArray
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo

import tf2_ros
from tf2_ros import TransformException

# 可选：MoveIt
try:
    import moveit_commander  # type: ignore
    _HAS_MOVEIT = True
except Exception:
    _HAS_MOVEIT = False


# ---------------------------- 工具函数 ---------------------------- #

def rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
    """固定轴 XYZ 顺序（Roll->Pitch->Yaw）。返回 3x3 旋转矩阵。"""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,             cp*cr]
    ], dtype=float)
    return R


def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """旋转矩阵 -> 四元数 [x,y,z,w]（右手系）。"""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    t = m00 + m11 + m22
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    # 归一化
    n = math.sqrt(x*x + y*y + z*z + w*w)
    return (x/n, y/n, z/n, w/n)


def pixel_to_ray(u: float, v: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """像素 -> 相机系视线（单位向量，optical: x右 y下 z前）。"""
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray = np.array([x, y, 1.0], dtype=float)
    ray /= np.linalg.norm(ray)
    return ray


# ---------------------------- 主节点 ---------------------------- #

class VirtualTopdownNode(Node):
    def __init__(self):
        super().__init__('virtual_topdown_locator')

        # ---------------- 参数 ----------------
        self.declare_parameter('use_camera_info', True)
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('fx', 800.0)
        self.declare_parameter('fy', 800.0)
        self.declare_parameter('cx', 640.0)
        self.declare_parameter('cy', 360.0)

        self.declare_parameter('t_tool_cam_xyz', [0.0, 0.0, 0.10])
        self.declare_parameter('t_tool_cam_rpy', [0.0, 0.0, 0.0])

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('group_name', 'ur_manipulator')
        self.declare_parameter('eef_link', 'tool0')

        self.declare_parameter('z_virt', 0.0)
        self.declare_parameter('h_above', 0.30)
        self.declare_parameter('approach_clearance', 0.10)

        self.declare_parameter('target_pixel_topic', '/target_pixel')
        self.declare_parameter('execute', False)
        self.declare_parameter('max_vel_scale', 0.2)
        self.declare_parameter('max_acc_scale', 0.2)

        # 读参数
        self.use_ci = self.get_parameter('use_camera_info').get_parameter_value().bool_value
        self.ci_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.fx = float(self.get_parameter('fx').get_parameter_value().double_value)
        self.fy = float(self.get_parameter('fy').get_parameter_value().double_value)
        self.cx = float(self.get_parameter('cx').get_parameter_value().double_value)
        self.cy = float(self.get_parameter('cy').get_parameter_value().double_value)

        self.t_tool_cam_xyz = np.array(self.get_parameter('t_tool_cam_xyz').get_parameter_value().double_array_value or [0.0,0.0,0.10], dtype=float)
        self.t_tool_cam_rpy = np.array(self.get_parameter('t_tool_cam_rpy').get_parameter_value().double_array_value or [0.0,0.0,0.0], dtype=float)

        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.tool_frame = self.get_parameter('tool_frame').get_parameter_value().string_value
        self.group_name = self.get_parameter('group_name').get_parameter_value().string_value
        self.eef_link = self.get_parameter('eef_link').get_parameter_value().string_value

        self.z_virt = float(self.get_parameter('z_virt').get_parameter_value().double_value)
        self.h_above = float(self.get_parameter('h_above').get_parameter_value().double_value)
        self.approach_clearance = float(self.get_parameter('approach_clearance').get_parameter_value().double_value)

        self.target_pixel_topic = self.get_parameter('target_pixel_topic').get_parameter_value().string_value
        self.execute = bool(self.get_parameter('execute').get_parameter_value().bool_value)
        self.max_vel = float(self.get_parameter('max_vel_scale').get_parameter_value().double_value)
        self.max_acc = float(self.get_parameter('max_acc_scale').get_parameter_value().double_value)

        # 相机内参缓存
        self._have_K = not self.use_ci
        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # MoveIt
        if self.execute:
            if not _HAS_MOVEIT:
                self.get_logger().warn('execute=True 但未找到 moveit_commander，将仅发布位姿。')
                self.execute = False
            else:
                moveit_commander.roscpp_initialize([])
                self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
                self.move_group.set_pose_reference_frame(self.base_frame)
                self.move_group.set_end_effector_link(self.eef_link)
                self.move_group.set_max_velocity_scaling_factor(self.max_vel)
                self.move_group.set_max_acceleration_scaling_factor(self.max_acc)

        # 订阅
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.history = HistoryPolicy.KEEP_LAST
        if self.use_ci:
            self.sub_ci = self.create_subscription(CameraInfo, self.ci_topic, self._cb_ci, qos)
        self.sub_uv = self.create_subscription(PointStamped, self.target_pixel_topic, self._cb_uv, qos)

        # 发布
        self.pub_p_top = self.create_publisher(PoseStamped, '/p_top', 1)
        self.pub_p_pre = self.create_publisher(PoseStamped, '/p_pre', 1)

        # 打印参数摘要
        self.get_logger().info(
            f"虚拟平面 z_virt={self.z_virt:.3f}, h_above={self.h_above:.3f}, approach_clearance={self.approach_clearance:.3f}; "
            f"use_camera_info={self.use_ci}, execute={self.execute}")

    # ---------------- 回调 ----------------
    def _cb_ci(self, msg: CameraInfo):
        # 从 CameraInfo 获取内参（去畸变后的 K）
        K = np.array(msg.k, dtype=float).reshape(3,3)
        self.fx, self.fy, self.cx, self.cy = K[0,0], K[1,1], K[0,2], K[1,2]
        self._have_K = True

    def _cb_uv(self, msg: PointStamped):
        if not self._have_K:
            self.get_logger().warn('未获得相机内参 K，忽略此帧。')
            return
        u = float(msg.point.x)
        v = float(msg.point.y)
        stamp = msg.header.stamp

        # 获取 T_base_tool at stamp
        try:
            T_base_tool = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, rclpy.time.Time.from_msg(stamp))
        except TransformException as ex:
            self.get_logger().warn(f'TF 未就绪: {ex}')
            return

        R_bt, t_bt = self._tf_to_R_t(T_base_tool)
        # 计算 T_tool_cam
        R_tc = rpy_to_rot(self.t_tool_cam_rpy[0], self.t_tool_cam_rpy[1], self.t_tool_cam_rpy[2])
        t_tc = self.t_tool_cam_xyz
        # T_base_cam
        R_bc = R_bt @ R_tc
        t_bc = R_bt @ t_tc + t_bt

        # 像素 -> 相机系射线
        r_cam = pixel_to_ray(u, v, self.fx, self.fy, self.cx, self.cy)
        # 转到 base
        r_base = R_bc @ r_cam
        r_base = r_base / np.linalg.norm(r_base)
        o = t_bc

        # 与 z=z_virt 求交
        rz = float(r_base[2])
        if abs(rz) < 1e-5:
            self.get_logger().warn('视线近乎水平（|r_z|≈0），无法与虚拟平面求交。请调整相机姿态/高度。')
            return
        t_star = (self.z_virt - float(o[2])) / rz
        if t_star < 0:
            self.get_logger().warn('平面交点在相机后方（t<0），忽略此帧。')
            return
        C = o + t_star * r_base  # 目标中心（base）

        # 计算上方与预靠近点
        n = np.array([0.0, 0.0, 1.0], dtype=float)
        P_top = C + self.h_above * n
        P_pre = P_top + self.approach_clearance * n

        # 构造“与桌面平行”的末端姿态：-Z_tool // +Z_base
        z_tool_in_base = -n  # [0,0,-1]
        x_tool_in_base = np.array([1.0, 0.0, 0.0])
        # 重新正交化：确保 y = z × x
        y_tool_in_base = np.cross(z_tool_in_base, x_tool_in_base)
        # 若几何退化（x 与 z 共线），换一个 x
        if np.linalg.norm(y_tool_in_base) < 1e-6:
            x_tool_in_base = np.array([0.0, 1.0, 0.0])
            y_tool_in_base = np.cross(z_tool_in_base, x_tool_in_base)
        # 归一化
        x_tool_in_base /= np.linalg.norm(x_tool_in_base)
        y_tool_in_base /= np.linalg.norm(y_tool_in_base)
        z_tool_in_base /= np.linalg.norm(z_tool_in_base)
        R_des = np.column_stack((x_tool_in_base, y_tool_in_base, z_tool_in_base))  # 列为工具轴
        qx, qy, qz, qw = rot_to_quat(R_des)

        # 发布姿态
        ps_top = PoseStamped()
        ps_top.header = Header(stamp=stamp, frame_id=self.base_frame)
        ps_top.pose.position.x, ps_top.pose.position.y, ps_top.pose.position.z = float(P_top[0]), float(P_top[1]), float(P_top[2])
        ps_top.pose.orientation.x, ps_top.pose.orientation.y, ps_top.pose.orientation.z, ps_top.pose.orientation.w = qx, qy, qz, qw
        self.pub_p_top.publish(ps_top)

        ps_pre = PoseStamped()
        ps_pre.header = Header(stamp=stamp, frame_id=self.base_frame)
        ps_pre.pose.position.x, ps_pre.pose.position.y, ps_pre.pose.position.z = float(P_pre[0]), float(P_pre[1]), float(P_pre[2])
        ps_pre.pose.orientation.x, ps_pre.pose.orientation.y, ps_pre.pose.orientation.z, ps_pre.pose.orientation.w = qx, qy, qz, qw
        self.pub_p_pre.publish(ps_pre)

        self.get_logger().info(f"C=({C[0]:.3f},{C[1]:.3f},{C[2]:.3f})  P_top=({P_top[0]:.3f},{P_top[1]:.3f},{P_top[2]:.3f})")

        # 执行（先到 P_pre，再笛卡尔直线到 P_top）
        if self.execute:
            try:
                self._execute_motion(ps_pre, ps_top)
            except Exception as e:
                self.get_logger().error(f"执行失败: {e}")

    # ---------------- 执行/规划 ----------------
    def _execute_motion(self, ps_pre: PoseStamped, ps_top: PoseStamped):
        assert _HAS_MOVEIT
        # 先到 P_pre（规划-执行）
        self.move_group.set_pose_target(ps_pre)
        plan1 = self.move_group.plan()
        ok1 = plan1 and len(plan1.joint_trajectory.points) > 0
        if not ok1:
            self.get_logger().warn('到 P_pre 的规划失败，尝试直接 go。')
            self.move_group.set_pose_target(ps_pre)
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        # 从 P_pre 笛卡尔直线到 P_top（保持姿态）
        waypoints: List[Pose] = []
        waypoints.append(ps_top.pose)
        (plan2, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step=0.005, jump_threshold=0.0)
        if fraction < 0.999:
            self.get_logger().warn(f'笛卡尔路径覆盖率 {fraction:.2f} < 1.0，改为常规规划到 P_top。')
            self.move_group.set_pose_target(ps_top)
            plan3 = self.move_group.plan()
            ok3 = plan3 and len(plan3.joint_trajectory.points) > 0
            if not ok3:
                self.get_logger().error('到 P_top 的规划失败。')
                return
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
        else:
            self.get_logger().info('执行笛卡尔直线下探到 P_top。')
            self.move_group.execute(plan2, wait=True)
            self.move_group.stop()

    # ---------------- TF 工具 ----------------
    @staticmethod
    def _tf_to_R_t(transform) -> Tuple[np.ndarray, np.ndarray]:
        q = transform.transform.rotation
        t = transform.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], dtype=float)
        t_vec = np.array([t.x, t.y, t.z], dtype=float)
        return R, t_vec


def main():
    rclpy.init()
    node = VirtualTopdownNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if _HAS_MOVEIT and getattr(node, 'execute', False):
            try:
                node.move_group.stop()
                node.move_group.clear_pose_targets()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
