#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual Top-Down Locator + GroundingDINO（无直线下探版）
- 从图像检测目标 -> (u,v) -> 射线与虚拟平面求交 -> 计算 P_top（物体上方）
- 末端姿态保持与虚拟平面平行（-Z_tool // +Z_base）
- 可选：仅规划/执行到 P_top（不做笛卡尔直线下探）
- 启动后先发送一次“初始观察位姿”（关节角）到 joint_trajectory 控制器
"""

# ===================== 可随时修改的“初始观察位姿”与发送函数 ======================
# 关节名称（UR5e）
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# 初始观察位姿（弧度），按照上面的顺序
POS1 = [0.9298571348, -1.3298700166, 1.9266884963, -2.1331087552, -1.6006286780, -1.0919039885]

# 发送到的轨迹话题与移动时间
JOINT_TRAJ_TOPIC = '/scaled_joint_trajectory_controller/joint_trajectory'
INIT_MOVE_TIME = 3.0  # 秒

# ===========================================================================

import math
from typing import Tuple, List, Optional
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from rclpy.duration import Duration as RclDuration

from geometry_msgs.msg import PointStamped, PoseStamped, Pose
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as MsgDuration

import tf2_ros
from tf2_ros import TransformException

from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage

# GroundingDINO 预测器（按你的工程路径导入）
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except ImportError:
    import sys
    sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
    from gripanything.core.detect_with_dino import GroundingDinoPredictor

# 可选：MoveIt
try:
    import moveit_commander  # type: ignore
    _HAS_MOVEIT = True
except Exception:
    _HAS_MOVEIT = False


# ---------------------------- 初始位姿发送工具 ---------------------------- #
def publish_trajectory(node: Node, pub, positions: List[float], move_time: float):
    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    traj.header.stamp = node.get_clock().now().to_msg()

    pt = JointTrajectoryPoint()
    pt.positions = list(positions)
    sec = int(move_time)
    nsec = int((move_time - sec) * 1e9)
    pt.time_from_start = MsgDuration(sec=sec, nanosec=nsec)
    traj.points.append(pt)

    pub.publish(traj)
    node.get_logger().info(f'已发送关节目标（初始观察位姿）: {np.round(positions, 3).tolist()}')


# ---------------------------- 基础数学 ---------------------------- #
def rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,             cp*cr]
    ], dtype=float)

def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
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
    n = math.sqrt(x*x + y*y + z*z + w*w)
    return (x/n, y/n, z/n, w/n)


# ---------------------------- 主节点 ---------------------------- #
class VirtualTopdownNode(Node):
    def __init__(self, **node_kwargs):
        super().__init__('virtual_topdown_locator', **node_kwargs)

        # 相机 / 外参 / 平面 / MoveIt / DINO 参数（默认值，main() 会覆盖）
        self.declare_parameter('use_camera_info', True)
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('fx', 800.0); self.declare_parameter('fy', 800.0)
        self.declare_parameter('cx', 640.0); self.declare_parameter('cy', 360.0)
        self.declare_parameter('t_tool_cam_xyz', [0.0, 0.0, 0.10])
        self.declare_parameter('t_tool_cam_rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('t_tool_cam_quat', [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('group_name', 'ur_manipulator')
        self.declare_parameter('eef_link', 'tool0')
        self.declare_parameter('z_virt', 0.0)
        self.declare_parameter('h_above', 0.30)
        self.declare_parameter('approach_clearance', 0.10)  # 仍计算但不用于执行
        self.declare_parameter('target_pixel_topic', '/target_pixel')
        self.declare_parameter('execute', False)
        self.declare_parameter('max_vel_scale', 0.2)
        self.declare_parameter('max_acc_scale', 0.2)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('text_prompt', 'object')
        self.declare_parameter('dino_model_id', 'IDEA-Research/grounding-dino-tiny')
        self.declare_parameter('dino_device', 'cuda')
        self.declare_parameter('box_threshold', 0.25)
        self.declare_parameter('text_threshold', 0.25)

        # 读参数
        g = self.get_parameter
        self.use_ci = g('use_camera_info').value
        self.ci_topic = g('camera_info_topic').value
        self.fx = float(g('fx').value); self.fy = float(g('fy').value)
        self.cx = float(g('cx').value); self.cy = float(g('cy').value)
        self.t_tool_cam_xyz = np.array(g('t_tool_cam_xyz').value, dtype=float)
        self.t_tool_cam_rpy = np.array(g('t_tool_cam_rpy').value, dtype=float)
        self.t_tool_cam_quat = np.array(g('t_tool_cam_quat').value, dtype=float)
        self.base_frame = g('base_frame').value
        self.tool_frame = g('tool_frame').value
        self.group_name = g('group_name').value
        self.eef_link = g('eef_link').value
        self.z_virt = float(g('z_virt').value)
        self.h_above = float(g('h_above').value)
        self.approach_clearance = float(g('approach_clearance').value)
        self.target_pixel_topic = g('target_pixel_topic').value
        self.execute = bool(g('execute').value)
        self.max_vel = float(g('max_vel_scale').value)
        self.max_acc = float(g('max_acc_scale').value)
        self.image_topic = g('image_topic').value
        self.text_prompt = g('text_prompt').value
        self.dino_model_id = g('dino_model_id').value
        self.dino_device = g('dino_device').value
        self.box_th = float(g('box_threshold').value)
        self.text_th = float(g('text_threshold').value)

        # 内参缓存
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0,      0,      1]], dtype=float)
        self.D: Optional[np.ndarray] = None
        self.dist_model: Optional[str] = None
        self._have_K = not self.use_ci

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # MoveIt（仅用于到 P_top 的一次规划）
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

        # 订阅/发布
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        if self.use_ci:
            self.sub_ci = self.create_subscription(CameraInfo, self.ci_topic, self._cb_ci, qos)
        self.sub_uv = self.create_subscription(PointStamped, self.target_pixel_topic, self._cb_uv, qos)

        self.bridge = CvBridge()
        self.predictor = GroundingDinoPredictor(model_id=self.dino_model_id, device=self.dino_device)
        self.sub_img = self.create_subscription(Image, self.image_topic, self._cb_image, qos)

        self.pub_p_top = self.create_publisher(PoseStamped, '/p_top', 1)
        self.pub_p_pre = self.create_publisher(PoseStamped, '/p_pre', 1)

        # 关节轨迹发布器（初始观察位姿）
        self.traj_pub = self.create_publisher(JointTrajectory, JOINT_TRAJ_TOPIC, 10)
        # 启动后稍等一下发一次初始位姿，避免控制器未就绪
        self._init_pose_sent = False
        self.create_timer(0.8, self._send_initial_once)

        self.get_logger().info(
            f"[VTDown] z_virt={self.z_virt:.3f}, h_above={self.h_above:.3f}; "
            f"use_camera_info={self.use_ci}, image_topic={self.image_topic}, prompt='{self.text_prompt}', execute={self.execute}"
        )
        self._busy = False

    # 只发送一次初始观察位姿
    def _send_initial_once(self):
        if self._init_pose_sent:
            return
        if self.traj_pub.get_subscription_count() is not None:
            publish_trajectory(self, self.traj_pub, POS1, INIT_MOVE_TIME)
            self._init_pose_sent = True

    # ---------------- CameraInfo 回调 ----------------
    def _cb_ci(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.K = K.copy()
        self.fx, self.fy, self.cx, self.cy = K[0,0], K[1,1], K[0,2], K[1,2]
        if msg.d is not None and len(msg.d) > 0:
            self.D = np.array(msg.d, dtype=float).reshape(-1)
        else:
            if hasattr(msg, 'distortion') and msg.distortion:
                self.D = np.array(msg.distortion, dtype=float).reshape(-1)
            else:
                self.D = None
        self.dist_model = msg.distortion_model or None
        self._have_K = True

    # ---------------- 图像回调（自动检测 -> (u,v)） ----------------
    def _cb_image(self, msg: Image):
        if self._busy:
            return
        if not self._have_K:
            self.get_logger().warn_once('等待 /camera_info 提供内参/畸变。')
            return
        try:
            self._busy = True
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            pil = PILImage.fromarray(rgb)
            boxes, labels = self.predictor.predict(
                pil, self.text_prompt,
                box_threshold=self.box_th, text_threshold=self.text_th
            )
            if boxes is None or len(boxes) == 0:
                self.get_logger().warn('未检测到目标，检查 text_prompt/阈值。')
                return
            x0, y0, x1, y1 = boxes[0].tolist()
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)
            self._process_uv(u, v, msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f"DINO/图像处理失败: {e}")
        finally:
            self._busy = False

    # ---------------- 手动像素回调（可选） ----------------
    def _cb_uv(self, msg: PointStamped):
        if not self._have_K:
            self.get_logger().warn('未获得相机内参/畸变，忽略此帧。')
            return
        self._process_uv(float(msg.point.x), float(msg.point.y), msg.header.stamp)

    # ---------------- 主流程：像素 -> 射线 -> 求交 -> 发布/执行 ----------------
    def _process_uv(self, u: float, v: float, stamp):
        # TF: base<-tool at stamp
        try:
            T_base_tool = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, rclpy.time.Time.from_msg(stamp))
        except TransformException as ex:
            self.get_logger().warn(f'TF 未就绪: {ex}')
            return

        R_bt, t_bt = self._tf_to_R_t(T_base_tool)

        # 外参：tool->cam（优先 quat）
        if self.t_tool_cam_quat is not None and not np.allclose(self.t_tool_cam_quat, [0,0,0,1.0]):
            R_tc = quat_to_rot(*self.t_tool_cam_quat.tolist())
        else:
            R_tc = rpy_to_rot(*self.t_tool_cam_rpy.tolist())
        t_tc = self.t_tool_cam_xyz

        # base 下的相机位姿
        R_bc = R_bt @ R_tc
        t_bc = R_bt @ t_tc + t_bt

        # (u,v) -> 归一化相机坐标（优先使用畸变去除）
        if self.D is not None and self.dist_model in (None, '', 'plumb_bob', 'rational_polynomial'):
            pts = np.array([[[u, v]]], dtype=np.float32)
            undist = cv2.undistortPoints(pts, self.K, self.D, P=None)  # -> 1x1x2
            x, y = float(undist[0,0,0]), float(undist[0,0,1])
            r_cam = np.array([x, y, 1.0], dtype=float)
        else:
            x = (u - self.cx) / self.fx
            y = (v - self.cy) / self.fy
            r_cam = np.array([x, y, 1.0], dtype=float)

        r_cam /= np.linalg.norm(r_cam)
        r_base = R_bc @ r_cam
        r_base /= np.linalg.norm(r_base)
        o = t_bc

        rz = float(r_base[2])
        if abs(rz) < 1e-6:
            self.get_logger().warn('视线近水平（|r_z|≈0），无法与虚拟平面求交。')
            return
        t_star = (self.z_virt - float(o[2])) / rz
        if t_star < 0:
            self.get_logger().warn('平面交点在相机后方（t<0），忽略。')
            return

        C = o + t_star * r_base  # base 下物体中心
        n = np.array([0.0, 0.0, 1.0], dtype=float)
        P_top = C + self.h_above * n
        P_pre = P_top + self.approach_clearance * n  # 仍发布供可视化

        # 末端姿态：-Z_tool // +Z_base
        z_tool_in_base = -n
        x_tool_in_base = np.array([1.0, 0.0, 0.0])
        y_tool_in_base = np.cross(z_tool_in_base, x_tool_in_base)
        if np.linalg.norm(y_tool_in_base) < 1e-6:
            x_tool_in_base = np.array([0.0, 1.0, 0.0])
            y_tool_in_base = np.cross(z_tool_in_base, x_tool_in_base)
        x_tool_in_base /= np.linalg.norm(x_tool_in_base)
        y_tool_in_base /= np.linalg.norm(y_tool_in_base)
        z_tool_in_base /= np.linalg.norm(z_tool_in_base)
        R_des = np.column_stack((x_tool_in_base, y_tool_in_base, z_tool_in_base))
        qx, qy, qz, qw = rot_to_quat(R_des)

        # 发布 P_top / P_pre（供可视化/调试）
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

        self.get_logger().info(f"(u,v)=({u:.1f},{v:.1f})  C=({C[0]:.3f},{C[1]:.3f},{C[2]:.3f})  P_top=({P_top[0]:.3f},{P_top[1]:.3f},{P_top[2]:.3f})")

        # 执行：仅到 P_top（不做直线下探）
        if self.execute:
            try:
                self._go_to_pose(ps_top)
            except Exception as e:
                self.get_logger().error(f"执行失败: {e}")

    # ---------------- 仅到 P_top 的 MoveIt 执行 ----------------
    def _go_to_pose(self, ps_target: PoseStamped):
        assert _HAS_MOVEIT
        self.move_group.set_pose_target(ps_target)
        plan = self.move_group.plan()
        ok = plan and len(plan.joint_trajectory.points) > 0
        if not ok:
            self.get_logger().warn('到 P_top 的规划失败，尝试直接 go。')
            self.move_group.set_pose_target(ps_target)
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

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


# ---------------------------- 入口：集中配置参数（按需随时改） ---------------------------- #
def main():
    rclpy.init()

    # —— 手眼标定（tool->cam）外参 —— 
    t_xyz = [0.05468636852374024, -0.029182661943126947, 0.05391824813032688]
    q = [-0.0036165657530785695, -0.000780788838366878, 0.7078681983794892, 0.7063348529868249]  # [qx,qy,qz,qw]

    # —— 相机内参（作为回退；若订阅到 /camera_info 会覆盖）——
    fx, fy = 2674.38037, 2667.42113
    cx, cy = 954.59221, 1074.96595
    # 如果你的图像是 rect 后的，可考虑改用投影矩阵 P 的参数：2577.14087, 2594.39014, 919.70325, 1091.32195

    overrides = [
        # 图像 + DINO
        Parameter('image_topic',         value='/image_raw'),
        Parameter('text_prompt',         value='object'),
        Parameter('dino_model_id',       value='IDEA-Research/grounding-dino-tiny'),
        Parameter('dino_device',         value='cuda'),    # 无 GPU 改 'cpu'
        Parameter('box_threshold',       value=0.25),
        Parameter('text_threshold',      value=0.25),

        # 相机内参（回退）
        Parameter('use_camera_info',     value=True),
        Parameter('camera_info_topic',   value='/camera_info'),
        Parameter('fx',                  value=float(fx)),
        Parameter('fy',                  value=float(fy)),
        Parameter('cx',                  value=float(cx)),
        Parameter('cy',                  value=float(cy)),

        # 手眼外参（优先 quat）
        Parameter('t_tool_cam_xyz',      value=t_xyz),
        Parameter('t_tool_cam_quat',     value=q),

        # 坐标系与末端
        Parameter('base_frame',          value='base_link'),
        Parameter('tool_frame',          value='tool0'),
        Parameter('group_name',          value='ur_manipulator'),
        Parameter('eef_link',            value='tool0'),

        # 虚拟平面与高度
        Parameter('z_virt',              value=0.0),
        Parameter('h_above',             value=0.30),
        Parameter('approach_clearance',  value=0.10),

        # 执行与限速（仅到 P_top）
        Parameter('execute',             value=False),
        Parameter('max_vel_scale',       value=0.2),
        Parameter('max_acc_scale',       value=0.2),
    ]

    node = VirtualTopdownNode(parameter_overrides=overrides)

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
