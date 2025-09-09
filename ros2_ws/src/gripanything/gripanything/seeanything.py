#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything.py — GroundingDINO + Virtual Top-Down Locator（无直线下探版 + 发布物体TF）
订阅相机图像 -> 调用 GroundingDinoPredictor -> 取首个检测框中心 (u,v)
-> 相机光线在 base 下与虚拟平面 z=z_virt 求交 -> 得到 C_base
-> 计算上方点 P_top / P_pre，发布 Pose 与 TF
-> 发布调试图像（框、标签、中心点）
"""

from dataclasses import dataclass
import math
from typing import Tuple, List, Optional
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration as RclDuration

from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as MsgDuration

import tf2_ros
from tf2_ros import TransformException

from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage

# ====== 关节与初始化运动 ======
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
POS1 = [0.9004912376403809, -1.2545607549003144, 1.2091739813434046,
        -1.488419310455658, -1.5398953596698206, -0.6954544226275843]
JOINT_TRAJ_TOPIC = '/scaled_joint_trajectory_controller/joint_trajectory'
INIT_MOVE_TIME = 3.0  # 秒

# ====== 你的 GroundingDinoPredictor（严格复用你给的模块与调用方式）======
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except ImportError:
    import sys
    sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
    from gripanything.core.detect_with_dino import GroundingDinoPredictor


# ====== 配置（可按需修改默认值）======
@dataclass(frozen=True)
class Config:
    # 话题名
    IMAGE_TOPIC: str = '/my_camera/pylon_ros2_camera_node/image_raw'
    CAMERA_INFO_TOPIC: str = '/my_camera/pylon_ros2_camera_node/camera_info'
    DEBUG_IMAGE_TOPIC: str = '/seeanything/debug_image'

    # CameraInfo
    USE_CAMERA_INFO: bool = True

    # 内参（无 /camera_info 时作为回退）
    FX: float = 2674.3803723910564
    FY: float = 2667.4211254043507
    CX: float = 954.5922081613583
    CY: float = 1074.965947832258

    # 坐标系
    BASE_FRAME: str = 'base_link'
    TOOL_FRAME: str = 'tool0'
    EEF_LINK: str = 'tool0'
    GROUP_NAME: str = 'ur_manipulator'

    # 虚拟平面与高度
    Z_VIRT: float = 0.0          # base 下工作面高度
    H_ABOVE: float = 0.30        # 物体上方点高度
    APPROACH_CLEARANCE: float = 0.10

    # DINO 参数
    TEXT_PROMPT: str = "yellow object ."
    DINO_MODEL_ID: str = 'IDEA-Research/grounding-dino-tiny'
    DINO_DEVICE: str = 'cuda'
    BOX_THRESHOLD: float = 0.20
    TEXT_THRESHOLD: float = 0.20

    # 手眼外参（tool->cam）
    T_TOOL_CAM_XYZ: List[float] = (-0.000006852374024, -0.099182661943126947, 0.02391824813032688)
    T_TOOL_CAM_QUAT: List[float] = (-0.0036165657530785695, -0.000780788838366878,
                                    0.7078681983794892, 0.7063348529868249)  # [qx, qy, qz, qw]

    # 物体 TF 名称
    OBJECT_FRAME: str = 'object_position'

    # 启动一次性发送初始观察位姿
    SEND_INIT_POSE: bool = True

    # TF 对齐策略
    USE_LATEST_TF_ON_FAIL: bool = True   # 对应时间不可用时，是否回退最新 TF
    STRICT_TIME_MODE: bool = False       # 若 True，严格时间对齐（失败直接丢帧，不回退）

CFG = Config()


# ====== 工具函数：发布一次关节轨迹（初始观察位姿）======
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


# ====== 旋转/四元数工具 ======
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


# ====== 变换(R,p)工具 ======
def tfmsg_to_Rp(transform: TransformStamped) -> Tuple[np.ndarray, np.ndarray]:
    q = transform.transform.rotation
    t = transform.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R_parent_child = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)
    p_parent_child = np.array([t.x, t.y, t.z], dtype=float)
    return R_parent_child, p_parent_child

def compose_Rp(R_ab: np.ndarray, p_ab: np.ndarray,
               R_bc: np.ndarray, p_bc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R_ac = R_ab @ R_bc
    p_ac = R_ab @ p_bc + p_ab
    return R_ac, p_ac


# ====== 主节点 ======
class SeeAnythingVTNode(Node):
    def __init__(self):
        super().__init__("seeanything")

        # 参数声明（含 DINO 与几何）
        self.declare_parameter("image_topic", CFG.IMAGE_TOPIC)
        self.declare_parameter("camera_info_topic", CFG.CAMERA_INFO_TOPIC)
        self.declare_parameter("debug_image_topic", CFG.DEBUG_IMAGE_TOPIC)
        self.declare_parameter("use_camera_info", CFG.USE_CAMERA_INFO)

        self.declare_parameter("fx", CFG.FX); self.declare_parameter("fy", CFG.FY)
        self.declare_parameter("cx", CFG.CX); self.declare_parameter("cy", CFG.CY)

        self.declare_parameter("base_frame", CFG.BASE_FRAME)
        self.declare_parameter("tool_frame", CFG.TOOL_FRAME)
        self.declare_parameter("object_frame", CFG.OBJECT_FRAME)

        self.declare_parameter("z_virt", CFG.Z_VIRT)
        self.declare_parameter("h_above", CFG.H_ABOVE)
        self.declare_parameter("approach_clearance", CFG.APPROACH_CLEARANCE)

        self.declare_parameter("text_prompt", CFG.TEXT_PROMPT)
        self.declare_parameter("model_id", CFG.DINO_MODEL_ID)
        self.declare_parameter("device", CFG.DINO_DEVICE)
        self.declare_parameter("box_threshold", CFG.BOX_THRESHOLD)
        self.declare_parameter("text_threshold", CFG.TEXT_THRESHOLD)

        self.declare_parameter('t_tool_cam_xyz', list(CFG.T_TOOL_CAM_XYZ))
        self.declare_parameter('t_tool_cam_quat', list(CFG.T_TOOL_CAM_QUAT))

        self.declare_parameter("send_init_pose", CFG.SEND_INIT_POSE)

        # TF 对齐策略
        self.declare_parameter("use_latest_tf_on_fail", CFG.USE_LATEST_TF_ON_FAIL)
        self.declare_parameter("strict_time_mode", CFG.STRICT_TIME_MODE)

        # 读取参数
        g = self.get_parameter
        self.image_topic = g("image_topic").value
        self.ci_topic = g("camera_info_topic").value
        self.debug_image_topic = g("debug_image_topic").value
        self.use_ci = bool(g("use_camera_info").value)

        self.fx = float(g('fx').value); self.fy = float(g('fy').value)
        self.cx = float(g('cx').value); self.cy = float(g('cy').value)

        self.base_frame = g('base_frame').value
        self.tool_frame = g('tool_frame').value
        self.object_frame = g('object_frame').value

        self.z_virt = float(g('z_virt').value)
        self.h_above = float(g('h_above').value)
        self.approach_clearance = float(g('approach_clearance').value)

        self.text_prompt = g("text_prompt").value
        self.model_id = g("model_id").value
        self.device = g("device").value
        self.box_th = float(g("box_threshold").value)
        self.text_th = float(g("text_threshold").value)

        self.t_tool_cam_xyz = np.array(g('t_tool_cam_xyz').value, dtype=float)
        self.t_tool_cam_quat = np.array(g('t_tool_cam_quat').value, dtype=float)

        self.send_init_pose = bool(g("send_init_pose").value)

        self.use_latest_tf_on_fail = bool(g("use_latest_tf_on_fail").value)
        self.strict_time_mode = bool(g("strict_time_mode").value)

        # 内参缓存（回退：若不用 /camera_info）
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0,      0,      1]], dtype=float)
        self.D: Optional[np.ndarray] = None
        self.dist_model: Optional[str] = None
        self._have_K = not self.use_ci
        self._warned_no_K = False  # 一次性告警标志

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # DINO 预测器（完全复用你的最小节点写法）
        self.predictor = GroundingDinoPredictor(self.model_id, self.device)

        # ROS 接口
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        if self.use_ci:
            self.sub_ci = self.create_subscription(CameraInfo, self.ci_topic, self._cb_ci, qos)
        self.sub_img = self.create_subscription(Image, self.image_topic, self._cb_image, qos)
        self.pub_dbg = self.create_publisher(Image, self.debug_image_topic, 1)

        # P_top / P_pre
        self.pub_p_top = self.create_publisher(PoseStamped, '/p_top', 1)
        self.pub_p_pre = self.create_publisher(PoseStamped, '/p_pre', 1)

        # 初始观察位姿一次性下发
        self.traj_pub = self.create_publisher(JointTrajectory, JOINT_TRAJ_TOPIC, 10)
        self._init_pose_sent = False
        if self.send_init_pose:
            self.create_timer(0.8, self._send_initial_once)

        self.get_logger().info(
            f"seeanything node started. image_topic={self.image_topic}, prompt='{self.text_prompt}', "
            f"z_virt={self.z_virt:.3f}, h_above={self.h_above:.3f}, object_frame='{self.object_frame}', "
            f"use_latest_tf_on_fail={self.use_latest_tf_on_fail}, strict_time_mode={self.strict_time_mode}"
        )
        self._busy = False

    # 一次性发送初始位姿
    def _send_initial_once(self):
        if self._init_pose_sent:
            return
        publish_trajectory(self, self.traj_pub, POS1, INIT_MOVE_TIME)
        self._init_pose_sent = True

    # CameraInfo 回调
    def _cb_ci(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.K = K.copy()
        self.fx, self.fy, self.cx, self.cy = K[0,0], K[1,1], K[0,2], K[1,2]
        if msg.d is not None and len(msg.d) > 0:
            self.D = np.array(msg.d, dtype=float).reshape(-1)
        else:
            self.D = None
        self.dist_model = getattr(msg, 'distortion_model', None) or None
        self._have_K = True

    # 图像回调：DINO -> (u,v) -> 几何
    def _cb_image(self, msg: Image):
        if self._busy:
            return
        if not self._have_K:
            if not self._warned_no_K:
                self.get_logger().warn('等待 /camera_info 提供内参/畸变。')
                self._warned_no_K = True
            return
        self._busy = True
        try:
            # 1) ROS Image -> PIL
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil = PILImage.fromarray(rgb)

            # 2) DINO 预测（完全照你的接口）
            boxes, labels = self.predictor.predict(
                pil, self.text_prompt, box_threshold=self.box_th, text_threshold=self.text_th
            )
            n = 0 if boxes is None else len(boxes)
            self.get_logger().info(f"DINO detections: {n}")
            if n == 0:
                return

            # 取首个目标中心 (u,v)
            x0, y0, x1, y1 = boxes[0].tolist()
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)

            # 3) 像素 -> 相机归一化光线（考虑畸变）
            if self.D is not None and self.dist_model in (None, '', 'plumb_bob', 'rational_polynomial'):
                pts = np.array([[[u, v]]], dtype=np.float32)
                undist = cv2.undistortPoints(pts, self.K, self.D, P=None)  # -> 1x1x2
                x_n, y_n = float(undist[0, 0, 0]), float(undist[0, 0, 1])
                d_cam_cam = np.array([x_n, y_n, 1.0], dtype=float)
            else:
                x_n = (u - self.cx) / self.fx
                y_n = (v - self.cy) / self.fy
                d_cam_cam = np.array([x_n, y_n, 1.0], dtype=float)
            d_cam_cam /= np.linalg.norm(d_cam_cam)

            # 4) TF: base <- tool（稳健查找）
            lookup_time = rclpy.time.Time.from_msg(msg.header.stamp)
            try:
                ok = self.tf_buffer.can_transform(
                    self.base_frame, self.tool_frame, lookup_time,
                    timeout=RclDuration(seconds=0.2)
                )
                if ok:
                    T_base_tool_msg = self.tf_buffer.lookup_transform(
                        self.base_frame, self.tool_frame, lookup_time
                    )
                else:
                    if self.strict_time_mode:
                        self.get_logger().warn(
                            "TF 对应时间不可用（严格模式），丢弃此帧。"
                        )
                        return
                    if self.use_latest_tf_on_fail:
                        self.get_logger().warn(
                            "TF 对应时间不可用，回退到最新 TF（Time(0)）。"
                        )
                        T_base_tool_msg = self.tf_buffer.lookup_transform(
                            self.base_frame, self.tool_frame, rclpy.time.Time()
                        )
                    else:
                        self.get_logger().warn(
                            "TF 对应时间不可用，未启用回退，丢弃此帧。"
                        )
                        return
            except TransformException as ex:
                if self.strict_time_mode or not self.use_latest_tf_on_fail:
                    self.get_logger().warn(f'TF 未就绪: {ex}；严格/禁用回退，丢弃此帧。')
                    return
                self.get_logger().warn(f'TF 未就绪: {ex}；改用最新 TF（Time(0)）。')
                T_base_tool_msg = self.tf_buffer.lookup_transform(
                    self.base_frame, self.tool_frame, rclpy.time.Time()
                )

            R_base_tool, p_base_tool = tfmsg_to_Rp(T_base_tool_msg)

            # 5) tool->cam（手眼外参，使用 quat）
            R_tool_cam = quat_to_rot(*self.t_tool_cam_quat.tolist())
            p_tool_cam = self.t_tool_cam_xyz

            # 6) 相机在 base 下位姿：T^B_C = T^B_T ∘ T^T_C
            R_base_cam = R_base_tool @ R_tool_cam
            p_base_cam = R_base_tool @ p_tool_cam + p_base_tool

            # 7) 光线转到 base
            d_cam_base = R_base_cam @ d_cam_cam
            d_cam_base /= np.linalg.norm(d_cam_base)
            o_base = p_base_cam

            # 8) 与虚拟平面 z=z_virt 求交
            rz = float(d_cam_base[2])
            if abs(rz) < 1e-6:
                self.get_logger().warn('视线近水平（|r_z|≈0），无法与虚拟平面求交。')
                return
            t_star = (self.z_virt - float(o_base[2])) / rz
            if t_star < 0:
                self.get_logger().warn('平面交点在相机后方（t<0），忽略。')
                return

            C_base = o_base + t_star * d_cam_base  # 交点（物体中心）in base
            n_base = np.array([0.0, 0.0, 1.0], dtype=float)
            P_top_base = C_base + self.h_above * n_base
            P_pre_base = P_top_base + self.approach_clearance * n_base

            # 9) 末端期望姿态：-Z_tool // +Z_base
            z_tool_in_base = -n_base
            x_tool_in_base = np.array([1.0, 0.0, 0.0])  # 任意与 z 不共线
            y_tool_in_base = np.cross(z_tool_in_base, x_tool_in_base)
            if np.linalg.norm(y_tool_in_base) < 1e-6:
                x_tool_in_base = np.array([0.0, 1.0, 0.0])
                y_tool_in_base = np.cross(z_tool_in_base, x_tool_in_base)
            x_tool_in_base /= np.linalg.norm(x_tool_in_base)
            y_tool_in_base /= np.linalg.norm(y_tool_in_base)
            z_tool_in_base /= np.linalg.norm(z_tool_in_base)
            R_base_tool_des = np.column_stack((x_tool_in_base, y_tool_in_base, z_tool_in_base))
            qx, qy, qz, qw = rot_to_quat(R_base_tool_des)

            # 10) 发布 P_top / P_pre
            ps_top = PoseStamped()
            ps_top.header = Header(stamp=msg.header.stamp, frame_id=self.base_frame)
            ps_top.pose.position.x, ps_top.pose.position.y, ps_top.pose.position.z = map(float, P_top_base)
            ps_top.pose.orientation.x, ps_top.pose.orientation.y, ps_top.pose.orientation.z, ps_top.pose.orientation.w = qx, qy, qz, qw
            self.pub_p_top.publish(ps_top)

            ps_pre = PoseStamped()
            ps_pre.header = Header(stamp=msg.header.stamp, frame_id=self.base_frame)
            ps_pre.pose.position.x, ps_pre.pose.position.y, ps_pre.pose.position.z = map(float, P_pre_base)
            ps_pre.pose.orientation.x, ps_pre.pose.orientation.y, ps_pre.pose.orientation.z, ps_pre.pose.orientation.w = qx, qy, qz, qw
            self.pub_p_pre.publish(ps_pre)

            # 11) 发布物体 TF（位于 C_base，姿态与 base 对齐）
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()  # 当前时间戳，RViz 更稳
            t.header.frame_id = self.base_frame
            t.child_frame_id = self.object_frame
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, C_base)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

            self.get_logger().info(
                f"(u,v)=({u:.1f},{v:.1f})  "
                f"C_base=({C_base[0]:.3f},{C_base[1]:.3f},{C_base[2]:.3f})  "
                f"P_top=({P_top_base[0]:.3f},{P_top_base[1]:.3f},{P_top_base[2]:.3f})  "
                f"-> TF:{self.object_frame}"
            )

            # 12) 调试图像可视化并发布
            vis = rgb.copy()
            for label, box in zip(labels, boxes):
                x0i, y0i, x1i, y1i = [int(v) for v in box.tolist()]
                cv2.rectangle(vis, (x0i, y0i), (x1i, y1i), (255, 0, 0), 2)
                cv2.putText(vis, str(label), (x0i, max(0, y0i - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.circle(vis, (int(round(u)), int(round(v))), 5, (0, 255, 0), -1)

            dbg_msg = self.bridge.cv2_to_imgmsg(vis, encoding="rgb8")
            dbg_msg.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            self.pub_dbg.publish(dbg_msg)

        except Exception as e:
            self.get_logger().error(f"DINO/几何处理失败: {e}")
        finally:
            self._busy = False


def main():
    rclpy.init()
    node = SeeAnythingVTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
