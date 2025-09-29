#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
detect_and_circle_scan_vggt.py — GroundingDINO 定位 + UR5e 连续/离散圆周采样 + 图像保存 + VGGT 重建
配置风格：关键参数在本文件顶部直接修改，无需 ros 参数传入
"""

# ================== 手动配置区 ==================

# ---- 话题 / 坐标系 ----
IMAGE_TOPIC = '/my_camera/pylon_ros2_camera_node/image_raw'
BASE_FRAME  = 'base_link'
TOOL_FRAME  = 'tool0'
OBJECT_FRAME = 'object_position'
POSE_FRAME  = 'base_link'
Z_VIRT      = 0.0  # 工作平面高度（base 下）

# ---- 相机内参（像素） ----
import numpy as np
FX = 2674.3803723910564
FY = 2667.4211254043507
CX = 954.5922081613583
CY = 1074.965947832258

# ---- 手眼外参：tool -> camera_(optical/link) ----
T_TOOL_CAM_XYZ  = np.array([-0.000006852374024, -0.099182661943126947, 0.02391824813032688], dtype=float)
# qx,qy,qz,qw
T_TOOL_CAM_QUAT = np.array([-0.0036165657530785695, -0.000780788838366878,
                            0.7078681983794892, 0.7063348529868249], dtype=float)
HAND_EYE_FRAME  = 'optical'  # 'optical' 或 'link'

# ---- GroundingDINO ----
TEXT_PROMPT    = 'orange object .'
DINO_MODEL_ID  = 'IDEA-Research/grounding-dino-tiny'
DINO_DEVICE    = 'cuda'
BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.25

# ---- 运行时开关 ----
TF_TIME_MODE   = 'latest'   # 'image' | 'latest'
FRAME_STRIDE   = 2
DEBUG_WINDOW   = False
DRAW_BEST_BOX  = False
DEBUG_HZ       = 5.0
TF_REBROADCAST_HZ = 20.0
FLIP_X = False
FLIP_Y = False

# ---- object_position 偏差（XY/Z 硬补偿，手动设置）----
ENABLE_BASE_BIAS = True
BIAS_BASE_X = -0.10   # +X 前/右（m）
BIAS_BASE_Y = -0.15   # +Y 左/侧（m）
BIAS_BASE_Z = 0.00    # Z 平面偏差（m）

# ---- 机械臂运动 / 控制 ----
GROUP_NAME = 'ur_manipulator'
EE_LINK = 'tool0'
IK_TIMEOUT = 2.0
IK_ATTEMPTS = 1
CONTROLLER_TOPIC = '/scaled_joint_trajectory_controller/joint_trajectory'

USE_INITIAL_POSE = True
INITIAL_JOINT_POSITIONS = [0.92, -1.18, 1.19, -1.57, -1.57, -0.57]
MOVE_TIME = 3.0
SETTLE_TIME = 0.8

# ---- 圆周参数 & 采样控制 ----
HOVER_ABOVE = 0.40           # 悬停高度：P_top.z = Z_VIRT + HOVER_ABOVE
RING_ABOVE  = 0.40           # 圆周平面高度：Z = Z_VIRT + RING_ABOVE
CIRCLE_RADIUS = 0.25         # 圆周半径（m）
FACE_CENTER_TOOL_Z = '+Z'    # '+Z' | '-Z'，工具 Z 轴指向目标中心

# 连续圆周采样配置
STOP_AND_GO = False          # False=连续轨迹+途中采样；True=逐点停—拍—走（旧逻辑）
CAPTURE_MODE = 'angle'       # 'angle' | 'time'
NUM_VIEWS = 10               # 拍几张（角度/时间两种模式都用到）
START_ANGLE_DEG = 0.0        # 相对“+X”方向的起始角（0° 在 center 右侧；逆时针为正）
CIRCLE_DIR = 'ccw'           # 'ccw'（逆时针）或 'cw'（顺时针）
TRAJ_STEP_DEG = 3.0          # 连续轨迹角步长（越小越平滑，IK 次数越多）
FULL_TURN_TIME = 20.0        # 整圈时长（秒），用于连续运动和按时间采样
CAPTURE_POLL_HZ = 50.0       # 角度模式：查询 TF 的频率（Hz）
MIN_DEG_GAP = 5.0            # 角度模式：相邻触发至少相差多少角度（防抖）

# ---- 数据输出 ----
SCENE_DIR = '/tmp/ur5e_circle_scene'
SAVE_JOINTS_JSON = True
DO_RECONSTRUCTION = True

# ---- 关节名称顺序（UR5e）----
JOINT_ORDER = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# ================== 代码区==================

from typing import Optional, Tuple, List
import math
import os
import time
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration as RclDuration
from rclpy.time import Time as RclTime

from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import TransformStamped, PoseStamped
from builtin_interfaces.msg import Duration as MsgDuration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK

import tf2_ros
from tf2_ros import TransformException
from cv_bridge import CvBridge
import cv2

# DINO
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except Exception:
    import sys
    sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
    from gripanything.core.detect_with_dino import GroundingDinoPredictor

# VGGT（按你的路径）
VGGT_AVAILABLE = True
try:
    from gripanything.core.vggtreconstruction import VGGTReconstructor
except Exception as e:
    VGGT_AVAILABLE = False
    _import_err = str(e)


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


def tfmsg_to_Rp(transform: TransformStamped) -> Tuple[np.ndarray, np.ndarray]:
    q = transform.transform.rotation
    t = transform.transform.translation
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    p = np.array([t.x, t.y, t.z], dtype=float)
    return R, p


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def quat_from_rotmat(R):
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def look_at_quaternion(eye, target, up=(0, 0, 1), tool_z_sign='+'):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = normalize(up)

    z_dir = normalize(target - eye)
    if tool_z_sign.strip() == '-':
        z_dir = -z_dir

    if abs(np.dot(up, z_dir)) > 0.98:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    x_axis = normalize(np.cross(up, z_dir))
    y_axis = normalize(np.cross(z_dir, x_axis))
    z_axis = z_dir
    R = np.column_stack([x_axis, y_axis, z_axis])
    return quat_from_rotmat(R)  # [w,x,y,z]


# camera_link <- camera_optical（REP-105）
R_CL_CO = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0]
], dtype=float)


class DetectAndCircleScanVGGT(Node):
    def __init__(self):
        super().__init__('detect_and_circle_scan_vggt')

        # 保存配置为成员
        self.image_topic = IMAGE_TOPIC
        self.base_frame  = BASE_FRAME
        self.tool_frame  = TOOL_FRAME
        self.object_frame = OBJECT_FRAME
        self.pose_frame  = POSE_FRAME
        self.Z_VIRT      = float(Z_VIRT)

        self.FX, self.FY, self.CX, self.CY = float(FX), float(FY), float(CX), float(CY)

        self.T_TOOL_CAM_XYZ  = T_TOOL_CAM_XYZ.astype(float)
        self.T_TOOL_CAM_QUAT = T_TOOL_CAM_QUAT.astype(float)
        self.HAND_EYE_FRAME  = HAND_EYE_FRAME

        self.TEXT_PROMPT   = TEXT_PROMPT
        self.DINO_MODEL_ID = DINO_MODEL_ID
        self.DINO_DEVICE   = DINO_DEVICE
        self.BOX_THRESHOLD = float(BOX_THRESHOLD)
        self.TEXT_THRESHOLD = float(TEXT_THRESHOLD)

        self.TF_TIME_MODE = TF_TIME_MODE
        self.FRAME_STRIDE = int(FRAME_STRIDE)
        self.DEBUG_WINDOW = bool(DEBUG_WINDOW)
        self.DRAW_BEST_BOX = bool(DRAW_BEST_BOX)
        self.DEBUG_HZ = float(DEBUG_HZ)

        self.ENABLE_BASE_BIAS = bool(ENABLE_BASE_BIAS)
        self.BIAS_BASE_X = float(BIAS_BASE_X)
        self.BIAS_BASE_Y = float(BIAS_BASE_Y)
        self.BIAS_BASE_Z = float(BIAS_BASE_Z)

        self.FLIP_X = bool(FLIP_X)
        self.FLIP_Y = bool(FLIP_Y)

        self.group_name = GROUP_NAME
        self.ee_link = EE_LINK
        self.ik_timeout = float(IK_TIMEOUT)
        self.ik_attempts = int(IK_ATTEMPTS)
        self.controller_topic = CONTROLLER_TOPIC

        self.use_initial_pose = bool(USE_INITIAL_POSE)
        self.initial_joint_positions = list(INITIAL_JOINT_POSITIONS)
        self.move_time = float(MOVE_TIME)
        self.settle_time = float(SETTLE_TIME)

        self.hover_above = float(HOVER_ABOVE)
        self.ring_above  = float(RING_ABOVE)
        self.radius      = float(CIRCLE_RADIUS)
        self.face_center_tool_z = FACE_CENTER_TOOL_Z

        self.stop_and_go = bool(STOP_AND_GO)
        self.capture_mode = CAPTURE_MODE
        self.num_views = int(NUM_VIEWS)
        self.start_angle_deg = float(START_ANGLE_DEG)
        self.circle_dir = CIRCLE_DIR
        self.traj_step_deg = float(TRAJ_STEP_DEG)
        self.full_turn_time = float(FULL_TURN_TIME)
        self.capture_poll_hz = float(CAPTURE_POLL_HZ)
        self.min_deg_gap = float(MIN_DEG_GAP)

        self.scene_dir = SCENE_DIR
        self.save_joints_json = bool(SAVE_JOINTS_JSON)
        self.do_reconstruction = bool(DO_RECONSTRUCTION)
        self.joint_order = list(JOINT_ORDER)

        # 输出目录
        self.images_dir = os.path.join(self.scene_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)

        # 订阅
        qos_img = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
        qos_js  = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, self.image_topic, self._cb_image, qos_img)
        self.sub_js  = self.create_subscription(JointState, '/joint_states', self._cb_joint, qos_js)

        self.last_image_msg: Optional[Image] = None
        self.last_joint_state: Optional[JointState] = None

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # DINO
        self.predictor = GroundingDinoPredictor(self.DINO_MODEL_ID, self.DINO_DEVICE)

        # 预计算 tool <- camera_optical/link
        R_t_cam = quat_to_rot(*self.T_TOOL_CAM_QUAT.tolist())
        self.R_t_co = R_t_cam if self.HAND_EYE_FRAME.lower() == 'optical' else (R_t_cam @ R_CL_CO)
        self.p_t_co = self.T_TOOL_CAM_XYZ

        # 控制发布者
        self.traj_pub = self.create_publisher(JointTrajectory, self.controller_topic, 10)

        # IK 客户端（延迟发现 + 兜底服务名）
        self._ik_client = None
        self._ik_candidates = ['/compute_ik', '/move_group/compute_ik']

        # 状态控制
        self._frame_count = 0
        self._last_debug_pub = self.get_clock().now()
        self._last_good_tf: Optional[TransformStamped] = None

        # 管线控制
        self._pipeline_started = False
        self._object_center_base: Optional[np.ndarray] = None

        # 定时器
        if TF_REBROADCAST_HZ > 0:
            self._reb_timer = self.create_timer(1.0 / TF_REBROADCAST_HZ, self._rebroadcast_tf)
        self._main_timer = self.create_timer(0.5, self._run_once)

        self.get_logger().info(
            f'[detect_and_circle_scan_vggt] topic={self.image_topic}, prompt="{self.TEXT_PROMPT}", '
            f'BIAS=({self.BIAS_BASE_X:.2f},{self.BIAS_BASE_Y:.2f},{self.BIAS_BASE_Z:.2f}), '
            f'mode={"STOP&GO" if self.stop_and_go else "CONTINUOUS"}, '
            f'capture={self.capture_mode}, views={self.num_views}, radius={self.radius:.3f}, '
            f'scene_dir={self.scene_dir}'
        )

        if self.DEBUG_WINDOW:
            try:
                cv2.namedWindow("DINO Debug", cv2.WINDOW_NORMAL)
            except Exception:
                self.get_logger().warn("无法创建可视化窗口。")

    # -------------------- Callbacks --------------------
    def _cb_image(self, msg: Image):
        self._frame_count += 1
        if self.FRAME_STRIDE > 1 and (self._frame_count % self.FRAME_STRIDE) != 0:
            return
        self.last_image_msg = msg

    def _cb_joint(self, msg: JointState):
        self.last_joint_state = msg

    def _rebroadcast_tf(self):
        if self._last_good_tf is None:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._last_good_tf.header.frame_id
        t.child_frame_id  = self._last_good_tf.child_frame_id
        t.transform = self._last_good_tf.transform
        self.tf_broadcaster.sendTransform(t)

    # -------------------- Helpers --------------------
    def _ensure_ik_client(self) -> bool:
        if self._ik_client is not None:
            return True
        for name in self._ik_candidates:
            cli = self.create_client(GetPositionIK, name)
            if cli.wait_for_service(timeout_sec=0.5):
                self._ik_client = cli
                self.get_logger().info(f'IK 服务可用：{name}')
                return True
        self.get_logger().warning(f'等待 IK 服务…（尝试：{self._ik_candidates}）')
        return False

    def get_current_tool_quat_wxyz(self) -> Optional[np.ndarray]:
        try:
            T_bt = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, rclpy.time.Time(),
                                                   timeout=RclDuration(seconds=0.2))
        except TransformException:
            return None
        q = T_bt.transform.rotation
        return np.array([q.w, q.x, q.y, q.z], dtype=float)

    def wait_for_joint_states(self, timeout=10.0):
        t0 = time.time()
        while rclpy.ok() and self.last_joint_state is None and (time.time() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_joint_state is not None

    def wait_for_image(self, timeout=5.0):
        t0 = time.time()
        cur_seen = self._frame_count
        while rclpy.ok() and (self.last_image_msg is None or self._frame_count == cur_seen) and (time.time() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.last_image_msg is not None

    def wait_seconds(self, sec: float):
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < sec:
            rclpy.spin_once(self, timeout_sec=0.05)

    def build_pose(self, p: np.ndarray, q_wxyz: np.ndarray) -> PoseStamped:
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.pose_frame
        ps.pose.position.x = float(p[0])
        ps.pose.position.y = float(p[1])
        ps.pose.position.z = float(p[2])
        ps.pose.orientation.w = float(q_wxyz[0])
        ps.pose.orientation.x = float(q_wxyz[1])
        ps.pose.orientation.y = float(q_wxyz[2])
        ps.pose.orientation.z = float(q_wxyz[3])
        return ps

    def call_ik(self, pose_stamped: PoseStamped) -> Optional[JointState]:
        if self.last_joint_state is None:
            return None
        if not self._ensure_ik_client():
            return None

        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = self.group_name
        ik.ik_link_name = self.ee_link
        ik.pose_stamped = pose_stamped
        ik.robot_state.joint_state = self.last_joint_state  # 用当前 /joint_states 作为种子
        ik.avoid_collisions = False
        ik.timeout = MsgDuration(
            sec=int(self.ik_timeout),
            nanosec=int((self.ik_timeout - int(self.ik_timeout)) * 1e9)
        )
        # 某些分支可能有 attempts 字段；Humble 官方无此字段
        if hasattr(ik, 'attempts'):
            try:
                setattr(ik, 'attempts', int(self.ik_attempts))
            except Exception:
                pass

        fut = self._ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.ik_timeout + 2.0)
        if not fut.done() or fut.result() is None:
            return None
        res = fut.result()
        if getattr(res.error_code, 'val', 0) != 1:
            return None
        return res.solution.joint_state

    def joints_from_solution(self, sol_js: JointState) -> Optional[List[float]]:
        m = {n: p for n, p in zip(sol_js.name, sol_js.position)}
        positions = []
        for n in self.joint_order:
            if n not in m:
                self.get_logger().warn(f'IK 解中缺少关节 {n}')
                return None
            positions.append(float(m[n]))
        return positions

    def publish_trajectory_point(self, positions, move_time):
        jt = JointTrajectory()
        jt.joint_names = self.joint_order
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = MsgDuration(
            sec=int(move_time), nanosec=int((move_time - int(move_time)) * 1e9)
        )
        jt.points.append(pt)
        self.traj_pub.publish(jt)

    def publish_trajectory(self, joint_waypoints: List[List[float]], full_time: float):
        """一次性发布整条 JointTrajectory，均匀时间分布"""
        if len(joint_waypoints) < 2:
            self.get_logger().error('轨迹点过少，无法发布。')
            return
        jt = JointTrajectory()
        jt.joint_names = self.joint_order
        total_pts = len(joint_waypoints)
        for i, pos in enumerate(joint_waypoints):
            pt = JointTrajectoryPoint()
            pt.positions = pos
            t = (i / (total_pts - 1)) * full_time
            pt.time_from_start = MsgDuration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
            jt.points.append(pt)
        self.traj_pub.publish(jt)

    def save_current_image(self, path: str):
        if self.last_image_msg is None:
            self.get_logger().warn('没有图像可保存。')
            return False
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.last_image_msg, desired_encoding='bgr8')
            return bool(cv2.imwrite(path, cv_img))
        except Exception as e:
            self.get_logger().error(f'保存图像失败: {e}')
            return False

    # -------------------- Detection --------------------
    def detect_object_once(self, max_tries=10) -> Optional[np.ndarray]:
        tries = 0
        while rclpy.ok() and tries < max_tries:
            tries += 1
            if not self.wait_for_image(timeout=2.0):
                continue
            msg = self.last_image_msg
            try:
                rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            except Exception as e:
                self.get_logger().warn(f'图像转换失败：{e}')
                continue

            from PIL import Image as PILImage
            pil = PILImage.fromarray(rgb)
            out = self.predictor.predict(
                pil, self.TEXT_PROMPT,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD
            )
            if isinstance(out, tuple) and len(out) == 3:
                boxes, labels, scores = out
            elif isinstance(out, tuple) and len(out) == 2:
                boxes, labels = out
                scores = [None] * len(boxes)
            else:
                self.get_logger().warn("DINO 返回格式不支持。")
                continue
            if len(boxes) == 0:
                self.get_logger().info("未检测到目标，重试。")
                continue

            s = np.array([float(s) if s is not None else -1.0 for s in scores])
            best = int(np.argmax(s))
            bx = boxes[best]
            x0, y0, x1, y1 = (bx.tolist() if hasattr(bx, 'tolist') else bx)
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)
            sc = s[best]

            x_n = (u - self.CX) / self.FX
            y_n = (v - self.CY) / self.FY
            if self.FLIP_X: x_n = -x_n
            if self.FLIP_Y: y_n = -y_n
            d_opt = np.array([x_n, y_n, 1.0], dtype=float)
            d_opt /= np.linalg.norm(d_opt)

            t_query = RclTime.from_msg(msg.header.stamp) if self.TF_TIME_MODE == 'image' else RclTime()
            try:
                T_bt = self.tf_buffer.lookup_transform(
                    self.base_frame, self.tool_frame, t_query, timeout=RclDuration(seconds=0.2)
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF 查找失败（{self.TF_TIME_MODE}, base<-tool）：{ex}")
                continue
            R_bt, p_bt = tfmsg_to_Rp(T_bt)

            # 相机位姿（base 下）
            R_bc = R_bt @ self.R_t_co
            p_bc = R_bt @ self.p_t_co + p_bt

            d_base = R_bc @ d_opt
            nrm = np.linalg.norm(d_base)
            if nrm < 1e-9:
                self.get_logger().warn("方向向量异常。")
                continue
            d_base /= nrm
            o_base = p_bc

            dz = float(d_base[2])
            if abs(dz) < 1e-6:
                self.get_logger().warn("视线近水平，无法与平面求交。")
                continue
            t_star = (self.Z_VIRT - float(o_base[2])) / dz
            if t_star < 0:
                self.get_logger().warn("交点在相机后方，忽略。")
                continue
            C_raw = o_base + t_star * d_base
            C = C_raw.copy()
            if self.ENABLE_BASE_BIAS:
                C[0] += self.BIAS_BASE_X
                C[1] += self.BIAS_BASE_Y
                C[2] += self.BIAS_BASE_Z

            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = self.base_frame
            tf_msg.child_frame_id  = self.object_frame
            tf_msg.transform.translation.x = float(C[0])
            tf_msg.transform.translation.y = float(C[1])
            tf_msg.transform.translation.z = float(C[2])
            tf_msg.transform.rotation.x = 0.0
            tf_msg.transform.rotation.y = 0.0
            tf_msg.transform.rotation.z = 0.0
            tf_msg.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(tf_msg)
            self._last_good_tf = tf_msg

            self.get_logger().info(
                f"[detect] score={(sc if sc>=0 else float('nan')):.3f} uv=({u:.1f},{v:.1f}) "
                f"C_raw=({C_raw[0]:.3f},{C_raw[1]:.3f},{C_raw[2]:.3f}) "
                f"C_corr=({C[0]:.3f},{C[1]:.3f},{C[2]:.3f}) -> {self.object_frame}"
            )

            # 可选可视化
            if self.DEBUG_WINDOW and self.DRAW_BEST_BOX:
                now = self.get_clock().now()
                if (now - self._last_debug_pub).nanoseconds >= int(1e9 / self.DEBUG_HZ):
                    dbg = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
                    x0i, y0i, x1i, y1i = map(int, [x0, y0, x1, y1])
                    txt = f"{sc:.2f}" if sc >= 0 else ""
                    cv2.rectangle(dbg, (x0i, y0i), (x1i, y1i), 2)
                    if txt:
                        cv2.putText(dbg, txt, (x0i, max(0, y0i-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.circle(dbg, (int(round(u)), int(round(v))), 5, -1)
                    cv2.imshow("DINO Debug", dbg)
                    cv2.waitKey(1)
                    self._last_debug_pub = now

            return C
        return None

    # -------------------- 连续圆周：轨迹与采样 --------------------
    def plan_circle_trajectory_joints(self, C: np.ndarray, ring_z: float,
                                      start_deg: float, step_deg: float,
                                      direction: str) -> Tuple[List[float], List[List[float]]]:
        """返回: (角度列表[deg], 对应的关节解列表)；遇到 IK 失败会尝试姿态回退，否则跳过该角度"""
        if direction.lower() == 'cw':
            angles = np.arange(start_deg, start_deg - 360.0 - 1e-6, -step_deg)
        else:
            angles = np.arange(start_deg, start_deg + 360.0 + 1e-6,  step_deg)
        if len(angles) == 0:
            angles = np.array([start_deg])

        keep_angles, joints_list = [], []
        # 获取当前 tool 姿态，供回退时使用
        q_keep = self.get_current_tool_quat_wxyz()

        for th in angles:
            rad = math.radians(th)
            px = C[0] + self.radius * math.cos(rad)
            py = C[1] + self.radius * math.sin(rad)
            pz = ring_z
            pos = np.array([px, py, pz], dtype=np.float64)

            # 首选：面向中心
            q = look_at_quaternion(eye=pos, target=C, up=(0,0,1), tool_z_sign=self.face_center_tool_z)
            ps = self.build_pose(pos, q)
            sol_js = self.call_ik(ps)

            # 回退：保持当前姿态
            if sol_js is None and q_keep is not None:
                ps_keep = self.build_pose(pos, q_keep)
                sol_js = self.call_ik(ps_keep)

            if sol_js is None:
                self.get_logger().warn(f'圆周角 {th:.1f}° IK 失败，跳过')
                continue
            joints = self.joints_from_solution(sol_js)
            if joints is None:
                self.get_logger().warn(f'圆周角 {th:.1f}° 关节映射失败，跳过')
                continue
            keep_angles.append(th)
            joints_list.append(joints)
        return keep_angles, joints_list

    def _norm_angle_deg(self, a):
        a = (a % 360.0 + 360.0) % 360.0
        return a

    def _compute_theta_deg(self, C: np.ndarray) -> Optional[float]:
        """读取 base->tool TF，计算相对 C 的极角（+X 为 0°，逆时针为正）"""
        try:
            T_bt = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, rclpy.time.Time(),
                                                   timeout=RclDuration(seconds=0.05))
        except TransformException:
            return None
        R_bt, p_bt = tfmsg_to_Rp(T_bt)
        dx, dy = float(p_bt[0] - C[0]), float(p_bt[1] - C[1])
        th = math.degrees(math.atan2(dy, dx))
        return self._norm_angle_deg(th)

    def start_angle_capture(self, C: np.ndarray, start_deg: float, num_views: int, dir_ccw: bool):
        """按角度等分触发保存（连续运动途中均匀拍照）"""
        self._cap_center = C.copy()
        self._cap_delta = 360.0 / float(num_views)
        self._cap_saved_idx = -1
        self._cap_last_prog = None
        self._cap_dir_ccw = dir_ccw
        self._cap_start = self._norm_angle_deg(start_deg)
        self._cap_done = False

        def tick():
            if self._cap_done:
                return
            th = self._compute_theta_deg(self._cap_center)
            if th is None:
                return
            # CCW: prog = (th - start) mod 360； CW: prog = (start - th) mod 360
            if self._cap_dir_ccw:
                prog = self._norm_angle_deg(th - self._cap_start)
            else:
                prog = self._norm_angle_deg(self._cap_start - th)

            idx = int(prog // self._cap_delta)
            if idx >= num_views:
                idx = num_views - 1

            if self._cap_last_prog is not None and abs(prog - self._cap_last_prog) < self.min_deg_gap:
                return

            if idx > self._cap_saved_idx:
                img_path = os.path.join(self.images_dir, f'view_{idx:02d}.png')
                if self.save_current_image(img_path):
                    self.get_logger().info(f'[capture-angle] #{idx+1}/{num_views} -> {img_path} (θ≈{th:.1f}°)')
                    self._cap_saved_idx = idx
                    self._cap_last_prog = prog
                    if self._cap_saved_idx >= num_views - 1:
                        self._cap_done = True
                        self._cap_timer.cancel()
            else:
                self._cap_last_prog = prog

        self._cap_timer = self.create_timer(1.0/float(self.capture_poll_hz), tick)

    def start_time_capture(self, num_views: int, full_time: float):
        """按时间等分触发保存（连续运动途中均匀拍照）"""
        self._cap_saved = 0
        interval = full_time / float(num_views)
        def tick():
            if self._cap_saved >= num_views:
                self._time_cap_timer.cancel()
                return
            img_path = os.path.join(self.images_dir, f'view_{self._cap_saved:02d}.png')
            if self.save_current_image(img_path):
                self.get_logger().info(f'[capture-time] #{self._cap_saved+1}/{num_views} -> {img_path}')
                self._cap_saved += 1
        self._time_cap_timer = self.create_timer(interval, tick)

    # -------------------- Main Pipeline --------------------
    def _run_once(self):
        if self._pipeline_started:
            return
        self._pipeline_started = True
        self._main_timer.cancel()

        if not self.wait_for_joint_states(timeout=10.0):
            self.get_logger().error('未收到 /joint_states，退出。')
            return

        if self.use_initial_pose and len(self.initial_joint_positions) == len(self.joint_order):
            self.get_logger().info('发布初始观察位...')
            self.publish_trajectory_point(self.initial_joint_positions, self.move_time)
            self.wait_seconds(self.move_time + self.settle_time)

        # 检测一次，得到 base 下物体中心点 C
        self.get_logger().info('开始 GroundingDINO 检测...')
        C = self.detect_object_once(max_tries=20)
        if C is None:
            self.get_logger().error('检测失败，管线中止。')
            return
        self._object_center_base = C.copy()

        # 悬停位姿（先“面向中心”，失败则回退“保持当前姿态”）
        hover_z = self.Z_VIRT + self.hover_above
        p_top = np.array([C[0], C[1], hover_z], dtype=np.float64)
        q_top = look_at_quaternion(eye=p_top, target=C, up=(0, 0, 1), tool_z_sign=self.face_center_tool_z)
        ps_top = self.build_pose(p_top, q_top)
        sol_js = self.call_ik(ps_top)
        if sol_js is None:
            q_keep = self.get_current_tool_quat_wxyz()
            if q_keep is not None:
                ps_top_keep = self.build_pose(p_top, q_keep)
                sol_js = self.call_ik(ps_top_keep)
                if sol_js is None:
                    self.get_logger().error('悬停位姿 IK 失败（含保持姿态回退）。')
                else:
                    self.get_logger().warn('悬停位姿：使用“保持当前姿态”回退方案。')
            else:
                self.get_logger().error('悬停位姿 IK 失败，且无法获取当前姿态。')

        if sol_js is not None:
            joints = self.joints_from_solution(sol_js)
            if joints is None:
                self.get_logger().error('悬停位姿 关节映射失败。')
            else:
                self.get_logger().info(f'执行悬停位姿，move_time={self.move_time:.2f}s')
                self.publish_trajectory_point(joints, self.move_time)
                self.wait_seconds(self.move_time + self.settle_time)

        ring_z = self.Z_VIRT + self.ring_above

        if self.stop_and_go:
            # ===== 老逻辑：逐点停—拍—走 =====
            self.get_logger().info('STOP_AND_GO=true：按等分角逐点采集。')
            joints_log = []
            q_keep = self.get_current_tool_quat_wxyz()
            for i in range(self.num_views):
                th = self.start_angle_deg + (360.0 / self.num_views) * i * (1.0 if self.circle_dir.lower()=='ccw' else -1.0)
                rad = math.radians(th)
                pos = np.array([C[0] + self.radius * math.cos(rad),
                                C[1] + self.radius * math.sin(rad),
                                ring_z], dtype=np.float64)
                # 先面向中心，失败回退保持姿态
                q = look_at_quaternion(eye=pos, target=C, up=(0,0,1), tool_z_sign=self.face_center_tool_z)
                ps = self.build_pose(pos, q)
                sol_js = self.call_ik(ps)
                if sol_js is None and q_keep is not None:
                    ps_keep = self.build_pose(pos, q_keep)
                    sol_js = self.call_ik(ps_keep)
                if sol_js is None:
                    self.get_logger().error(f'角 {th:.1f}° IK 失败，跳过。')
                    continue
                joints = self.joints_from_solution(sol_js)
                if joints is None:
                    self.get_logger().error(f'角 {th:.1f}° 关节映射失败，跳过。')
                    continue

                self.publish_trajectory_point(joints, self.move_time)
                self.wait_seconds(self.move_time + self.settle_time)

                img_path = os.path.join(self.images_dir, f'view_{i:02d}.png')
                if self.save_current_image(img_path):
                    self.get_logger().info(f'[stop&go] #{i+1}/{self.num_views} -> {img_path}')
                if self.save_joints_json:
                    joints_log.append({
                        'index': i, 'theta_deg': float(th),
                        'pos': pos.tolist(), 'quat_wxyz': q.tolist(),
                        'joint_order': self.joint_order, 'joint_positions': joints,
                    })
            if self.save_joints_json and joints_log:
                meta_path = os.path.join(self.scene_dir, 'circle_scan_meta.json')
                os.makedirs(self.scene_dir, exist_ok=True)
                with open(meta_path, 'w') as f:
                    json.dump({'center': C.tolist(), 'Z_VIRT': self.Z_VIRT,
                               'ring_above': self.ring_above, 'radius': self.radius,
                               'mode': 'stop_and_go', 'views': joints_log}, f, indent=2)
                self.get_logger().info(f'已保存采集元数据: {meta_path}')

        else:
            # ===== 连续圆周 + 均匀采样（按角度或时间）=====
            # 1) 先到起始角 START_ANGLE_DEG 的圆周起点（先面向中心，失败则保持当前姿态）
            th0 = self.start_angle_deg
            rad0 = math.radians(th0)
            p0 = np.array([C[0] + self.radius * math.cos(rad0),
                           C[1] + self.radius * math.sin(rad0),
                           ring_z], dtype=np.float64)
            q0 = look_at_quaternion(eye=p0, target=C, up=(0,0,1), tool_z_sign=self.face_center_tool_z)
            ps0 = self.build_pose(p0, q0)
            sol0 = self.call_ik(ps0)
            if sol0 is None:
                q_keep = self.get_current_tool_quat_wxyz()
                if q_keep is not None:
                    ps0_keep = self.build_pose(p0, q_keep)
                    sol0 = self.call_ik(ps0_keep)
                    if sol0 is None:
                        self.get_logger().error('起始角 IK 仍无解（含保持姿态回退），退出。')
                        return
                    else:
                        self.get_logger().warn('起始角：使用“保持当前姿态”回退方案。')
                else:
                    self.get_logger().error('起始角 IK 失败，且无法获取当前姿态。')
                    return
            j0 = self.joints_from_solution(sol0)
            if j0 is None:
                self.get_logger().error('起始角 关节映射失败，退出。')
                return
            self.get_logger().info(f'移动到圆周起点 {self.start_angle_deg:.1f}°...')
            self.publish_trajectory_point(j0, self.move_time)
            self.wait_seconds(self.move_time + self.settle_time)

            # 2) 规划整圈平滑轨迹（带姿态回退）
            self.get_logger().info('生成连续圆周轨迹...')
            angles, joints_list = self.plan_circle_trajectory_joints(
                C, ring_z,
                start_deg=self.start_angle_deg,
                step_deg=self.traj_step_deg,
                direction=self.circle_dir
            )
            if len(joints_list) < 2:
                self.get_logger().error('有效轨迹点不足，无法执行连续圆周。')
                return

            # 当前关节状态作为轨迹第 0 点
            cur_map = {n:p for n,p in zip(self.last_joint_state.name, self.last_joint_state.position)}
            cur_j = []
            for n in self.joint_order:
                if n not in cur_map:
                    self.get_logger().error(f'当前关节状态缺少 {n}')
                    return
                cur_j.append(float(cur_map[n]))
            joint_waypoints = [cur_j] + joints_list

            # 3) 启动采样定时器
            if self.capture_mode == 'angle':
                self.start_angle_capture(C, start_deg=self.start_angle_deg,
                                         num_views=self.num_views,
                                         dir_ccw=(self.circle_dir.lower()=='ccw'))
            else:
                self.start_time_capture(num_views=self.num_views, full_time=self.full_turn_time)

            # 4) 发布整条轨迹并等待结束
            self.get_logger().info(f'发布连续圆周轨迹（{len(joint_waypoints)} 点，{self.full_turn_time:.1f}s）...')
            self.publish_trajectory(joint_waypoints, full_time=self.full_turn_time)
            self.wait_seconds(self.full_turn_time + 1.0)

            # 角度模式：如未满额，再给一点时间补齐
            if self.capture_mode == 'angle':
                t0 = time.time()
                while getattr(self, '_cap_done', True) is False and (time.time() - t0) < 3.0:
                    self.wait_seconds(0.05)

        # ===== 重建 =====
        if self.do_reconstruction:
            if not VGGT_AVAILABLE:
                self.get_logger().error(f'未找到 VGGTReconstructor：{_import_err}')
            else:
                try:
                    import torch
                    self.get_logger().info('开始 VGGT 重建...')
                    reconstructor = VGGTReconstructor(
                        scene_dir=self.scene_dir,
                        batch_size=4,
                        max_points=100000,
                        resolution=518,
                        conf_thresh=3.5,
                        img_limit=None
                    )
                    with torch.no_grad():
                        reconstructor.run()
                    self.get_logger().info('VGGT 重建完成。')
                except Exception as e:
                    self.get_logger().error(f'VGGT 重建失败: {e}')

        self.get_logger().info('流程结束。')

    def destroy_node(self):
        try:
            if self.DEBUG_WINDOW:
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = DetectAndCircleScanVGGT()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
