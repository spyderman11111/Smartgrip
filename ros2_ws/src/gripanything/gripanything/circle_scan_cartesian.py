#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything_circle_once.py — 一次检测 + 一圈圆周扫描（ROS 2 Humble / UR5e）

流程：
  1) 发布“初始关节位姿”→ 等到位且静止；
  2) GroundingDINO 仅在静止时检测一次，使用【图像时间戳】查询 base<-tool0 TF；
  3) 像素射线与 z=Z_VIRT 求交得圆心 C(base)，叠加 bias；
  4) 设定圆周平面高度 = C.z + HOVER_ABOVE；
  5) 生成圆周轨迹（仅绕世界Z调整 yaw，使相机视线XY指向圆心；tool-Z 与 world-Z 对齐，sign 可选 ±）；
  6) （可选）先对“圆周起点”做一次 IK 并发送关节目标；
  7) 调用 /compute_cartesian_path 生成整圈轨迹 → 均匀时间化 → 发布到 scaled_joint_trajectory_controller；
  8) 持续重广播 object_position（RViz 可见）及 tool0->camera_{link,optical}；完成后退出。

说明：
  - 支持 /image_raw (用 K+D 去畸变) 与 /image_rect (用 P，无畸变)。
  - 手眼外参使用你给定的 tool<-camera_(optical/link)。圆周 yaw 求解自动使用该外参推得相机光轴与平移（在工具系下）。
  - 关键参数集中在“手动配置区”。可通过 --ros-args -p 覆盖。
"""

from typing import Optional, Dict, List, Tuple
import math
import numpy as np
import cv2
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration as RclDuration
from rclpy.time import Time
from builtin_interfaces.msg import Duration as MsgDuration

from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_ros
from tf2_ros import TransformException
from cv_bridge import CvBridge
from PIL import Image as PILImage

from moveit_msgs.srv import GetPositionIK, GetCartesianPath

# ==========================
# 手动配置区
# ==========================
# --- 话题 / 坐标系 ---
IMAGE_TOPIC        = "/my_camera/pylon_ros2_camera_node/image_raw"  # 若用整列图，改为 /image_rect
BASE_FRAME         = "base_link"
TOOL_FRAME         = "tool0"
POSE_FRAME         = "base_link"         # IK/Cartesian 参考系
OBJECT_FRAME       = "object_position"   # 发布目标点 TF 供 RViz 查看
Z_VIRT             = 0.0                  # 工作平面 z（m）

# --- 相机内参（来自你的标定，分辨率 2500x1532）---
CALIB_WIDTH        = 2500
CALIB_HEIGHT       = 1532
K_RAW = np.array([
    [2674.3803723910564, 0.0,                 954.5922081613583],
    [0.0,                2667.4211254043507,  1074.965947832258],
    [0.0,                   0.0,                 1.0]
], dtype=np.float32)
D_RAW = np.array([-0.13009829401968415, 0.06793160264832976,
                  0.017127492925545672, -0.01132952745721562, 0.0], dtype=np.float32)
P_RECT = np.array([
    [2577.140869140625, 0.0,                919.7032521532165, 0.0],
    [0.0,               2594.39013671875,  1091.3219517491962, 0.0],
    [0.0,                  0.0,               1.0,             0.0]
], dtype=np.float32)
USE_IMAGE_RECT     = False  # False: /image_raw 用 K+D； True: /image_rect 用 P（无畸变）
FLIP_X             = False
FLIP_Y             = False

# --- 手眼外参（tool <- camera_(optical/link)）---
HAND_EYE_FRAME     = "optical"  # "optical" 或 "link"
T_TOOL_CAM_XYZ     = [ 0.05468636852374024, -0.029182661943126947, 0.05391824813032688 ]   # m
T_TOOL_CAM_QUAT    = [ -0.0036165657530785695, -0.000780788838366878,
                        0.7078681983794892,    0.7063348529868249 ]  # [qx,qy,qz,qw]
# 为了可视化（相对 tool0）
CAMERA_LINK_FRAME     = "camera_link"
CAMERA_OPTICAL_FRAME  = "camera_optical"

# --- DINO 检测 ---
TEXT_PROMPT        = "orange object ."
DINO_MODEL_ID      = "IDEA-Research/grounding-dino-tiny"
DINO_DEVICE        = "cuda"
BOX_THRESHOLD      = 0.25
TEXT_THRESHOLD     = 0.25
SCORE_OK           = 0.60
SCORE_EPS          = 1e-6
TF_LOOKUP_TIMEOUT  = 0.5
FRAME_STRIDE       = 2

# --- 初始位姿（先到位再检测）---
INIT_AT_START      = True
UR5E_JOINT_ORDER   = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
INIT_POS           = [
    0.9239029288291931,
   -1.186562405233719,
    1.1997712294207972,
   -1.5745235882201136,
   -1.5696094671832483,
   -0.579871956502096,
]
INIT_MOVE_TIME     = 3.0   # s
INIT_EXTRA_WAIT    = 0.3   # s

# --- 检测触发（仅静止时检测更稳）---
REQUIRE_STATIONARY   = True
VEL_EPS_RAD_PER_SEC  = 0.02

# --- 额外平面偏置（base 系，快速微调）---
BIAS_BASE_X        = 0.0
BIAS_BASE_Y        = 0.0
BIAS_BASE_Z        = 0.0

# --- 扫描平面高度 ---
HOVER_ABOVE        = 0.40   # 圆周平面高度 = C.z + HOVER_ABOVE

# --- 圆周参数 ---
RADIUS             = 0.25
START_ANGLE_DEG    = 0.0
CIRCLE_DIR         = "ccw"              # "ccw"/"cw"
NUM_TURNS          = 1
CLOSE_LOOP         = True
TOOL_Z_ALIGN_SIGN  = "-"                # tool-Z 与 world-Z 对齐方向：“+”或“-”；相机朝下通常为 "-"

# --- CartesianPath/IK 参数 ---
GROUP_NAME         = "ur_manipulator"
IK_LINK_NAME       = "tool0"
IK_SERVICE_CAND    = ["/compute_ik", "/move_group/compute_ik"]
EEF_STEP           = 0.005
EEF_STEP_SCALE     = 0.5
JUMP_THRESHOLD     = 0.0
AVOID_COLLISIONS   = True
MAX_RETRIES        = 3
MOVE_TO_START_VIA_IK = True
IK_TIMEOUT         = 2.0
IK_START_MOVE_TIME = 3.0

# --- 执行 ---
CONTROLLER_TOPIC   = "/scaled_joint_trajectory_controller/joint_trajectory"
FULL_TURN_TIME     = 20.0
MIN_POINTS_TIME    = 1.0

# --- joint_states 种子 ---
REQUIRE_JS         = True
JS_WAIT_TIMEOUT    = 2.0
FALLBACK_ZERO_SEED = False
ZERO_SEED          = [0, 0, 0, 0, 0, 0]
JS_RELIABILITY     = "reliable"  # reliable / best_effort

# --- TF 重广播 ---
TF_REBROADCAST_HZ  = 30.0
CAM_TF_BROADCAST_HZ= 30.0

# ==========================
# 工具函数
# ==========================

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

# REP-105: camera_link <- camera_optical 的固定旋转
R_CL_CO = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0]
], dtype=float)
R_CO_CL = R_CL_CO.T


def scale_K(K: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _wrap_pi(a):
    a = (a + math.pi) % (2*math.pi) - math.pi
    return a


def rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)


def yaw_to_quat_wxyz(yaw: float, sign: str = "+"):
    """仅绕世界Z旋转 yaw；若 sign=="-"，再乘以 qx(pi) 将 tool-Z 翻向 -Z_world。返回 [w,x,y,z]。"""
    c = math.cos(0.5*yaw)
    s = math.sin(0.5*yaw)
    q = np.array([c, 0.0, 0.0, s], dtype=float)  # qz(yaw) in [w,x,y,z]
    if str(sign).strip() == "-":
        # qx(pi) * qz(yaw)
        w1,x1,y1,z1 = math.cos(math.pi/2), math.sin(math.pi/2), 0.0, 0.0
        w2,x2,y2,z2 = q
        q = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def pose_from_pq(p: np.ndarray, q_wxyz: np.ndarray, frame: str) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p[:3])
    ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = map(float, q_wxyz)
    return ps

# ==========================
# 节点实现
# ==========================
class SeeAnythingCircleOnce(Node):
    def __init__(self):
        super().__init__("seeanything_circle_once")

        # ==== 配置 → 成员 ====
        self.image_topic = IMAGE_TOPIC
        self.base_frame = BASE_FRAME
        self.tool_frame = TOOL_FRAME
        self.pose_frame = POSE_FRAME
        self.object_frame = OBJECT_FRAME
        self.z_virt = float(Z_VIRT)

        self.use_image_rect = bool(USE_IMAGE_RECT)
        self.flip_x = bool(FLIP_X)
        self.flip_y = bool(FLIP_Y)

        self.hand_eye_frame = HAND_EYE_FRAME.strip().lower()
        self.t_tool_cam_xyz = np.array(T_TOOL_CAM_XYZ, dtype=float)
        self.t_tool_cam_quat = np.array(T_TOOL_CAM_QUAT, dtype=float)
        self.camera_link_frame = CAMERA_LINK_FRAME
        self.camera_optical_frame = CAMERA_OPTICAL_FRAME

        self.text_prompt = TEXT_PROMPT
        self.dino_model_id = DINO_MODEL_ID
        self.dino_device = DINO_DEVICE
        self.box_threshold = float(BOX_THRESHOLD)
        self.text_threshold = float(TEXT_THRESHOLD)
        self.score_ok = float(SCORE_OK)
        self.score_eps = float(SCORE_EPS)
        self.tf_lookup_timeout = float(TF_LOOKUP_TIMEOUT)
        self.frame_stride = int(FRAME_STRIDE)

        self.init_at_start = bool(INIT_AT_START)
        self.ur5e_joint_order = list(UR5E_JOINT_ORDER)
        self.init_pos = list(INIT_POS)
        self.init_move_time = float(INIT_MOVE_TIME)
        self.init_extra_wait = float(INIT_EXTRA_WAIT)

        self.require_stationary = bool(REQUIRE_STATIONARY)
        self.vel_eps = float(VEL_EPS_RAD_PER_SEC)

        self.bias_base_x = float(BIAS_BASE_X)
        self.bias_base_y = float(BIAS_BASE_Y)
        self.bias_base_z = float(BIAS_BASE_Z)

        self.hover_above = float(HOVER_ABOVE)

        self.radius = float(RADIUS)
        self.start_angle_deg = float(START_ANGLE_DEG)
        self.circle_dir = CIRCLE_DIR.strip().lower()
        self.num_turns = max(1, int(NUM_TURNS))
        self.close_loop = bool(CLOSE_LOOP)
        self.tool_z_align_sign = TOOL_Z_ALIGN_SIGN

        self.group_name = GROUP_NAME
        self.ik_link_name = IK_LINK_NAME
        self.ik_service_cand = list(IK_SERVICE_CAND)
        self.eef_step = float(EEF_STEP)
        self.eef_step_scale = float(EEF_STEP_SCALE)
        self.jump_threshold = float(JUMP_THRESHOLD)
        self.avoid_collisions = bool(AVOID_COLLISIONS)
        self.max_retries = int(MAX_RETRIES)
        self.move_to_start_via_ik = bool(MOVE_TO_START_VIA_IK)
        self.ik_timeout = float(IK_TIMEOUT)
        self.ik_start_move_time = float(IK_START_MOVE_TIME)

        self.controller_topic = CONTROLLER_TOPIC
        self.full_turn_time = float(FULL_TURN_TIME)
        self.min_points_time = float(MIN_POINTS_TIME)

        self.require_js = bool(REQUIRE_JS)
        self.js_wait_timeout = float(JS_WAIT_TIMEOUT)
        self.fallback_zero_if_timeout = bool(FALLBACK_ZERO_SEED)
        self.zero_seed = list(ZERO_SEED)
        self.js_reliability = JS_RELIABILITY.strip().lower()

        self.tf_rebroadcast_hz = float(TF_REBROADCAST_HZ)
        self.cam_tf_broadcast_hz = float(CAM_TF_BROADCAST_HZ)

        # ==== 通信 ====
        qos_img = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        self._img_count = 0
        self.create_subscription(Image, self.image_topic, self._cb_image, qos_img)

        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        if self.js_reliability == "best_effort":
            qos_js = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        else:
            qos_js = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        self._last_js: Optional[JointState] = None
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_js)

        self.pub_traj = self.create_publisher(JointTrajectory, self.controller_topic, 1)
        self.ik_client = None
        self.ik_service_name = None
        self.cart_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")

        # ==== DINO 预测器 ====
        try:
            from gripanything.core.detect_with_dino import GroundingDinoPredictor
        except Exception:
            import sys
            sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
            from gripanything.core.detect_with_dino import GroundingDinoPredictor
        self.predictor = GroundingDinoPredictor(self.dino_model_id, self.dino_device)

        # ==== 预计算手眼两种旋转（tool <- camera_{optical,link}）====
        R_t_cam_given = quat_to_rot(*self.t_tool_cam_quat.tolist())  # tool <- camera_(given)
        if self.hand_eye_frame == "optical":
            self.R_t_co = R_t_cam_given
            self.R_t_cl = self.R_t_co @ R_CO_CL
        else:
            self.R_t_cl = R_t_cam_given
            self.R_t_co = self.R_t_cl @ R_CL_CO
        self.p_t_c = self.t_tool_cam_xyz.copy()     # tool 下相机原点
        # 相机光轴（optical 的 +Z）在工具系下：
        self.cam_axis_in_tool = _unit(self.R_t_co @ np.array([0.0, 0.0, 1.0], dtype=float))

        # ==== 状态 ====
        self._phase = "init_needed" if self.init_at_start else "wait_detect"
        self._busy = False
        self._inflight_ik = False
        self._warned = set()
        self._motion_due_ns: Optional[int] = None
        self._done = False

        self._object_tf: Optional[TransformStamped] = None
        self._object_tf_published = False
        self._C_base: Optional[np.ndarray] = None
        self._ring_z: Optional[float] = None
        self._circle_waypoints: Optional[List[PoseStamped]] = None

        # 内参缓存（按当前图像尺寸缩放）
        self._cache_size = None
        self._K_scaled = None
        self._P_scaled = None

        # 定时器
        self.create_timer(0.05, self._tick)
        if self.tf_rebroadcast_hz > 0:
            self.create_timer(1.0 / self.tf_rebroadcast_hz, self._rebroadcast_object_tf)
        if self.cam_tf_broadcast_hz > 0:
            self.create_timer(1.0 / self.cam_tf_broadcast_hz, self._broadcast_camera_tfs)

        self.get_logger().info(
            f"[seeanything_circle_once] topic={self.image_topic}, rect={self.use_image_rect}, "
            f"hover={self.hover_above:.3f}m, bias=({self.bias_base_x:.3f},{self.bias_base_y:.3f},{self.bias_base_z:.3f}), "
            f"hand_eye={self.hand_eye_frame}, cam_frames=({self.camera_link_frame},{self.camera_optical_frame})"
        )

    # ---------- 工具：按当前图像尺寸准备内参 ----------
    def _prepare_intrinsics(self, w: int, h: int):
        if self._cache_size == (w, h):
            return
        if self.use_image_rect:
            self._P_scaled = scale_K(P_RECT.copy(), CALIB_WIDTH, CALIB_HEIGHT, w, h)
        else:
            self._K_scaled = scale_K(K_RAW, CALIB_WIDTH, CALIB_HEIGHT, w, h)
        self._cache_size = (w, h)

    # ---------- joint_states ----------
    def _on_js(self, msg: JointState):
        if self._last_js is None:
            self.get_logger().info(f"已收到 /joint_states（{len(msg.name)} 关节）。")
        self._last_js = msg

    def _is_stationary(self) -> bool:
        now_ns = self.get_clock().now().nanoseconds
        if self._motion_due_ns is not None and now_ns < self._motion_due_ns:
            return False
        if not self.require_stationary:
            return True
        if self._last_js is None or not self._last_js.velocity:
            return True
        try:
            vels = [abs(float(v)) for v in self._last_js.velocity]
            return all(v <= self.vel_eps for v in vels)
        except Exception:
            return True

    # ---------- 图像回调：检测并确定圆心 ----------
    def _cb_image(self, msg: Image):
        if self._done or self._busy or self._inflight_ik:
            return
        if self._phase != "wait_detect":
            return
        if not self._is_stationary():
            return

        self._img_count += 1
        if self.frame_stride > 1 and (self._img_count % self.frame_stride) != 0:
            return

        self._busy = True
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            h, w = rgb.shape[:2]
            self._prepare_intrinsics(w, h)

            pil = PILImage.fromarray(rgb)
            out = self.predictor.predict(pil, self.text_prompt,
                                         box_threshold=self.box_threshold, text_threshold=self.text_threshold)
            boxes, scores = None, None
            if isinstance(out, tuple):
                if len(out) == 3:
                    b, _labels, s = out
                    boxes = np.array(b) if not isinstance(b, np.ndarray) else b
                    scores = np.array(s) if s is not None else None
                elif len(out) == 2:
                    b, _labels = out
                    boxes = np.array(b) if not isinstance(b, np.ndarray) else b
            if boxes is None or len(boxes) == 0:
                return

            if scores is None:
                scores_np = -np.ones((len(boxes),), dtype=float)
            else:
                scores_np = np.array(scores).reshape(-1)
                if scores_np.shape[0] != len(boxes):
                    m = min(scores_np.shape[0], len(boxes))
                    tmp = -np.ones((len(boxes),), dtype=float)
                    tmp[:m] = scores_np[:m]
                    scores_np = tmp

            best = int(np.argmax(scores_np))
            score = float(scores_np[best])
            if score + self.score_eps < self.score_ok:
                self.get_logger().info(f"[detect] 分数不足：{score:.3f} < {self.score_ok:.2f}，继续等待。")
                return

            x0, y0, x1, y1 = [float(v) for v in boxes[best][:4]]
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)

            # 用图像时间戳查 base<-tool0
            t_query = Time.from_msg(msg.header.stamp)
            try:
                T_bt = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, t_query,
                                                       timeout=RclDuration(seconds=self.tf_lookup_timeout))
            except TransformException as ex:
                self.get_logger().info(f"[detect] 图像时刻 TF 不可得：{ex}，忽略此帧。")
                return
            R_bt, p_bt = tfmsg_to_Rp(T_bt)

            # base 下相机位姿（选 optical 轴用于投影）
            R_bc = R_bt @ self.R_t_co
            p_bc = R_bt @ self.p_t_c + p_bt

            # 像素 → 光学系单位视线
            if self.use_image_rect:
                fx = float(self._P_scaled[0, 0]); fy = float(self._P_scaled[1, 1])
                cx = float(self._P_scaled[0, 2]); cy = float(self._P_scaled[1, 2])
                xn = (u - cx) / fx
                yn = (v - cy) / fy
            else:
                K_scaled = self._K_scaled.astype(np.float32)
                pts = np.array([[[u, v]]], dtype=np.float32)
                und = cv2.undistortPoints(pts, K_scaled, D_RAW, P=None)
                xn, yn = float(und[0, 0, 0]), float(und[0, 0, 1])

            if self.flip_x: xn = -xn
            if self.flip_y: yn = -yn

            d_opt = np.array([xn, yn, 1.0], dtype=float)
            d_opt /= (np.linalg.norm(d_opt) + 1e-12)

            # 射线转到 base，并与 z=Z_VIRT 求交
            d_base = R_bc @ d_opt
            dz = float(d_base[2])
            if abs(dz) < 1e-6:
                self.get_logger().error("视线近水平，无法与平面求交。")
                return
            t_star = (self.z_virt - float(p_bc[2])) / dz
            if t_star < 0:
                self.get_logger().error("交点在相机后方，忽略。")
                return

            C = p_bc + t_star * d_base
            C[0] += self.bias_base_x
            C[1] += self.bias_base_y
            C[2] += self.bias_base_z

            # 保存圆心与圆周高度
            ring_z = float(C[2] + self.hover_above)
            self._C_base = C.copy()
            self._ring_z = ring_z

            # 可视化 TF：base <- object_position
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
            self._object_tf = tf_msg
            self.tf_broadcaster.sendTransform(tf_msg)
            self._object_tf_published = True

            self.get_logger().info(
                f"[detect] C=({C[0]:.3f},{C[1]:.3f},{C[2]:.3f}), ring_z={ring_z:.3f}, score={score:.3f}"
            )

            # 进入圆周阶段
            self._phase = "make_circle"

        except Exception as e:
            self.get_logger().error(f"图像处理失败：{e}")
        finally:
            self._busy = False

    # ---------- camera_link / camera_optical 的 TF 广播（相对 tool0） ----------
    def _broadcast_camera_tfs(self):
        # tool <- camera_link
        t1 = TransformStamped()
        t1.header.stamp = self.get_clock().now().to_msg()
        t1.header.frame_id = self.tool_frame
        t1.child_frame_id  = self.camera_link_frame
        t1.transform.translation.x = float(self.p_t_c[0])
        t1.transform.translation.y = float(self.p_t_c[1])
        t1.transform.translation.z = float(self.p_t_c[2])
        qx, qy, qz, qw = self._rot_to_quat(self.R_t_cl)
        t1.transform.rotation.x = qx
        t1.transform.rotation.y = qy
        t1.transform.rotation.z = qz
        t1.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t1)

        # tool <- camera_optical
        t2 = TransformStamped()
        t2.header.stamp = t1.header.stamp
        t2.header.frame_id = self.tool_frame
        t2.child_frame_id  = self.camera_optical_frame
        t2.transform.translation.x = float(self.p_t_c[0])
        t2.transform.translation.y = float(self.p_t_c[1])
        t2.transform.translation.z = float(self.p_t_c[2])
        qx, qy, qz, qw = self._rot_to_quat(self.R_t_co)
        t2.transform.rotation.x = qx
        t2.transform.rotation.y = qy
        t2.transform.rotation.z = qz
        t2.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t2)

    def _rot_to_quat(self, R: np.ndarray) -> Tuple[float, float, float, float]:
        t = float(np.trace(R))
        if t > 0:
            s = math.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            i = int(np.argmax(np.diag(R)))
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
        q = np.array([x, y, z, w], dtype=float)
        q /= (np.linalg.norm(q) + 1e-12)
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])

    # ---------- TF 重广播（object_position 可视化用） ----------
    def _rebroadcast_object_tf(self):
        if not self._object_tf_published or self._object_tf is None or self._done:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._object_tf.header.frame_id
        t.child_frame_id  = self._object_tf.child_frame_id
        t.transform = self._object_tf.transform
        self.tf_broadcaster.sendTransform(t)

    # ---------- yaw 求解：仅绕Z，使相机视线XY指向圆心 ----------
    def solve_yaw_to_center(self, p_tool_xy: np.ndarray, C_xy: np.ndarray) -> float:
        a = _unit(self.cam_axis_in_tool)
        a_xy = a[:2]
        if np.linalg.norm(a_xy) < 1e-6:
            # 相机几乎正对Z，仅绕Z影响极小 → 退化为朝向圆心的方位角
            v0 = C_xy - p_tool_xy
            return math.atan2(v0[1], v0[0])

        # 初始：忽略相机平移
        v = C_xy - p_tool_xy
        ang_v = math.atan2(v[1], v[0])
        ang_a = math.atan2(a_xy[1], a_xy[0])
        psi = _wrap_pi(ang_v - ang_a)

        # 1~2 次迭代考虑相机在工具系的平移
        t_xy = self.p_t_c[:2]
        for _ in range(2):
            Rz = rotz(psi)
            p_cam_xy = p_tool_xy + Rz.dot(t_xy)
            v = C_xy - p_cam_xy
            ang_v = math.atan2(v[1], v[0])
            psi = _wrap_pi(ang_v - ang_a)
        return psi

    # ---------- 生成圆周 waypoints ----------
    def make_circle_waypoints(self, C: np.ndarray, ring_z: float) -> List[PoseStamped]:
        ccw = (self.circle_dir == "ccw")
        total_deg = 360.0 * self.num_turns
        step_deg = 360.0 / 120.0  # 120 个点一圈

        waypoints: List[PoseStamped] = []
        def add_point(th_deg: float):
            th = math.radians(th_deg)
            p_tool = np.array([
                C[0] + self.radius * math.cos(th),
                C[1] + self.radius * math.sin(th),
                ring_z
            ], dtype=float)
            psi = self.solve_yaw_to_center(p_tool_xy=p_tool[:2], C_xy=C[:2])
            q = yaw_to_quat_wxyz(psi, sign=self.tool_z_align_sign)
            ps = pose_from_pq(p_tool, q, self.pose_frame)
            waypoints.append(ps)

        start = self.start_angle_deg
        if ccw:
            angles = np.arange(start, start + total_deg + 1e-6, step_deg)
        else:
            angles = np.arange(start, start - total_deg - 1e-6, -step_deg)
        for a in angles:
            add_point(a)
        if self.close_loop:
            add_point(start)
        return waypoints

    # ---------- IK 到圆周起点 ----------
    def _ensure_ik_client(self) -> bool:
        if self.ik_client:
            return True
        for name in self.ik_service_cand:
            cli = self.create_client(GetPositionIK, name)
            if cli.wait_for_service(timeout_sec=0.5):
                self.ik_client = cli
                self.ik_service_name = name
                self.get_logger().info(f"IK 服务可用：{name}")
                return True
        self.get_logger().warn(f"等待 IK 服务…（尝试：{self.ik_service_cand}）")
        return False

    def _get_seed(self) -> Optional[JointState]:
        if self._last_js:
            return self._last_js
        t0 = self.get_clock().now()
        while self.require_js and (self.get_clock().now() - t0) < RclDuration(seconds=self.js_wait_timeout):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._last_js:
                return self._last_js
        if not self.require_js and not self._last_js:
            return None
        if not self.fallback_zero_if_timeout:
            self.get_logger().warn("等待 /joint_states 超时，仍在等待或设置 FALLBACK_ZERO_SEED=True 以使用零位种子。")
            return None
        js = JointState()
        js.name = UR5E_JOINT_ORDER.copy()
        js.position = ZERO_SEED[:6]
        js.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().warn("使用零位种子。")
        return js

    def move_to_start_via_ik(self, start_pose: PoseStamped) -> bool:
        if not self._ensure_ik_client():
            return False
        seed = self._get_seed()
        if seed is None:
            return False
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ik_link_name
        req.ik_request.pose_stamped = start_pose
        req.ik_request.avoid_collisions = False
        req.ik_request.robot_state.joint_state = seed
        req.ik_request.timeout = MsgDuration(sec=int(self.ik_timeout), nanosec=int((self.ik_timeout % 1.0) * 1e9))
        self._inflight_ik = True
        fut = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self.ik_timeout + 1.0)
        self._inflight_ik = False
        res = fut.result()
        if res is None or res.error_code.val != 1:
            self.get_logger().error("起点 IK 失败。")
            return False
        jt = JointTrajectory()
        jt.joint_names = list(res.solution.joint_state.name)
        pt = JointTrajectoryPoint()
        pt.positions = list(res.solution.joint_state.position)
        pt.time_from_start = MsgDuration(sec=int(self.ik_start_move_time), nanosec=0)
        jt.points = [pt]
        self.pub_traj.publish(jt)
        self.get_logger().info(f"已发送起点关节目标（{self.ik_start_move_time:.1f}s）。")
        # 等待到位再继续
        self._motion_due_ns = self.get_clock().now().nanoseconds + int((self.ik_start_move_time + 0.3) * 1e9)
        return True

    # ---------- CartesianPath 生成 + 均匀时间化 ----------
    def compute_cartesian(self, waypoints: List[PoseStamped]) -> Tuple[float, JointTrajectory]:
        if not self.cart_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("等待 /compute_cartesian_path 服务超时。")
            return 0.0, JointTrajectory()
        poses = [Pose(position=ps.pose.position, orientation=ps.pose.orientation) for ps in waypoints]
        step = float(self.eef_step)
        fraction = 0.0
        traj_out = JointTrajectory()
        for _ in range(self.max_retries + 1):
            req = GetCartesianPath.Request()
            if self._last_js is not None:
                req.start_state.joint_state = self._last_js
            req.group_name = self.group_name
            req.link_name = self.ik_link_name
            req.waypoints = poses
            req.max_step = float(step)
            req.jump_threshold = float(self.jump_threshold)
            req.avoid_collisions = bool(self.avoid_collisions)
            self.get_logger().info(f"CartesianPath：eef_step={step:.4f}")
            fut = self.cart_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=15.0)
            res = fut.result()
            if res is None:
                self.get_logger().error("CartesianPath 服务无返回。")
                return 0.0, JointTrajectory()
            fraction = float(res.fraction)
            traj_out = res.solution.joint_trajectory
            if fraction >= 0.999:
                break
            step *= self.eef_step_scale
            self.get_logger().warn(f"fraction={fraction:.3f}，减小步长重试。")
        return fraction, traj_out

    def retime_uniform(self, jt: JointTrajectory, total_time: float) -> JointTrajectory:
        if len(jt.points) <= 1:
            return jt
        total_time = max(float(total_time), float(self.min_points_time))
        N = len(jt.points)
        for i, pt in enumerate(jt.points):
            t = (i / (N - 1)) * total_time
            pt.time_from_start = MsgDuration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            pt.velocities = []
            pt.accelerations = []
            pt.effort = []
        return jt

    def publish_trajectory(self, jt: JointTrajectory):
        if not jt.joint_names or not jt.points:
            self.get_logger().error("轨迹为空，未发布。")
            return
        self.pub_traj.publish(jt)
        self.get_logger().info(f"已发布轨迹（{len(jt.points)} 点）。")

    # ---------- 主循环 ----------
    def _publish_init_pose(self):
        jt = JointTrajectory()
        jt.joint_names = self.ur5e_joint_order
        pt = JointTrajectoryPoint()
        pt.positions = self.init_pos
        pt.time_from_start = MsgDuration(sec=int(self.init_move_time), nanosec=int((self.init_move_time % 1.0) * 1e9))
        jt.points = [pt]
        self.pub_traj.publish(jt)
        self.get_logger().info("已发布初始位姿…")
        self._motion_due_ns = self.get_clock().now().nanoseconds + int((self.init_move_time + self.init_extra_wait) * 1e9)

    def _tick(self):
        if self._done:
            return

        if self._phase == "init_needed":
            self._publish_init_pose()
            self._phase = "init_moving"
            return

        if self._phase == "init_moving":
            if self._is_stationary():
                self._phase = "wait_detect"
                self.get_logger().info("初始位姿到位，开始等待检测。")
            return

        if self._phase == "make_circle":
            if self._C_base is None or self._ring_z is None:
                return
            waypoints = self.make_circle_waypoints(self._C_base, self._ring_z)
            self._circle_waypoints = waypoints
            self.get_logger().info(f"生成圆周点：{len(waypoints)} 个。")
            # 先到起点（可选）
            if self.move_to_start_via_ik and len(waypoints) > 0:
                ok = self.move_to_start_via_ik(waypoints[0])
                if not ok:
                    self.get_logger().warn("未能先到起点，直接尝试 CartesianPath。")
            self._phase = "cartesian"
            return

        if self._phase == "cartesian":
            if not self._is_stationary():
                return
            if not self._circle_waypoints:
                return
            fraction, jt = self.compute_cartesian(self._circle_waypoints)
            self.get_logger().info(f"CartesianPath fraction={fraction:.3f}")
            if fraction < 0.8:
                self.get_logger().error("有效比例过低（<0.8），放弃发布。")
                self._done = True
                self.create_timer(0.5, self._shutdown_once)
                return
            total_time = self.full_turn_time * self.num_turns
            jt = self.retime_uniform(jt, total_time)
            self.publish_trajectory(jt)
            self._done = True
            self.create_timer(total_time + 1.0, self._shutdown_once)
            self._phase = "executing"
            return

    # ---------- 收尾 ----------
    def _shutdown_once(self):
        self.get_logger().info("seeanything_circle_once 完成，退出。")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = SeeAnythingCircleOnce()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
