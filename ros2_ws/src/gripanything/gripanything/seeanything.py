#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything_center_then_hover.py
阶段1：DINO 检测 -> 仅平移使目标居中（保持当前姿态 + 高度，纯 XY，可叠加微调 BIAS）
阶段2：再次检测 -> 发布 object_position（含可选 XY/Z 硬补偿）-> 悬停到目标上方（末端朝下，可选 yaw）

新增：
- 启动即发布“初始位姿”（你提供的关节角），等待到位后再开始阶段1。
- 阶段1平移引入 BASE 下 XY 微调 BIAS（方便手动微调居中行为）。

前置：
- MoveIt 已启动且 /compute_ik 可用（仅阶段1/2用 IK；初始位姿直接发关节轨迹，不用 IK）
- 控制器 /scaled_joint_trajectory_controller/joint_trajectory 可用
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import math
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration as RclDuration
from rclpy.time import Time
from builtin_interfaces.msg import Duration

from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import TransformStamped, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_ros
from tf2_ros import TransformException
from cv_bridge import CvBridge
from PIL import Image as PILImage

from moveit_msgs.srv import GetPositionIK


# ========= 误差补偿（用于阶段2的 object_position）=========
ENABLE_BASE_BIAS = True
BIAS_BASE_X =  0.00
BIAS_BASE_Y =  0.00
BIAS_BASE_Z =  0.00

# ========= 阶段1平移的微调 BIAS（在 BASE 下的 XY，默认打开，便于微调）=========
ENABLE_CENTER_BIAS = True
BIAS_CENTER_X = -0.10   # +X 物体在相机画面更靠右，通常需要把相机往 +X 走
BIAS_CENTER_Y = -0.10   # +Y 物体在相机画面更靠上，通常需要把相机往 +Y 走

# ========= 基本配置 =========
IMAGE_TOPIC = '/my_camera/pylon_ros2_camera_node/image_raw'
BASE_FRAME   = 'base_link'
TOOL_FRAME   = 'tool0'
OBJECT_FRAME = 'object_position'
Z_VIRT       = 0.0   # 工作面高度

# 相机内参（像素系）
FX = 2674.3803723910564
FY = 2667.4211254043507
CX = 954.5922081613583
CY = 1074.965947832258

# 手眼外参：tool <- camera_(optical or link)
T_TOOL_CAM_XYZ  = np.array([-0.000006852374024, -0.099182661943126947, 0.02391824813032688], dtype=float)
T_TOOL_CAM_QUAT = np.array([-0.0036165657530785695, -0.000780788838366878,
                            0.7078681983794892, 0.7063348529868249], dtype=float)
HAND_EYE_FRAME  = 'optical'   # 'optical' 或 'link'

# DINO
TEXT_PROMPT    = 'yellow object .'
DINO_MODEL_ID  = 'IDEA-Research/grounding-dino-tiny'
DINO_DEVICE    = 'cuda'
BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.25

# 运行时开关
TF_TIME_MODE         = 'latest'   # 'image' | 'latest'
FRAME_STRIDE         = 2
DEBUG_WINDOW         = False
DRAW_BEST_BOX        = False
DEBUG_HZ             = 5.0
TF_REBROADCAST_HZ    = 20.0
FLIP_X               = False
FLIP_Y               = False

# 悬停/IK 与控制
POSE_FRAME           = 'base_link'
GROUP_NAME           = 'ur_manipulator'
IK_LINK_NAME         = 'tool0'
HOVER_ABOVE          = 0.30
YAW_DEG              = 0.0
IK_SERVICE_EXPLICIT  = ''
IK_TIMEOUT_SEC       = 2.0
CONTROLLER_TOPIC     = '/scaled_joint_trajectory_controller/joint_trajectory'
MOVE_TIME_SEC        = 3.0

# /joint_states 作为 IK 种子
REQUIRE_JS               = True
JS_WAIT_TIMEOUT_SEC      = 2.0
FALLBACK_ZERO_IF_TIMEOUT = False
ZERO_SEED                = [0, 0, 0, 0, 0, 0]
JS_RELIABILITY           = 'reliable'  # reliable / best_effort

# 居中阶段运动收敛等待（粗略按时间）
CENTER_SETTLE_EXTRA_SEC  = 0.3  # 轨迹时长外再等一点点
# 可选限制“单次平移步长”，None 表示不限制
MAX_CENTER_XY_STEP: Optional[float] = None  # 例如 0.25

# ========= 启动初始位姿（新增）=========
INIT_AT_START = True
UR5E_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
INIT_POS = [
    0.9239029288291931,
   -1.186562405233719,
    1.1997712294207972,
   -1.5745235882201136,
   -1.5696094671832483,
   -0.579871956502096,
]
INIT_MOVE_TIME_SEC = 3.0
INIT_EXTRA_WAIT_SEC = 0.3  # 给控制器一点缓冲


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


# camera_link <- camera_optical 固定旋转（REP-105）
R_CL_CO = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0]
], dtype=float)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    )


def quat_from_yaw(yaw_rad: float):
    s, c = math.sin(0.5 * yaw_rad), math.cos(0.5 * yaw_rad)
    return (0.0, 0.0, s, c)


class SeeAnythingCenterThenHover(Node):
    def __init__(self):
        super().__init__('seeanything_center_then_hover')

        qos_img = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, IMAGE_TOPIC, self._cb_image, qos_img)

        # /joint_states
        if JS_RELIABILITY.strip().lower() == "best_effort":
            qos_js = QoSProfile(depth=10,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST,
                                durability=DurabilityPolicy.VOLATILE)
        else:
            qos_js = QoSProfile(depth=10,
                                reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST,
                                durability=DurabilityPolicy.VOLATILE)
        self._last_js: Optional[JointState] = None
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_js)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # DINO
        try:
            from gripanything.core.detect_with_dino import GroundingDinoPredictor
        except Exception:
            import sys
            sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
            from gripanything.core.detect_with_dino import GroundingDinoPredictor
        self.predictor = GroundingDinoPredictor(DINO_MODEL_ID, DINO_DEVICE)

        # tool <- camera_optical
        R_t_cam = quat_to_rot(*T_TOOL_CAM_QUAT.tolist())
        if HAND_EYE_FRAME.lower() == 'optical':
            self.R_t_co = R_t_cam
        else:
            self.R_t_co = R_t_cam @ R_CL_CO
        self.p_t_co = T_TOOL_CAM_XYZ

        # 调试窗口
        self._last_debug_pub = self.get_clock().now()
        if DEBUG_WINDOW:
            try:
                cv2.namedWindow("DINO Debug", cv2.WINDOW_NORMAL)
            except Exception:
                self.get_logger().warn("无法创建可视化窗口。")

        # 最近一次有效 TF（阶段2重广播）
        self._last_good_tf: Optional[TransformStamped] = None
        if TF_REBROADCAST_HZ > 0:
            self.create_timer(1.0/TF_REBROADCAST_HZ, self._rebroadcast_tf)

        # 控制与 IK
        self.pub_traj = self.create_publisher(JointTrajectory, CONTROLLER_TOPIC, 1)
        self._ik_client = None
        self._ik_service_name = None
        self._ik_candidates = [IK_SERVICE_EXPLICIT] if IK_SERVICE_EXPLICIT else ["/compute_ik", "/move_group/compute_ik"]

        # 状态机 & 计时
        self._busy = False
        self._frame_count = 0
        self._warned = set()
        self._deadline_js: Optional[int] = None
        self._inflight_ik = False
        self._pending_motion_tag: Optional[str] = None  # "center" or "hover"
        self._center_due_ns: Optional[int] = None

        # 初始位姿相关
        self._init_due_ns: Optional[int] = None
        self._init_published = False

        # phases: "init_needed" -> "init_moving" -> "init_settling" -> "center_needed" -> "center_moving" -> ...
        self._phase = "init_needed" if INIT_AT_START else "center_needed"
        self._hover_requested = False
        self._done_all = False

        self.get_logger().info(
            f"[center->hover] topic={IMAGE_TOPIC}, prompt='{TEXT_PROMPT}', stride={FRAME_STRIDE}, "
            f"hand_eye={HAND_EYE_FRAME}, hover={HOVER_ABOVE:.3f}m, yaw={YAW_DEG:.1f}°"
        )

        self.create_timer(0.1, self._tick)

    # ---------- common utils ----------
    def _warn_once(self, key: str, text: str):
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warning(text)

    def _rebroadcast_tf(self):
        if self._last_good_tf is None:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._last_good_tf.header.frame_id
        t.child_frame_id  = self._last_good_tf.child_frame_id
        t.transform = self._last_good_tf.transform
        self.tf_broadcaster.sendTransform(t)

    def _on_js(self, msg: JointState):
        if self._last_js is None:
            self.get_logger().info(f"已收到 /joint_states（{len(msg.name)} 关节）。")
        self._last_js = msg

    # ---------- image callback ----------
    def _cb_image(self, msg: Image):
        # 若仍处于初始位姿阶段，直接忽略图像，避免并行动作
        if self._phase.startswith("init_"):
            return

        # 限速
        self._frame_count += 1
        if FRAME_STRIDE > 1 and (self._frame_count % FRAME_STRIDE) != 0:
            return
        if self._busy:
            return

        self._busy = True
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil = PILImage.fromarray(rgb)

            out = self.predictor.predict(pil, TEXT_PROMPT,
                                         box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)
            if isinstance(out, tuple) and len(out) == 3:
                boxes, labels, scores = out
            elif isinstance(out, tuple) and len(out) == 2:
                boxes, labels = out
                scores = [None] * len(boxes)
            else:
                self.get_logger().warn("DINO 返回格式不支持。")
                return
            if len(boxes) == 0:
                if self._phase == "center_needed":
                    self.get_logger().info("未检测到目标（等待首次居中用检测）。")
                else:
                    self.get_logger().info("未检测到目标。")
                return

            # 最高分框
            s = np.array([float(s) if s is not None else -1.0 for s in scores])
            best = int(np.argmax(s))
            x0, y0, x1, y1 = (boxes[best].tolist() if hasattr(boxes[best], 'tolist') else boxes[best])
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)
            sc = s[best]

            # base<-tool0（按图像时刻或最新）
            t_query = Time.from_msg(msg.header.stamp) if TF_TIME_MODE == 'image' else Time()
            try:
                T_bt = self.tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, t_query,
                                                       timeout=RclDuration(seconds=0.2))
            except TransformException as ex:
                self.get_logger().warn(f"TF 查找失败（{TF_TIME_MODE}，base<-tool0）：{ex}")
                return
            R_bt, p_bt = tfmsg_to_Rp(T_bt)
            q_bt = T_bt.transform.rotation

            # 当前相机位姿（base）
            R_bc = R_bt @ self.R_t_co
            p_bc = R_bt @ self.p_t_co + p_bt

            # ---------- 阶段1：基于 (u,v) 做“纯 XY 平移”以居中 ----------
            if self._phase == "center_needed":
                # 用 (u,v) 求虚拟平面交点 C_raw（不施加 XY 硬补偿）
                x_n = (u - CX) / FX
                y_n = (v - CY) / FY
                if FLIP_X: x_n = -x_n
                if FLIP_Y: y_n = -y_n
                d_opt = np.array([x_n, y_n, 1.0], dtype=float)
                d_opt /= np.linalg.norm(d_opt)

                d_base = R_bc @ d_opt
                if abs(float(d_base[2])) < 1e-6:
                    self.get_logger().warn("视线近水平，无法与平面求交（居中阶段）。")
                    return
                t_star = (Z_VIRT - float(p_bc[2])) / float(d_base[2])
                if t_star < 0:
                    self.get_logger().warn("交点在相机后方（居中阶段）。")
                    return
                C_raw = p_bc + t_star * d_base  # 目标在平面上的点（不补偿）

                # 计算需要的相机新原点 o_new（仅 XY 变更）：使中心射线 [0,0,1] 穿过 C_raw
                d_center = R_bc @ np.array([0.0, 0.0, 1.0], dtype=float)
                dz = float(d_center[2])
                if abs(dz) < 1e-6:
                    self.get_logger().warn("中心射线近水平，无法求 o_new。")
                    return
                oz = float(p_bc[2])  # 保持高度不变
                ox = float(C_raw[0]) - ((Z_VIRT - oz)/dz) * float(d_center[0])
                oy = float(C_raw[1]) - ((Z_VIRT - oz)/dz) * float(d_center[1])
                o_new = np.array([ox, oy, oz], dtype=float)

                # 阶段1平移的 XY 微调 BIAS（在 BASE 坐标系）
                if ENABLE_CENTER_BIAS:
                    o_new[0] += float(BIAS_CENTER_X)
                    o_new[1] += float(BIAS_CENTER_Y)

                # 可选步长限制
                delta = o_new - p_bc
                delta_xy = np.linalg.norm(delta[:2])
                if (MAX_CENTER_XY_STEP is not None) and (delta_xy > MAX_CENTER_XY_STEP):
                    scale = MAX_CENTER_XY_STEP / max(delta_xy, 1e-9)
                    o_new = p_bc + np.array([delta[0]*scale, delta[1]*scale, 0.0])

                # 求目标 tool 原点（保持 R_bt 不变）：p_bt_new = o_new - R_bt @ p_t_co
                p_bt_new = o_new - (R_bt @ self.p_t_co)

                # 生成姿态（保持当前末端姿态）
                ps = PoseStamped()
                ps.header.frame_id = POSE_FRAME
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.pose.position.x = float(p_bt_new[0])
                ps.pose.position.y = float(p_bt_new[1])
                ps.pose.position.z = float(p_bt_new[2])
                ps.pose.orientation = q_bt  # 保持当前姿态

                # 触发 IK（center）
                seed = self._get_seed()
                if seed is None:
                    self._warn_once("wait_js_center", "等待 /joint_states 作为 IK 种子（居中阶段）…")
                    return
                self.get_logger().info(
                    f"[CENTER] uv=({u:.1f},{v:.1f}), "
                    f"move XY = ({(o_new[0]-p_bc[0]):.3f},{(o_new[1]-p_bc[1]):.3f}) m, "
                    f"bias=({BIAS_CENTER_X:.3f},{BIAS_CENTER_Y:.3f})"
                )
                self._request_ik(ps, seed, tag="center")
                self._phase = "center_moving"
                return

            # ---------- 阶段2：常规检测，发布 object_position，并触发 hover ----------
            # 用 (u,v) 求平面交点 C_raw
            x_n = (u - CX) / FX
            y_n = (v - CY) / FY
            if FLIP_X: x_n = -x_n
            if FLIP_Y: y_n = -y_n
            d_opt = np.array([x_n, y_n, 1.0], dtype=float)
            d_opt /= np.linalg.norm(d_opt)

            d_base = R_bc @ d_opt
            if abs(float(d_base[2])) < 1e-6:
                self.get_logger().warn("视线近水平，无法与平面求交（阶段2）。")
                return
            t_star = (Z_VIRT - float(p_bc[2])) / float(d_base[2])
            if t_star < 0:
                self.get_logger().warn("交点在相机后方（阶段2）。")
                return
            C_raw = p_bc + t_star * d_base

            # XY/Z 硬补偿仅在阶段2用于 object_position
            C = C_raw.copy()
            if ENABLE_BASE_BIAS:
                C[0] += float(BIAS_BASE_X)
                C[1] += float(BIAS_BASE_Y)
                C[2] += float(BIAS_BASE_Z)

            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = BASE_FRAME
            tf_msg.child_frame_id  = OBJECT_FRAME
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
                f"[STAGE2] score={(sc if sc>=0 else float('nan')):.3f} "
                f"uv=({u:.1f},{v:.1f}) "
                f"C_raw=({C_raw[0]:.3f},{C_raw[1]:.3f},{C_raw[2]:.3f}) "
                f"C_corr=({C[0]:.3f},{C[1]:.3f},{C[2]:.3f}) -> {OBJECT_FRAME}"
            )

            # 居中已完成后，拿到有效 TF 即可触发 hover
            if self._phase == "center_done":
                self._hover_requested = True

        except Exception as e:
            self.get_logger().error(f"图像处理失败：{e}")
        finally:
            self._busy = False

    # ---------- 主循环 ----------
    def _tick(self):
        if self._done_all or self._inflight_ik:
            return

        now_ns = self.get_clock().now().nanoseconds

        # ====== 初始位姿阶段 ======
        if self._phase == "init_needed":
            # 发布一次初始位姿（不依赖 IK）
            if not self._init_published:
                self._publish_init_pose()
                self._init_due_ns = now_ns + int((INIT_MOVE_TIME_SEC + INIT_EXTRA_WAIT_SEC) * 1e9)
                self._phase = "init_moving"
                return

        if self._phase == "init_moving" and self._init_due_ns is not None:
            if now_ns >= self._init_due_ns:
                self._phase = "center_needed"
                self.get_logger().info("初始位姿到位，开始阶段1（检测并平移居中）。")

        # ====== 居中阶段收敛等待 ======
        if self._phase == "center_settling" and self._center_due_ns is not None:
            if now_ns >= self._center_due_ns:
                self._phase = "center_done"
                self.get_logger().info("CENTER 运动完成，进入阶段2（再次检测并发布 object_position）。")

        # ====== 悬停触发 ======
        if self._phase == "center_done" and self._hover_requested and self._last_good_tf is not None:
            if not self._ensure_ik_client():
                return
            seed = self._get_seed()
            if seed is None:
                self._warn_once("wait_js_hover", "等待 /joint_states 作为 IK 种子（hover 阶段）…")
                return
            ps = self._make_hover_pose()
            if ps is None:
                return
            self.get_logger().info(
                f"[HOVER] p=({ps.pose.position.x:.3f},{ps.pose.position.y:.3f},{ps.pose.position.z:.3f}), "
                f"yaw={YAW_DEG:.1f}°, hover={HOVER_ABOVE:.3f}m"
            )
            self._request_ik(ps, seed, tag="hover")
            self._hover_requested = False  # 防多次触发

    # ---------- IK/种子 ----------
    def _ensure_ik_client(self) -> bool:
        if self._ik_client:
            return True
        for name in self._ik_candidates:
            if not name:
                continue
            cli = self.create_client(GetPositionIK, name)
            if cli.wait_for_service(timeout_sec=0.5):
                self._ik_client = cli
                self._ik_service_name = name
                self.get_logger().info(f"IK 服务可用：{name}")
                return True
        self._warn_once("wait_ik", f"等待 IK 服务…（尝试：{self._ik_candidates}）")
        return False

    def _get_seed(self) -> Optional[JointState]:
        if self._last_js:
            return self._last_js
        now_ns = self.get_clock().now().nanoseconds
        if self._deadline_js is None:
            self._deadline_js = now_ns + int(JS_WAIT_TIMEOUT_SEC * 1e9)
            if REQUIRE_JS:
                self._warn_once("wait_js", "等待 /joint_states …")
            return None
        if now_ns < self._deadline_js and REQUIRE_JS:
            return None
        if not FALLBACK_ZERO_IF_TIMEOUT and REQUIRE_JS:
            self._warn_once("js_timeout", "等待 /joint_states 超时，仍在等待（可设 FALLBACK_ZERO_IF_TIMEOUT=True 使用零位种子）。")
            return None
        js = JointState()
        js.name = UR5E_ORDER
        js.position = ZERO_SEED[:6]
        js.header.stamp = self.get_clock().now().to_msg()
        return js

    # ---------- 生成悬停位姿 ----------
    def _make_hover_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self.tf_buffer.lookup_transform(POSE_FRAME, OBJECT_FRAME, Time(),
                                                 timeout=RclDuration(seconds=0.5))
        except TransformException as ex:
            self._warn_once("tf_object", f"TF 未就绪：{POSE_FRAME} <- {OBJECT_FRAME} ：{ex}")
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        z = tf.transform.translation.z + HOVER_ABOVE

        q_down = (1.0, 0.0, 0.0, 0.0)
        q_yaw  = quat_from_yaw(math.radians(YAW_DEG))
        q      = quat_mul(q_yaw, q_down)

        ps = PoseStamped()
        ps.header.frame_id = POSE_FRAME
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        return ps

    # ---------- 发起 IK ----------
    def _request_ik(self, pose: PoseStamped, seed: JointState, tag: str):
        if not self._ensure_ik_client():
            return
        req = GetPositionIK.Request()
        req.ik_request.group_name = GROUP_NAME
        req.ik_request.ik_link_name = IK_LINK_NAME
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = False
        req.ik_request.robot_state.joint_state = seed
        req.ik_request.timeout = Duration(
            sec=int(IK_TIMEOUT_SEC),
            nanosec=int((IK_TIMEOUT_SEC % 1.0) * 1e9),
        )
        self._inflight_ik = True
        self._pending_motion_tag = tag
        fut = self._ik_client.call_async(req)
        fut.add_done_callback(self._on_ik_done)

    # ---------- IK 回调 ----------
    def _on_ik_done(self, fut):
        self._inflight_ik = False
        tag = self._pending_motion_tag
        self._pending_motion_tag = None

        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"IK 调用异常[{tag}]：{e}")
            return
        if res is None or res.error_code.val != 1:
            code = None if res is None else res.error_code.val
            self.get_logger().error(f"IK 未找到解[{tag}]，error_code={code}")
            return

        name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(res.solution.joint_state.name)}
        target_positions: List[float] = []
        missing = []
        for jn in UR5E_ORDER:
            if jn not in name_to_idx:
                missing.append(jn)
            else:
                target_positions.append(res.solution.joint_state.position[name_to_idx[jn]])
        if missing:
            self.get_logger().error(f"IK 结果缺少关节[{tag}]: {missing}")
            return

        traj = JointTrajectory()
        traj.joint_names = UR5E_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = target_positions
        pt.time_from_start = Duration(
            sec=int(MOVE_TIME_SEC),
            nanosec=int((MOVE_TIME_SEC % 1.0) * 1e9)
        )
        traj.points = [pt]
        self.pub_traj.publish(traj)

        self.get_logger().info(
            f"已发布关节目标[{tag}]："
            + "[" + ", ".join(f"{v:.6f}" for v in target_positions) + f"], T={MOVE_TIME_SEC:.1f}s"
        )

        if tag == "center":
            self._phase = "center_settling"
            self._center_due_ns = self.get_clock().now().nanoseconds + int((MOVE_TIME_SEC + CENTER_SETTLE_EXTRA_SEC)*1e9)
        elif tag == "hover":
            self._done_all = True
            self.create_timer(0.5, self._shutdown_once)

    # ---------- 启动时发布初始位姿（不走 IK） ----------
    def _publish_init_pose(self):
        traj = JointTrajectory()
        traj.joint_names = UR5E_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = INIT_POS
        pt.time_from_start = Duration(
            sec=int(INIT_MOVE_TIME_SEC),
            nanosec=int((INIT_MOVE_TIME_SEC % 1.0) * 1e9)
        )
        traj.points = [pt]
        self.pub_traj.publish(traj)
        self._init_published = True
        self.get_logger().info(
            "已发布初始位姿：[" + ", ".join(f"{v:.6f}" for v in INIT_POS) +
            f"], T={INIT_MOVE_TIME_SEC:.1f}s"
        )

    # ---------- 收尾 ----------
    def _shutdown_once(self):
        self.get_logger().info("全部完成，退出节点。")
        rclpy.shutdown()

    def destroy_node(self):
        try:
            if DEBUG_WINDOW:
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = SeeAnythingCenterThenHover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
