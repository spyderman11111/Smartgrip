#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything_center_then_hover.py — 单帧触发 + 静止一帧取TF（严格按图像时间）+ 首次居中XY位移限幅

流程（每阶段仅使用“第一帧达标且机械臂静止”的检测）：
1) 回到初始位姿（可选）
2) 阶段1 center_needed：等待“第一帧且机械臂静止且score>=0.6” -> 仅一次 XY 平移（保持姿态+高度，且XY位移限幅）-> IK -> 运动
3) 静置一段时间（保证确实到位并稳定）
4) 阶段2 hover_needed：再次等待“第一帧且机械臂静止且score>=0.6” -> 发布 object_position -> 计算悬停 -> IK -> 运动
5) 结束

关键点：
- 严格使用【图像时间戳】查询 base<-tool0 TF（不再使用 latest）。
- 仅在判定“机械臂静止”时才处理图像；运动/未到位期间的图像忽略。
- 首次居中的 tool0 在 BASE 下 XY 位移做径向限幅（默认 ≤0.25 m）。
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


# ========= 置信度门限 =========
SCORE_OK   = 0.6
SCORE_FAIL = 0.4

# ========= 阶段2的 object_position 偏置（BASE 下）=========
ENABLE_BASE_BIAS = True
BIAS_BASE_X =  0.00
BIAS_BASE_Y =  0.00
BIAS_BASE_Z =  0.00

# ========= 阶段1平移微调（BASE 下 XY 微调，仅在居中阶段叠加）=========
ENABLE_CENTER_BIAS = True
BIAS_CENTER_X = -0.10
BIAS_CENTER_Y = -0.15

# ========= 居中位移限幅（仅第一次居中调整）=========
ENABLE_CENTER_XY_LIMIT = True   # 打开/关闭限幅
LIMIT_CENTER_XY = 0.25          # 最大 XY 平移半径（米）

# ========= 基本配置 =========
IMAGE_TOPIC = '/my_camera/pylon_ros2_camera_node/image_raw'
BASE_FRAME   = 'base_link'
TOOL_FRAME   = 'tool0'
OBJECT_FRAME = 'object_position'
POSE_FRAME   = 'base_link'
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

# 取TF策略（严格用图像时间戳）
TF_LOOKUP_TIMEOUT_SEC = 0.5  # 查询 TF 超时时间；查不到这帧就忽略

# 帧率/调试
FRAME_STRIDE         = 2
DEBUG_WINDOW         = False
DRAW_BEST_BOX        = False
DEBUG_HZ             = 5.0
TF_REBROADCAST_HZ    = 20.0
FLIP_X               = False
FLIP_Y               = False

# 悬停/IK 与控制
GROUP_NAME           = 'ur_manipulator'
IK_LINK_NAME         = 'tool0'
HOVER_ABOVE          = 0.40
YAW_DEG              = 0.0
IK_SERVICE_EXPLICIT  = ''         # 留空将自动尝试 ['/compute_ik','/move_group/compute_ik']
IK_TIMEOUT_SEC       = 2.0
CONTROLLER_TOPIC     = '/scaled_joint_trajectory_controller/joint_trajectory'
MOVE_TIME_SEC        = 3.0

# 静止判定（到位后仍需额外静置 + 速度阈值）
SETTLE_EXTRA_SEC         = 0.30   # 每次运动后额外静置，防止残余震荡
STILL_REQUIRE_JS_VEL     = True   # 若 JointState 含 velocity，则要求速度低于阈值
VEL_EPS_RAD_PER_SEC      = 0.02   # 关节速度阈值

# /joint_states 作为 IK 种子
REQUIRE_JS               = True
JS_WAIT_TIMEOUT_SEC      = 2.0
FALLBACK_ZERO_IF_TIMEOUT = False
ZERO_SEED                = [0, 0, 0, 0, 0, 0]

# ========= 启动初始位姿 =========
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

        # tool <- camera_(optical/link)
        R_t_cam = quat_to_rot(*T_TOOL_CAM_QUAT.tolist())
        if HAND_EYE_FRAME.lower() == 'optical':
            self.R_t_co = R_t_cam
        else:
            self.R_t_co = R_t_cam @ R_CL_CO
        self.p_t_co = T_TOOL_CAM_XYZ

        # 调试窗口
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

        # 运动到位/静止判定
        self._motion_due_ns: Optional[int] = None  # 最近一次“预计到位时刻”（发布轨迹时设定）
        self._inflight_ik = False

        # 状态机
        # phases: init_needed -> init_moving -> center_needed -> center_moving -> hover_needed -> hover_moving -> done
        self._phase = "init_needed" if INIT_AT_START else "center_needed"
        self._busy = False
        self._frame_count = 0
        self._warned = set()
        self._deadline_js: Optional[int] = None
        self._finished = False

        self.get_logger().info(
            f"[single-frame + still] topic={IMAGE_TOPIC}, prompt='{TEXT_PROMPT}', "
            f"score_ok={SCORE_OK:.2f}, score_fail={SCORE_FAIL:.2f}, hover={HOVER_ABOVE:.3f}m, limitXY={LIMIT_CENTER_XY:.3f}m"
        )
        self.create_timer(0.05, self._tick)

    # ---------- utils ----------
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

    def _is_stationary(self) -> bool:
        """满足：已过预计到位时间+静置，且（若有速度信息）所有速度|v|<阈值"""
        now_ns = self.get_clock().now().nanoseconds
        if self._motion_due_ns is not None and now_ns < self._motion_due_ns:
            return False
        if STILL_REQUIRE_JS_VEL and self._last_js is not None and self._last_js.velocity:
            try:
                vels = [abs(float(v)) for v in self._last_js.velocity]
                if any(v > VEL_EPS_RAD_PER_SEC for v in vels):
                    return False
            except Exception:
                pass
        return True

    # ---------- image callback ----------
    def _cb_image(self, msg: Image):
        # 只在需要检测的阶段处理：center_needed / hover_needed
        if self._phase not in ("center_needed", "hover_needed"):
            return
        # 仅在机械臂静止时才处理（保证“位姿运动完成后的一帧”）
        if not self._is_stationary():
            return

        # 限帧
        self._frame_count += 1
        if FRAME_STRIDE > 1 and (self._frame_count % FRAME_STRIDE) != 0:
            return
        if self._busy or self._inflight_ik:
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
                self.get_logger().error(f"[{self._phase}] 本帧无检测结果，退出。")
                self._shutdown_once()
                return

            s = np.array([float(s) if s is not None else -1.0 for s in scores])
            best = int(np.argmax(s))
            score = s[best]
            x0, y0, x1, y1 = (boxes[best].tolist() if hasattr(boxes[best], 'tolist') else boxes[best])
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)

            if score < SCORE_FAIL:
                self.get_logger().error(f"[{self._phase}] 检测置信度 {score:.3f} < {SCORE_FAIL:.2f}，退出。")
                self._shutdown_once()
                return
            if score < SCORE_OK:
                self.get_logger().info(f"[{self._phase}] 置信度 {score:.3f} 介于 {SCORE_FAIL:.2f}~{SCORE_OK:.2f}，继续等待。")
                return

            # —— 至此：score >= SCORE_OK，且机械臂静止 —— #
            # 严格使用图像时间戳查询 base<-tool0
            t_query = Time.from_msg(msg.header.stamp)
            try:
                T_bt = self.tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, t_query,
                                                       timeout=RclDuration(seconds=TF_LOOKUP_TIMEOUT_SEC))
            except TransformException as ex:
                self.get_logger().info(f"[{self._phase}] 图像时刻TF不可得（base<-tool0）：{ex}，忽略此帧，继续等待下一帧。")
                return

            R_bt, p_bt = tfmsg_to_Rp(T_bt)
            q_bt = T_bt.transform.rotation

            # 相机位姿（base）
            R_bc = R_bt @ self.R_t_co
            p_bc = R_bt @ self.p_t_co + p_bt

            # 像素 -> 光学系方向
            x_n = (u - CX) / FX
            y_n = (v - CY) / FY
            if FLIP_X: x_n = -x_n
            if FLIP_Y: y_n = -y_n
            d_opt = np.array([x_n, y_n, 1.0], dtype=float)
            d_opt /= np.linalg.norm(d_opt)

            # base 下与 z=Z_VIRT 平面求交
            d_base = R_bc @ d_opt
            dz = float(d_base[2])
            if abs(dz) < 1e-6:
                self.get_logger().error(f"[{self._phase}] 视线近水平，无法求交，退出。")
                self._shutdown_once()
                return
            t_star = (Z_VIRT - float(p_bc[2])) / dz
            if t_star < 0:
                self.get_logger().error(f"[{self._phase}] 交点在相机后方，退出。")
                self._shutdown_once()
                return
            C_raw = p_bc + t_star * d_base

            if self._phase == "center_needed":
                # 只用这帧一次性计算目标相机原点（只改 XY，保持高度与姿态）
                d_center = R_bc @ np.array([0.0, 0.0, 1.0], dtype=float)
                dzc = float(d_center[2])
                if abs(dzc) < 1e-6:
                    self.get_logger().error("[center] 中心射线近水平，退出。")
                    self._shutdown_once()
                    return
                oz = float(p_bc[2])  # 高度保持
                ox = float(C_raw[0]) - ((Z_VIRT - oz) / dzc) * float(d_center[0])
                oy = float(C_raw[1]) - ((Z_VIRT - oz) / dzc) * float(d_center[1])
                o_new = np.array([ox, oy, oz], dtype=float)

                # 叠加 BASE 下微调（仅首次居中）
                if ENABLE_CENTER_BIAS:
                    o_new[0] += float(BIAS_CENTER_X)
                    o_new[1] += float(BIAS_CENTER_Y)

                # 目标 tool0 原点（姿态保持为 q_bt）
                p_bt_new = o_new - (R_bt @ self.p_t_co)

                # ===== 新增：对“首次居中”的 XY 位移做限幅（以 tool0 在 BASE 下的位移为准）=====
                delta = p_bt_new - p_bt               # 当前 tool0 -> 目标 tool0
                dx, dy = float(delta[0]), float(delta[1])
                dist_xy = math.hypot(dx, dy)
                clipped = False
                if ENABLE_CENTER_XY_LIMIT and dist_xy > float(LIMIT_CENTER_XY):
                    scale = float(LIMIT_CENTER_XY) / dist_xy
                    p_bt_new = np.array([
                        float(p_bt[0]) + dx * scale,
                        float(p_bt[1]) + dy * scale,
                        float(p_bt_new[2]),            # Z 不变
                    ], dtype=float)
                    clipped = True
                # =======================================================================

                ps = PoseStamped()
                ps.header.frame_id = POSE_FRAME
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.pose.position.x = float(p_bt_new[0])
                ps.pose.position.y = float(p_bt_new[1])
                ps.pose.position.z = float(p_bt_new[2])
                ps.pose.orientation = q_bt  # 姿态保持

                seed = self._get_seed()
                if seed is None:
                    self._warn_once("wait_js_center", "等待 /joint_states 作为 IK 种子（居中阶段）…")
                    return

                move_xy_cmd = (float(p_bt_new[0] - p_bt[0]), float(p_bt_new[1] - p_bt[1]))
                self.get_logger().info(
                    f"[CENTER] 触发首帧达标: score={score:.3f}, uv=({u:.1f},{v:.1f}), "
                    f"moveXY_cmd=({move_xy_cmd[0]:.3f},{move_xy_cmd[1]:.3f})m"
                    + (" [CLIPPED]" if clipped else "")
                )

                self._request_ik(ps, seed, tag="center")
                self._phase = "center_moving"
                return

            if self._phase == "hover_needed":
                # 发布 object_position（只在这帧）
                C = C_raw.copy()
                if ENABLE_BASE_BIAS:
                    C[0] += float(BIAS_BASE_X); C[1] += float(BIAS_BASE_Y); C[2] += float(BIAS_BASE_Z)

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

                # 基于这帧一次性求悬停位姿并移动
                x, y, z = C[0], C[1], C[2] + HOVER_ABOVE
                q_down = (1.0, 0.0, 0.0, 0.0)
                q_yaw = quat_from_yaw(math.radians(YAW_DEG))
                q = quat_mul(q_yaw, q_down)

                ps = PoseStamped()
                ps.header.frame_id = POSE_FRAME
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.pose.position.x = float(x)
                ps.pose.position.y = float(y)
                ps.pose.position.z = float(z)
                ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q

                seed = self._get_seed()
                if seed is None:
                    self._warn_once("wait_js_hover", "等待 /joint_states 作为 IK 种子（hover 阶段）…")
                    return
                self.get_logger().info(
                    f"[HOVER] 触发首帧达标: score={score:.3f}, p=({x:.3f},{y:.3f},{z:.3f}), yaw={YAW_DEG:.1f}°"
                )
                self._request_ik(ps, seed, tag="hover")
                self._phase = "hover_moving"
                return

        except Exception as e:
            self.get_logger().error(f"图像处理失败：{e}")
        finally:
            self._busy = False

    # ---------- 主循环 ----------
    def _tick(self):
        if self._finished or self._inflight_ik:
            return

        now_ns = self.get_clock().now().nanoseconds

        # 初始位姿
        if self._phase == "init_needed":
            self._publish_init_pose()
            self._phase = "init_moving"
            # 预计到位 + 静置
            self._motion_due_ns = now_ns + int((INIT_MOVE_TIME_SEC + INIT_EXTRA_WAIT_SEC + SETTLE_EXTRA_SEC) * 1e9)
            return

        if self._phase == "init_moving":
            if self._is_stationary():
                self._phase = "center_needed"
                self.get_logger().info("初始位姿到位，等待“静止一帧达标”用于居中。")
                return

        if self._phase == "center_moving":
            if self._is_stationary():
                self._phase = "hover_needed"
                self.get_logger().info("居中完成，等待“静止一帧达标”用于悬停。")
                return

        if self._phase == "hover_moving":
            if self._is_stationary():
                self._finished = True
                self.get_logger().info("悬停运动完成，流程结束。")
                self.create_timer(0.5, self._shutdown_once)
                return

    # ---------- IK/种子 ----------
    def _ensure_ik_client(self) -> bool:
        if self._ik_client:
            return True
        candidates = [IK_SERVICE_EXPLICIT] if IK_SERVICE_EXPLICIT else ["/compute_ik", "/move_group/compute_ik"]
        for name in candidates:
            if not name:
                continue
            cli = self.create_client(GetPositionIK, name)
            if cli.wait_for_service(timeout_sec=0.5):
                self._ik_client = cli
                self._ik_service_name = name
                self.get_logger().info(f"IK 服务可用：{name}")
                return True
        self._warn_once("wait_ik", "等待 IK 服务…")
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
        fut = self._ik_client.call_async(req)
        fut.add_done_callback(lambda f: self._on_ik_done(f, tag))

    # ---------- IK 回调 ----------
    def _on_ik_done(self, fut, tag: str):
        self._inflight_ik = False
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"IK 调用异常[{tag}]：{e}")
            self._shutdown_once()
            return
        if res is None or res.error_code.val != 1:
            code = None if res is None else res.error_code.val
            self.get_logger().error(f"IK 未找到解[{tag}]，error_code={code}。退出。")
            self._shutdown_once()
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
            self.get_logger().error(f"IK 结果缺少关节[{tag}]: {missing}。退出。")
            self._shutdown_once()
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

        # 设定“预计到位时刻” + 额外静置
        now_ns = self.get_clock().now().nanoseconds
        self._motion_due_ns = now_ns + int((MOVE_TIME_SEC + SETTLE_EXTRA_SEC) * 1e9)

        self.get_logger().info(f"已发布关节目标[{tag}]，T={MOVE_TIME_SEC:.1f}s")

        if tag == "center":
            self._phase = "center_moving"
        elif tag == "hover":
            self._phase = "hover_moving"

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
        self.get_logger().info("已发布初始位姿，等待到位…")

    # ---------- 收尾 ----------
    def _shutdown_once(self):
        if rclpy.ok():
            self.get_logger().info("退出节点。")
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
