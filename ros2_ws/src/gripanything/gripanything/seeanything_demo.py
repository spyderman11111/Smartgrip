#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything_minimal_clean.py — 一次检测 + 悬停 + N点圆周逐点IK + 返回初始姿态（不闭合）

更新要点：
- 圆周**起点 = 开始圆周前“当前末端执行器（tool X 轴）在XY平面上的朝向”**；
- 顺序与方向严格一致（POLY_DIR），不闭合；
- EE 朝向默认“径向朝内”(ORIENT_MODE='radial_in')，始终指向圆心；可选 radial_out/tangent；
- 关节跳变检测忽略末端轴（默认忽略 wrist_3_joint），仅限制其它关节；保留 2π 邻近展开。
"""

from typing import Optional, Tuple, List
import numpy as np
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration as RclDuration
from rclpy.time import Time
from builtin_interfaces.msg import Duration as MsgDuration

from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import TransformStamped, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_ros
from tf2_ros import TransformException
from cv_bridge import CvBridge
from PIL import Image as PILImage

from moveit_msgs.srv import GetPositionIK

# ====== 误差补偿（保留可改） ======
ENABLE_BASE_BIAS = True
BIAS_BASE_X = -0.05
BIAS_BASE_Y = -0.25
BIAS_BASE_Z = 0.00

# ================== 基本配置 ==================
IMAGE_TOPIC = '/my_camera/pylon_ros2_camera_node/image_raw'
BASE_FRAME   = 'base_link'
TOOL_FRAME   = 'tool0'
POSE_FRAME   = 'base_link'
OBJECT_FRAME = 'object_position'
OBJECT_CIRCLE_FRAME = 'object_circle'
Z_VIRT       = 0.0

# 相机内参（像素系）
FX = 2674.3803723910564
FY = 2667.4211254043507
CX = 954.5922081613583
CY = 1074.965947832258

# 手眼外参：tool <- camera_(optical/link)
T_TOOL_CAM_XYZ  = np.array([-0.000006852374024, -0.059182661943126947, -0.00391824813032688], dtype=float)
T_TOOL_CAM_QUAT = np.array([-0.0036165657530785695, -0.000780788838366878,
                            0.7078681983794892, 0.7063348529868249], dtype=float)  # qx,qy,qz,qw
HAND_EYE_FRAME  = 'optical'   # 'optical' 或 'link'

# DINO
TEXT_PROMPT    = 'orange object .'
DINO_MODEL_ID  = 'IDEA-Research/grounding-dino-tiny'
DINO_DEVICE    = 'cuda'
BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.25

# 运行时开关
TF_TIME_MODE         = 'latest'   # 'image' | 'latest'
FRAME_STRIDE         = 2
TF_REBROADCAST_HZ    = 20.0
FLIP_X               = False
FLIP_Y               = False

# ===== IK / 控制 =====
HOVER_ABOVE      = 0.30                 # 悬停高度（m）
GROUP_NAME       = 'ur_manipulator'
IK_LINK_NAME     = 'tool0'
IK_TIMEOUT       = 2.0                  # s
CONTROLLER_TOPIC = '/scaled_joint_trajectory_controller/joint_trajectory'
MOVE_TIME        = 3.0                  # 悬停/多边形点默认移动时间

# 初始位姿（开始与结束都用 5s）
UR5E_JOINT_ORDER = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]
INIT_POS = [
    0.7734344005584717,
   -1.0457398456386109,
    1.0822847525226038,
   -1.581707616845602,
   -1.5601266066180628,
   -0.8573678175555628,
]
INIT_MOVE_TIME  = 5.0  # s（开始与结束统一 5s）
INIT_EXTRA_WAIT = 0.3  # s

# 检测触发更稳：仅静止时
REQUIRE_STATIONARY  = True
VEL_EPS_RAD_PER_SEC = 0.02

# ====== N 点圆周 + 逐点IK ======
POLY_N_VERTICES      = 4        # 4–16 之间为宜
# 起点方向相对“当前EE朝向”的额外旋转（在 base_link XY 平面；正=CCW）
START_DIR_OFFSET_DEG = -90.0
POLY_NUM_TURNS       = 1        # 旋转圈数
POLY_DIR             = 'ccw'    # 'ccw' 或 'cw'
CIRCLE_RADIUS        = 0.15     # 圆半径（m）

# EE 朝向模式：径向朝内/朝外/切向（默认径向朝内，指向圆心）
ORIENT_MODE          = 'radial_in'       # 'radial_in' | 'radial_out' | 'tangent'
TOOL_Z_SIGN          = '-'               # 工具 Z 与 world Z 对齐方向（'-' = 朝下）
DWELL_TIME           = 0.25              # 每个顶点停顿（拍照）秒
EDGE_MOVE_TIME       = 3.0               # 单段移动时间（s）

# 跳变监测阈值（对“非忽略关节”生效）
MAX_SAFE_JUMP_RAD    = 1.2
MAX_WARN_JUMP_RAD    = 2.2
SAFE_JUMP_IGNORE_JOINTS = ['wrist_3_joint']  # 默认忽略末端一轴的跳变

# ================== DINO 封装 ==================
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except Exception:
    import sys
    sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
    from gripanything.core.detect_with_dino import GroundingDinoPredictor

def _safe_float(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
    except Exception:
        pass
    return float(x)

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

# camera_link <- camera_optical（REP-105）
R_CL_CO = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0]
], dtype=float)

TWO_PI = 2.0 * math.pi
def _wrap_to_near(angle: float, ref: float) -> float:
    return angle + round((ref - angle) / TWO_PI) * TWO_PI

def _quat_mul(q1_wxyz, q2_wxyz):
    w1,x1,y1,z1 = q1_wxyz
    w2,x2,y2,z2 = q2_wxyz
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def yaw_to_quat_wxyz(yaw: float, sign: str = "-") -> np.ndarray:
    """q = Rz(yaw) 或 Rz(yaw)*Rx(pi)（工具Z朝下），返回 [w,x,y,z]。"""
    c = math.cos(0.5*yaw)
    s = math.sin(0.5*yaw)
    qz = np.array([c, 0.0, 0.0, s], dtype=float)
    if str(sign).strip() == "-":
        qx_pi = np.array([math.cos(math.pi/2), math.sin(math.pi/2), 0.0, 0.0], dtype=float)
        q = _quat_mul(qz, qx_pi)   # Rz * Rx
    else:
        q = qz
    q /= (np.linalg.norm(q) + 1e-12)
    return q

def pose_from_pq(p_xyz, q_wxyz, frame):
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p_xyz[:3])
    w,x,y,z = map(float, q_wxyz)
    ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = w,x,y,z
    return ps


class SeeAnythingMinimal(Node):
    def __init__(self):
        super().__init__('seeanything_minimal_clean')

        # 订阅图像
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, IMAGE_TOPIC, self._cb_image, qos)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # DINO
        self.predictor = GroundingDinoPredictor(DINO_MODEL_ID, DINO_DEVICE)

        # 预计算 tool <- camera_optical
        R_t_cam = quat_to_rot(*T_TOOL_CAM_QUAT.tolist())  # tool <- camera_(?)
        self.R_t_co = R_t_cam if HAND_EYE_FRAME.lower() == 'optical' else (R_t_cam @ R_CL_CO)
        self.p_t_co = T_TOOL_CAM_XYZ  # 光学与 link 原点一致

        # 最近一次有效 TF（用于重广播）
        self._last_good_tf: Optional[TransformStamped] = None
        self._circle_tf: Optional[TransformStamped] = None
        if TF_REBROADCAST_HZ > 0:
            self.create_timer(1.0/TF_REBROADCAST_HZ, self._rebroadcast_tfs)

        # 轨迹/IK
        self.pub_traj = self.create_publisher(JointTrajectory, CONTROLLER_TOPIC, 1)
        self._ik_client = None
        self._ik_candidates = ['/compute_ik', '/move_group/compute_ik']

        qos_js = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                            history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        self._last_js: Optional[JointState] = None
        self.create_subscription(JointState, '/joint_states', self._on_js, qos_js)

        # —— 软种子（上一解），用于 2π 邻近展开 ——
        self._seed_hint: Optional[np.ndarray] = None

        # 圆心/圆周数据
        self._circle_center: Optional[np.ndarray] = None  # 仅初始帧，已加 bias
        self._ring_z: Optional[float] = None
        self._poly_wps: List[PoseStamped] = []
        self._poly_idx: int = 0
        self._poly_dwell_due_ns: Optional[int] = None
        self._start_yaw: Optional[float] = None  #  圆周起点的 yaw = 开始时 EE 朝向

        # 状态机
        # init_needed -> init_moving -> wait_detect -> hover_to_center -> poly_prepare -> poly_moving -> return_init -> done
        self._phase = 'init_needed'
        self._busy = False
        self._inflight = False
        self._motion_due_ns: Optional[int] = None
        self._done = False
        self._warned_once = set()
        self._fixed_hover_pose: Optional[PoseStamped] = None
        self._frame_count = 0

        self.create_timer(0.05, self._tick)

        self.get_logger().info(
            f"[seeanything_minimal_clean] topic={IMAGE_TOPIC}, hover={HOVER_ABOVE:.3f}m, "
            f"bias=({BIAS_BASE_X:.3f},{BIAS_BASE_Y:.3f},{BIAS_BASE_Z:.3f}); N={POLY_N_VERTICES}, R={CIRCLE_RADIUS:.3f}m, "
            f"orient={ORIENT_MODE}, dir={POLY_DIR}"
        )

    # ---------- TF 重广播 ----------
    def _rebroadcast_tfs(self):
        now = self.get_clock().now().to_msg()
        if self._last_good_tf is not None:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self._last_good_tf.header.frame_id
            t.child_frame_id  = self._last_good_tf.child_frame_id
            t.transform = self._last_good_tf.transform
            self.tf_broadcaster.sendTransform(t)
        if self._circle_tf is not None:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self._circle_tf.header.frame_id
            t.child_frame_id  = self._circle_tf.child_frame_id
            t.transform = self._circle_tf.transform
            self.tf_broadcaster.sendTransform(t)

    # ---------- joint_states ----------
    def _on_js(self, msg: JointState):
        if self._last_js is None:
            self.get_logger().info(f"已收到 /joint_states（{len(msg.name)} 关节）。")
        self._last_js = msg

    def _is_stationary(self) -> bool:
        now_ns = self.get_clock().now().nanoseconds
        if self._motion_due_ns is not None and now_ns < self._motion_due_ns:
            return False
        if not REQUIRE_STATIONARY:
            return True
        if self._last_js is None or not self._last_js.velocity:
            return True
        try:
            return all(abs(float(v)) <= VEL_EPS_RAD_PER_SEC for v in self._last_js.velocity)
        except Exception:
            return True

    # ---------- 读取当前 tool0 的 XY 平面 yaw（以工具 X 轴为“前向”） ----------
    def _get_tool_yaw_xy(self) -> Optional[float]:
        try:
            T_bt = self.tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, Time(),
                                                   timeout=RclDuration(seconds=0.2))
            R_bt, _ = tfmsg_to_Rp(T_bt)
            ex = R_bt[:, 0]  # tool x 轴在 base 下的方向
            yaw = math.atan2(float(ex[1]), float(ex[0]))  # [-pi, pi]
            # 不 wrap，交给下游保持连续
            return float(yaw)
        except Exception as ex:
            self.get_logger().warn(f"读取当前 tool yaw 失败：{ex}")
            return None

    # ---------- 主循环 ----------
    def _tick(self):
        if self._done or self._inflight:
            return

        if self._phase == 'init_needed':
            self._publish_init_pose(INIT_MOVE_TIME)  # 5s
            self._phase = 'init_moving'
            return

        if self._phase == 'init_moving':
            if self._is_stationary():
                self._phase = 'wait_detect'
                self.get_logger().info("初始位姿到位，开始等待检测。")
            return

        if self._phase == 'wait_detect' and self._fixed_hover_pose is not None:
            if not self._ensure_ik_client():
                return
            seed = self._get_seed()
            if seed is None:
                return
            # 先到圆心上方的悬停（朝下）
            self._request_ik(self._fixed_hover_pose, seed, MOVE_TIME)
            self._phase = 'hover_to_center'
            return

        if self._phase == 'hover_to_center':
            if not self._is_stationary():
                return
            # 记录开始圆周前的“当前 EE 朝向”
            self._start_yaw = self._get_tool_yaw_xy()
            # 准备圆周顶点
            if self._circle_center is not None and self._ring_z is not None and self._start_yaw is not None:
                self._poly_wps = self._make_polygon_vertices(self._circle_center, self._ring_z, self._start_yaw)
                self._poly_idx = 0
                self.get_logger().info(f"顶点序列生成：{len(self._poly_wps)} 个，start_yaw={self._start_yaw:.3f} rad。")
                self._phase = 'poly_prepare'
            else:
                self._done = True
                self.create_timer(0.5, self._shutdown_once)
            return

        if self._phase == 'poly_prepare':
            if not self._poly_wps:
                self._done = True
                self.create_timer(0.5, self._shutdown_once)
                return
            # 先到第一个顶点
            self._ik_to_vertex(self._poly_wps[0], EDGE_MOVE_TIME)
            self._poly_idx = 1
            self._phase = 'poly_moving'
            return

        if self._phase == 'poly_moving':
            now = self.get_clock().now().nanoseconds
            if not self._is_stationary():
                return
            # 到位 → 停顿拍照
            if self._poly_dwell_due_ns is None:
                self._poly_dwell_due_ns = now + int(DWELL_TIME * 1e9)
                self.get_logger().info("到位，停顿拍照…")
                return
            if now < self._poly_dwell_due_ns:
                return
            self._poly_dwell_due_ns = None

            # 下一步或结束
            if self._poly_idx >= len(self._poly_wps):
                self._publish_init_pose(INIT_MOVE_TIME)  # 回初始位姿
                self._phase = 'return_init'
                return

            self._ik_to_vertex(self._poly_wps[self._poly_idx], EDGE_MOVE_TIME)
            self._poly_idx += 1
            return

        if self._phase == 'return_init':
            if not self._is_stationary():
                return
            self._done = True
            self.create_timer(0.5, self._shutdown_once)
            return

    # ---------- 发送“回初始姿态”的关节轨迹 ----------
    def _publish_init_pose(self, move_time: float):
        traj = JointTrajectory()
        traj.joint_names = UR5E_JOINT_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = INIT_POS
        pt.time_from_start = MsgDuration(sec=int(move_time), nanosec=int((move_time % 1.0) * 1e9))
        traj.points = [pt]
        self.pub_traj.publish(traj)
        self.get_logger().info(f"已发布关节初始位姿（T={move_time:.1f}s）…")
        now_ns = self.get_clock().now().nanoseconds
        self._motion_due_ns = now_ns + int((move_time + INIT_EXTRA_WAIT) * 1e9)
        self._seed_hint = np.array(INIT_POS, dtype=float)  # 第一解作为软种子

    # ---------- 图像回调：只在“初始位姿静止”后触发一次 ----------
    def _cb_image(self, msg: Image):
        if self._done or self._inflight or self._busy:
            return
        if self._phase != 'wait_detect':
            return
        if not self._is_stationary():
            return
        if FRAME_STRIDE > 1:
            self._frame_count += 1
            if (self._frame_count % FRAME_STRIDE) != 0:
                return

        self._busy = True
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil = PILImage.fromarray(rgb)

            out = self.predictor.predict(
                pil, TEXT_PROMPT, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD
            )
            if isinstance(out, tuple) and len(out) == 3:
                boxes, labels, scores = out
            elif isinstance(out, tuple) and len(out) == 2:
                boxes, labels = out
                scores = [None] * len(boxes)
            else:
                self.get_logger().warn("DINO 返回格式不支持。")
                return
            if len(boxes) == 0:
                self.get_logger().info("未检测到目标。")
                return

            # 最高分
            s = np.array([_safe_float(s) if s is not None else -1.0 for s in scores], dtype=float)
            best = int(np.argmax(s))
            bx = boxes[best]
            x0, y0, x1, y1 = (bx.tolist() if hasattr(bx, 'tolist') else bx)
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)

            # 像素 -> 光学系视线（与你原先一致）
            x_n0 = (u - CX) / FX
            y_n0 = (v - CY) / FY
            x_n =  y_n0
            y_n = -x_n0
            if FLIP_X: x_n = -x_n
            if FLIP_Y: y_n = -y_n
            d_opt = np.array([x_n, y_n, 1.0], dtype=float)
            d_opt /= np.linalg.norm(d_opt)

            # 取 TF（最新或图像时刻）
            t_query = Time.from_msg(msg.header.stamp) if TF_TIME_MODE == 'image' else Time()
            try:
                T_bt = self.tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, t_query,
                                                       timeout=RclDuration(seconds=0.2))
            except TransformException as ex:
                self.get_logger().warn(f"TF 查找失败（{TF_TIME_MODE}，base<-tool0）：{ex}")
                return
            R_bt, p_bt = tfmsg_to_Rp(T_bt)

            # base 下相机位姿（用 optical）
            R_bc = R_bt @ self.R_t_co
            p_bc = R_bt @ self.p_t_co + p_bt

            # 射线 -> base，与 z=Z_VIRT 求交
            d_base = R_bc @ d_opt
            dz = float(d_base[2])
            if abs(dz) < 1e-6:
                self.get_logger().warn("视线近水平，无法与平面求交。")
                return
            t_star = (Z_VIRT - float(p_bc[2])) / dz
            if t_star < 0:
                self.get_logger().warn("交点在相机后方，忽略。")
                return
            C_raw = p_bc + t_star * d_base

            # 加 bias 的 C 作为圆心（仅取这一次）
            C = C_raw.copy()
            if ENABLE_BASE_BIAS:
                C[0] += float(BIAS_BASE_X)
                C[1] += float(BIAS_BASE_Y)
                C[2] += float(BIAS_BASE_Z)

            # 发布 TF（那一帧）
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

            # 发布 object_circle（与 object_position 一致，仅初次）
            if self._circle_center is None:
                self._circle_center = C.copy()
                self._ring_z = float(C[2] + HOVER_ABOVE)

                tf_circle = TransformStamped()
                tf_circle.header.stamp = tf_msg.header.stamp
                tf_circle.header.frame_id = BASE_FRAME
                tf_circle.child_frame_id  = OBJECT_CIRCLE_FRAME
                tf_circle.transform.translation.x = float(C[0])
                tf_circle.transform.translation.y = float(C[1])
                tf_circle.transform.translation.z = float(C[2])
                tf_circle.transform.rotation.x = 0.0
                tf_circle.transform.rotation.y = 0.0
                tf_circle.transform.rotation.z = 0.0
                tf_circle.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(tf_circle)
                self._circle_tf = tf_circle

            # 悬停位姿（末端朝下，绕X轴180° → (1,0,0,0)）
            ps = PoseStamped()
            ps.header.frame_id = POSE_FRAME
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = float(C[0])
            ps.pose.position.y = float(C[1])
            ps.pose.position.z = float(C[2] + HOVER_ABOVE)
            ps.pose.orientation.x = 1.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = 0.0
            ps.pose.orientation.w = 0.0
            self._fixed_hover_pose = ps

            self.get_logger().info(
                f"[detect once] C_corr=({C[0]:.3f},{C[1]:.3f},{C[2]:.3f}), hover_z={ps.pose.position.z:.3f}"
            )

        except Exception as e:
            self.get_logger().error(f"处理失败：{e}")
        finally:
            self._busy = False

    # ---------- IK 客户端 ----------
    def _ensure_ik_client(self) -> bool:
        if self._ik_client:
            return True
        for name in self._ik_candidates:
            cli = self.create_client(GetPositionIK, name)
            if cli.wait_for_service(timeout_sec=0.5):
                self._ik_client = cli
                self.get_logger().info(f"IK 服务可用：{name}")
                return True
        if 'wait_ik' not in self._warned_once:
            self._warned_once.add('wait_ik')
            self.get_logger().warn(f"等待 IK 服务…（尝试：{self._ik_candidates}）")
        return False

    # ---------- 种子：优先用上一次“我们自己发出的解” ----------
    def _get_seed(self) -> Optional[JointState]:
        if self._seed_hint is not None:
            js = JointState()
            js.name = UR5E_JOINT_ORDER
            js.position = [float(a) for a in self._seed_hint]
            return js
        if self._last_js:
            return self._last_js
        if 'wait_js' not in self._warned_once:
            self._warned_once.add('wait_js')
            self.get_logger().warn("等待 /joint_states …")
        return None

    def _request_ik(self, pose: PoseStamped, seed: JointState, move_time: float):
        req = GetPositionIK.Request()
        req.ik_request.group_name = GROUP_NAME
        req.ik_request.ik_link_name = IK_LINK_NAME
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = False
        req.ik_request.robot_state.joint_state = seed
        req.ik_request.timeout = MsgDuration(
            sec=int(IK_TIMEOUT),
            nanosec=int((IK_TIMEOUT % 1.0) * 1e9),
        )
        self._inflight = True
        fut = self._ik_client.call_async(req)
        fut.add_done_callback(lambda f, sd=seed, mt=move_time: self._on_ik_done(f, mt, sd))

    def _on_ik_done(self, fut, move_time: float, seed: JointState):
        self._inflight = False
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"IK 调用异常：{e}")
            return
        if res is None or res.error_code.val != 1:
            code = None if res is None else res.error_code.val
            self.get_logger().error(f"IK 未成功（error_code={code}）。")
            return

        name_to_idx = {n: i for i, n in enumerate(res.solution.joint_state.name)}
        # 参考角：优先用软件种子（上一解），否则用传入 seed
        if self._seed_hint is not None:
            ref = {n: float(a) for n, a in zip(UR5E_JOINT_ORDER, self._seed_hint)}
        else:
            ref = {n: float(p) for n, p in zip(seed.name, seed.position)}

        target_positions: List[float] = []
        jumps = []
        missing = []
        for jn in UR5E_JOINT_ORDER:
            if jn not in name_to_idx:
                missing.append(jn)
                continue
            raw = float(res.solution.joint_state.position[name_to_idx[jn]])
            near = _wrap_to_near(raw, ref.get(jn, raw))
            target_positions.append(near)
            if jn not in SAFE_JUMP_IGNORE_JOINTS:
                jumps.append(abs(near - ref.get(jn, near)))

        if missing:
            self.get_logger().error(f"IK 结果缺少关节: {missing}")
            return

        # 跳变检测（仅非忽略关节）
        max_jump = max(jumps) if jumps else 0.0
        if max_jump > MAX_WARN_JUMP_RAD:
            self.get_logger().warn(
                f"检测到异常关节跳变（max={max_jump:.3f} rad > {MAX_WARN_JUMP_RAD:.3f}），跳过该顶点。"
            )
            return
        if max_jump > MAX_SAFE_JUMP_RAD:
            self.get_logger().warn(
                f"关节跳变偏大（max={max_jump:.3f} rad > {MAX_SAFE_JUMP_RAD:.3f}），已发布但请留意。"
            )

        # 保存为下一步的软件种子
        self._seed_hint = np.array(target_positions, dtype=float)

        traj = JointTrajectory()
        traj.joint_names = UR5E_JOINT_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = target_positions
        pt.time_from_start = MsgDuration(sec=int(move_time), nanosec=int((move_time % 1.0) * 1e9))
        traj.points = [pt]
        self.pub_traj.publish(traj)

        self.get_logger().info(
            "已发布关节目标：[" + ", ".join(f"{v:.6f}" for v in target_positions) + f"], T={move_time:.1f}s"
        )
        self._motion_due_ns = self.get_clock().now().nanoseconds + int((move_time + 0.3) * 1e9)

    def _ik_to_vertex(self, ps: PoseStamped, move_time: float):
        if not self._ensure_ik_client():
            return
        seed = self._get_seed()
        if seed is None:
            return
        self._request_ik(ps, seed, move_time)

    # ---------- 生成 N 个等分顶点（不闭合；起点=当前EE朝向） ----------
    def _make_polygon_vertices(self, C: np.ndarray, ring_z: float, start_yaw: float) -> List[PoseStamped]:
        """
        令顶点角 θ 的位置 p = C + r[cosθ, sinθ]。
        - ORIENT_MODE='radial_in'  → 期望 yaw = θ + π
        - ORIENT_MODE='radial_out' → 期望 yaw = θ
        - ORIENT_MODE='tangent'    → 期望 yaw = θ + s·π/2，s=+1(ccw) / -1(cw)

        这里引入 START_DIR_OFFSET_DEG（正=逆时针）对 θ 统一加偏移，同时 yaw 减同样的偏移，
        从而满足：起点处 yaw(θ0) = start_yaw，且第一步位移方向可按偏移动。
        """
        n = max(3, int(POLY_N_VERTICES))
        turns = max(1, int(POLY_NUM_TURNS))
        total_deg = 360.0 * turns
        ccw = (POLY_DIR.strip().lower() == 'ccw')
        step = math.radians(total_deg / n)
        dir_sign = +1.0 if ccw else -1.0
        s_tan = +1.0 if ccw else -1.0

        offset = math.radians(START_DIR_OFFSET_DEG)

        # 由“当前 yaw”反解基础起点角，再叠加 offset 作为真实起点角 θ0
        if ORIENT_MODE == 'radial_in':
            theta0 = (start_yaw - math.pi) + offset
        elif ORIENT_MODE == 'radial_out':
            theta0 = start_yaw + offset
        else:  # tangent
            theta0 = (start_yaw - s_tan * (math.pi / 2)) + offset

        wps: List[PoseStamped] = []
        for i in range(n):
            th = theta0 + dir_sign * (i * step)       # 顶点位置角
            px = C[0] + CIRCLE_RADIUS * math.cos(th)
            py = C[1] + CIRCLE_RADIUS * math.sin(th)
            p  = np.array([px, py, ring_z], dtype=float)

            # yaw 减去相同的 offset，保证 i=0 时 yaw == start_yaw
            if ORIENT_MODE == 'radial_in':
                yaw = th + math.pi - offset
            elif ORIENT_MODE == 'radial_out':
                yaw = th - offset
            else:
                yaw = th + s_tan * (math.pi/2) - offset

            q_wxyz = yaw_to_quat_wxyz(yaw, sign=TOOL_Z_SIGN)
            ps = pose_from_pq(p, q_wxyz, POSE_FRAME)
            ps.header.stamp = self.get_clock().now().to_msg()
            wps.append(ps)

        return wps  # 不闭合


    # ---------- 收尾 ----------
    def _shutdown_once(self):
        self.get_logger().info("seeanything_minimal_clean 圆周完成并返回初始姿态，退出。")
        if rclpy.ok():
            rclpy.shutdown()


def main():
    rclpy.init()
    node = SeeAnythingMinimal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
