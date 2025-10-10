#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything_minimal_clean.py — GroundingDINO + 虚拟平面投影
（含 XY 硬补偿 + UV 径向补偿 + “两次检测/两段运动” + 一次最终下降悬停）

流程：
1) 上电 -> 发布初始位姿 -> 等静止
2) 第一次检测：得到 C1（含手动偏差），构造“仅改 XY、不改 Z”的靠近姿态，IK + 运动
3) 到位静止
4) 第二次检测：得到 C2（含手动偏差），构造“改 XY + 下降到 C2.z + HOVER_ABOVE”的姿态，IK + 运动
5) 完成退出

说明：
- BIAS_BASE_* 为手动偏差，保留。
- ENABLE_UV_RADIAL_FIX 等补偿参数保留，可关。
"""

# ====== 快速误差补偿（默认启用） ======
ENABLE_BASE_BIAS = True
BIAS_BASE_X = -0.03   # 工作台左右，-为往右（单位 m）
BIAS_BASE_Y = -0.17   # 工作台上下（单位 m）
BIAS_BASE_Z = 0.00    # Z 平面误差（通常 0）

# ====== UV 径向/非等向补偿（像素域→光学域之前；默认启用） ======
ENABLE_UV_RADIAL_FIX = True
K1_ISO = 0.2
K2_ISO = 0.1
ENABLE_UV_ANISO = True
LONG_EDGE_AXIS = 'auto'  # 'auto' | 'x' | 'y'
R_ECC = 1.55
KX1 = 0.2
KY1 = 0.02
UV_SCALE_MIN = 0.85
UV_SCALE_MAX = 1.35

from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import math
import cv2

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

# ================== 基本配置（按需修改） ==================
IMAGE_TOPIC = '/my_camera/pylon_ros2_camera_node/image_raw'

BASE_FRAME   = 'base_link'
TOOL_FRAME   = 'tool0'
OBJECT_FRAME = 'object_position'
Z_VIRT       = 0.0   # 工作面高度（base 下）

# 相机内参（像素系）
FX = 2674.3803723910564
FY = 2667.4211254043507
CX = 954.5922081613583
CY = 1074.965947832258

# 手眼外参：tool -> camera_(optical or link)
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
TF_TIME_MODE         = 'latest'   # 'image' | 'latest'  建议 'latest' 可避免 extrapolation
FRAME_STRIDE         = 2          # 每 N 帧处理一帧（>=1）
DEBUG_WINDOW         = False
DRAW_BEST_BOX        = False
DEBUG_HZ             = 5.0
TF_REBROADCAST_HZ    = 20.0
FLIP_X               = False
FLIP_Y               = False

# ===== IK / 控制 =====
POSE_FRAME       = 'base_link'
HOVER_ABOVE      = 0.30                 # 第二段运动的目标：最终 z = C2.z + HOVER_ABOVE
GROUP_NAME       = 'ur_manipulator'
IK_LINK_NAME     = 'tool0'
IK_TIMEOUT       = 2.0                  # s
CONTROLLER_TOPIC = '/scaled_joint_trajectory_controller/joint_trajectory'
MOVE_TIME        = 3.0                  # s

# 初始位姿（到位后才开始检测）
UR5E_JOINT_ORDER = [
    'shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
    'wrist_1_joint','wrist_2_joint','wrist_3_joint'
]
INIT_POS = [
    0.7734344005584717, -1.0457398456386109, 1.0822847525226038,
   -1.581707616845602, -1.5601266066180628, -0.8573678175555628,
]
INIT_MOVE_TIME  = 3.0  # s
INIT_EXTRA_WAIT = 0.5  # s

# 仅静止时触发检测
REQUIRE_STATIONARY  = True
VEL_EPS_RAD_PER_SEC = 0.02

# ================== 你的 DINO 封装 ==================
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


R_CL_CO = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0]
], dtype=float)


def correct_uv_with_radial(u: float, v: float, cx: float, cy: float,
                           fx: float, fy: float,
                           img_w: int, img_h: int) -> Tuple[float, float]:
    if not ENABLE_UV_RADIAL_FIX:
        return u, v
    x = (u - cx) / fx
    y = (v - cy) / fy
    r2 = x*x + y*y
    scale_iso = 1.0 + K1_ISO * r2 + K2_ISO * (r2*r2)
    sx, sy = scale_iso, scale_iso
    if ENABLE_UV_ANISO:
        if LONG_EDGE_AXIS == 'x':
            long_x = True
        elif LONG_EDGE_AXIS == 'y':
            long_x = False
        else:
            long_x = (img_w >= img_h)
        if long_x:
            r2_e = (x * R_ECC) ** 2 + (y ** 2)
            sx = scale_iso * (1.0 + KX1 * r2_e)
            sy = scale_iso * (1.0 + KY1 * r2_e)
        else:
            r2_e = (x ** 2) + (y * R_ECC) ** 2
            sx = scale_iso * (1.0 + KX1 * r2_e)
            sy = scale_iso * (1.0 + KY1 * r2_e)
    sx = min(max(sx, UV_SCALE_MIN), UV_SCALE_MAX)
    sy = min(max(sy, UV_SCALE_MIN), UV_SCALE_MAX)
    x_corr = x * sx
    y_corr = y * sy
    u_corr = cx + x_corr * fx
    v_corr = cy + y_corr * fy
    return float(u_corr), float(v_corr)


class SeeAnythingMinimal(Node):
    def __init__(self):
        super().__init__('seeanything_minimal_clean')

        # QoS：丢帧避免积压
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, IMAGE_TOPIC, self._cb_image, qos)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # DINO
        self.predictor = GroundingDinoPredictor(DINO_MODEL_ID, DINO_DEVICE)

        # 预计算 tool <- camera_optical
        R_t_cam = quat_to_rot(*T_TOOL_CAM_QUAT.tolist())
        if HAND_EYE_FRAME.lower() == 'optical':
            self.R_t_co = R_t_cam
        else:
            self.R_t_co = R_t_cam @ R_CL_CO
        self.p_t_co = T_TOOL_CAM_XYZ

        # 最近一次有效 TF（用于重广播）
        self._last_good_tf: Optional[TransformStamped] = None
        if TF_REBROADCAST_HZ > 0:
            self.create_timer(1.0/TF_REBROADCAST_HZ, self._rebroadcast_tf)

        # ---- IK / 轨迹发布 / 状态机 ----
        self.pub_traj = self.create_publisher(JointTrajectory, CONTROLLER_TOPIC, 1)
        self._ik_client = None
        self._ik_candidates = ['/compute_ik', '/move_group/compute_ik']

        qos_js = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                            history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        self._last_js: Optional[JointState] = None
        self.create_subscription(JointState, '/joint_states', self._on_js, qos_js)

        # 状态
        # init_needed -> init_moving -> wait_detect_coarse -> ik1_sent -> moving1
        # -> wait_detect_fine -> ik2_sent -> done
        self._phase = 'init_needed'
        self._busy = False
        self._inflight = False
        self._motion_due_ns: Optional[int] = None
        self._done = False
        self._warned_once = set()
        self._frame_count = 0

        # 缓存两次检测生成的目标姿态
        self._pose_stage1: Optional[PoseStamped] = None
        self._pose_stage2: Optional[PoseStamped] = None

        # tick 主循环
        self.create_timer(0.05, self._tick)

        self.get_logger().info(
            f"[seeanything_minimal_clean] topic={IMAGE_TOPIC}, "
            f"UVfix={'on' if ENABLE_UV_RADIAL_FIX else 'off'} aniso={'on' if ENABLE_UV_ANISO else 'off'}, "
            f"bias=({BIAS_BASE_X:.3f},{BIAS_BASE_Y:.3f},{BIAS_BASE_Z:.3f}), hover={HOVER_ABOVE:.3f}m"
        )

    # ---------- TF 重广播 ----------
    def _rebroadcast_tf(self):
        if self._last_good_tf is None:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._last_good_tf.header.frame_id
        t.child_frame_id  = self._last_good_tf.child_frame_id
        t.transform = self._last_good_tf.transform
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

    def _get_tool_z_now(self) -> Optional[float]:
        """读取当前 tool0 在 base 下的 z 高度。"""
        try:
            T_bt = self.tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, Time(),
                                                   timeout=RclDuration(seconds=0.2))
            _, p_bt = tfmsg_to_Rp(T_bt)
            return float(p_bt[2])
        except Exception as ex:
            self.get_logger().warn(f"读取 tool z 失败：{ex}")
            return None

    # ---------- 主循环 ----------
    def _tick(self):
        if self._done or self._inflight:
            return

        if self._phase == 'init_needed':
            self._publish_init_pose()
            self._phase = 'init_moving'
            return

        if self._phase == 'init_moving':
            if self._is_stationary():
                self._phase = 'wait_detect_coarse'
                self.get_logger().info("初始位姿到位，等待第一次检测（粗定位，仅改 XY）。")
            return

        if self._phase == 'wait_detect_coarse' and self._pose_stage1 is not None:
            if not self._ensure_ik_client():
                return
            seed = self._get_seed()
            if seed is None:
                return
            self._request_ik(self._pose_stage1)
            self._phase = 'ik1_sent'
            return

        if self._phase == 'ik1_sent':
            # 等 IK 回调设置 moving1
            return

        if self._phase == 'moving1':
            if self._is_stationary():
                self._pose_stage1 = None  # 清掉旧的
                self._phase = 'wait_detect_fine'
                self.get_logger().info("已靠近目标（保持 Z），开始第二次检测（精定位 + 下降）。")
            return

        if self._phase == 'wait_detect_fine' and self._pose_stage2 is not None:
            if not self._ensure_ik_client():
                return
            seed = self._get_seed()
            if seed is None:
                return
            self._request_ik(self._pose_stage2)
            self._phase = 'ik2_sent'
            return

        if self._phase in ('ik2_sent',):
            # 等 IK 回调结束
            return

    # ---------- 初始位姿 ----------
    def _publish_init_pose(self):
        traj = JointTrajectory()
        traj.joint_names = UR5E_JOINT_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = INIT_POS
        pt.time_from_start = MsgDuration(sec=int(INIT_MOVE_TIME),
                                         nanosec=int((INIT_MOVE_TIME % 1.0) * 1e9))
        traj.points = [pt]
        self.pub_traj.publish(traj)
        self.get_logger().info("已发布初始位姿…")
        now_ns = self.get_clock().now().nanoseconds
        self._motion_due_ns = now_ns + int((INIT_MOVE_TIME + INIT_EXTRA_WAIT) * 1e9)

    # ---------- 图像回调（两阶段逻辑） ----------
    def _cb_image(self, msg: Image):
        if self._done or self._inflight or self._busy:
            return
        if self._phase not in ('wait_detect_coarse', 'wait_detect_fine'):
            return
        if not self._is_stationary():
            return

        # 丢帧限速
        if FRAME_STRIDE > 1:
            self._frame_count += 1
            if (self._frame_count % FRAME_STRIDE) != 0:
                return

        self._busy = True
        try:
            # 1) 图像到 PIL
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil = PILImage.fromarray(rgb)

            # 2) DINO 推理：取最高分框中心 (u,v)
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

            s = np.array([_safe_float(s) if s is not None else -1.0 for s in scores], dtype=float)
            best = int(np.argmax(s))
            bx = boxes[best]
            x0, y0, x1, y1 = (bx.tolist() if hasattr(bx, 'tolist') else bx)
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)
            sc = s[best]

            # 3) UV 径向/非等向补偿
            h, w = rgb.shape[:2]
            u, v = correct_uv_with_radial(u, v, CX, CY, FX, FY, w, h)

            # 4) 像素 -> 光学系视线（你的最简二维变换）
            x_n0 = (u - CX) / FX
            y_n0 = (v - CY) / FY
            x_n =  y_n0
            y_n = -x_n0
            if FLIP_X: x_n = -x_n
            if FLIP_Y: y_n = -y_n
            d_opt = np.array([x_n, y_n, 1.0], dtype=float)
            d_opt /= np.linalg.norm(d_opt)

            # 5) TF：T^B_T 和 T^B_C
            t_query = Time.from_msg(msg.header.stamp) if TF_TIME_MODE == 'image' else Time()
            try:
                T_bt = self.tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, t_query,
                                                       timeout=RclDuration(seconds=0.2))
            except TransformException as ex:
                self.get_logger().warn(f"TF 查找失败（{TF_TIME_MODE}，base<-tool0）：{ex}")
                return
            R_bt, p_bt = tfmsg_to_Rp(T_bt)
            R_bc = R_bt @ self.R_t_co
            p_bc = R_bt @ self.p_t_co + p_bt

            # 6) 光线与 z=Z_VIRT 求交
            d_base = R_bc @ d_opt
            nrm = np.linalg.norm(d_base)
            if nrm < 1e-9:
                self.get_logger().warn("方向向量异常。")
                return
            d_base /= nrm
            o_base = p_bc
            dz = float(d_base[2])
            if abs(dz) < 1e-6:
                self.get_logger().warn("视线近水平，无法与平面求交。")
                return
            t_star = (Z_VIRT - float(o_base[2])) / dz
            if t_star < 0:
                self.get_logger().warn("交点在相机后方，忽略。")
                return
            C_raw = o_base + t_star * d_base

            # 6.5) 手动偏差（base 下）
            C = C_raw.copy()
            if ENABLE_BASE_BIAS:
                C[0] += float(BIAS_BASE_X)
                C[1] += float(BIAS_BASE_Y)
                C[2] += float(BIAS_BASE_Z)

            # 7) 发布 TF（用于 RViz/调试）
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

            # 8) 根据阶段生成目标姿态
            if self._phase == 'wait_detect_coarse':
                # 阶段1：仅改 XY，不改 Z（保持当前 tool0 的 z）
                z_keep = self._get_tool_z_now()
                if z_keep is None:
                    self.get_logger().warn("无法获取当前 Z，高度保持失败。")
                    return
                ps = PoseStamped()
                ps.header.frame_id = POSE_FRAME
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.pose.position.x = float(C[0])
                ps.pose.position.y = float(C[1])
                ps.pose.position.z = float(z_keep)
                ps.pose.orientation.x = 1.0
                ps.pose.orientation.y = 0.0
                ps.pose.orientation.z = 0.0
                ps.pose.orientation.w = 0.0
                self._pose_stage1 = ps
                self.get_logger().info(
                    f"[阶段1] score={sc:.3f} uv_corr=({u:.1f},{v:.1f}) "
                    f"C=({C[0]:.3f},{C[1]:.3f},{C[2]:.3f}) -> 仅XY靠近, Z保持={z_keep:.3f}"
                )

            elif self._phase == 'wait_detect_fine':
                # 阶段2：改 XY + 下降到 C2.z + HOVER_ABOVE
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
                self._pose_stage2 = ps
                self.get_logger().info(
                    f"[阶段2] score={sc:.3f} uv_corr=({u:.1f},{v:.1f}) "
                    f"C=({C[0]:.3f},{C[1]:.3f},{C[2]:.3f}) -> XY修正 + 下降到 {ps.pose.position.z:.3f}"
                )

            # 可选调试图
            if DEBUG_WINDOW and DRAW_BEST_BOX:
                dbg = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
                x0i, y0i, x1i, y1i = map(int, [x0, y0, x1, y1])
                txt = f"{sc:.2f}" if sc >= 0 else ""
                color = (0, 255, 0)
                cv2.rectangle(dbg, (x0i, y0i), (x1i, y1i), color, 2)
                if txt:
                    cv2.putText(dbg, txt, (x0i, max(0, y0i-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.circle(dbg, (int(round(u)), int(round(v))), 5, (255,0,0), -1)
                cv2.imshow("DINO Debug", dbg)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"处理失败：{e}")
        finally:
            self._busy = False

    # ---------- IK ----------
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

    def _get_seed(self) -> Optional[JointState]:
        if self._last_js:
            return self._last_js
        if 'wait_js' not in self._warned_once:
            self._warned_once.add('wait_js')
            self.get_logger().warn("等待 /joint_states …")
        return None

    def _request_ik(self, pose: PoseStamped):
        req = GetPositionIK.Request()
        req.ik_request.group_name = GROUP_NAME
        req.ik_request.ik_link_name = IK_LINK_NAME
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = False
        # 用当前关节作为 seed
        if self._last_js:
            req.ik_request.robot_state.joint_state = self._last_js
        req.ik_request.timeout = MsgDuration(
            sec=int(IK_TIMEOUT),
            nanosec=int((IK_TIMEOUT % 1.0) * 1e9),
        )
        self._inflight = True
        fut = self._ik_client.call_async(req)
        fut.add_done_callback(self._on_ik_done)

    def _on_ik_done(self, fut):
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
        target_positions: List[float] = []
        missing = []
        for jn in UR5E_JOINT_ORDER:
            if jn not in name_to_idx:
                missing.append(jn)
            else:
                target_positions.append(float(res.solution.joint_state.position[name_to_idx[jn]]))
        if missing:
            self.get_logger().error(f"IK 结果缺少关节: {missing}")
            return

        traj = JointTrajectory()
        traj.joint_names = UR5E_JOINT_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = target_positions
        pt.time_from_start = MsgDuration(sec=int(MOVE_TIME), nanosec=int((MOVE_TIME % 1.0) * 1e9))
        traj.points = [pt]
        self.pub_traj.publish(traj)

        self.get_logger().info(
            "已发布关节目标：[" + ", ".join(f"{v:.6f}" for v in target_positions) + f"], T={MOVE_TIME:.1f}s"
        )
        # 设定运动缓冲时间，等待静止再进入下一阶段
        now_ns = self.get_clock().now().nanoseconds
        self._motion_due_ns = now_ns + int((MOVE_TIME + 0.3) * 1e9)

        if self._phase == 'ik1_sent':
            self._phase = 'moving1'
        elif self._phase == 'ik2_sent':
            self._phase = 'done'
            # 给轨迹一点时间下发，再退出
            self.create_timer(0.5, self._shutdown_once)

    # ---------- 收尾 ----------
    def _shutdown_once(self):
        self.get_logger().info("seeanything_minimal_clean 两段运动完成，退出。")
        if rclpy.ok():
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
