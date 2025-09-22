#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything_ibvs_seek.py — 估计器 + 逐步下降 + 复获 + 圆周扫描
- u/v → 射线 → 虚拟平面交点 的几何与 minimal_clean 完全一致
- 两类“微调”：
  (A) 控制侧居中微调：ENABLE_CENTER_BIAS / BIAS_CENTER_X / BIAS_CENTER_Y（仅影响 IBVS XY 控制量）
  (B) TF 硬编码偏置：ENABLE_TF_BIAS / TF_BIAS_X / TF_BIAS_Y / TF_BIAS_Z（仅影响发布的 object_position）
- 策略：可见→下降（10cm/步）；不可见→小幅 XY 复获；长丢→慢速圆周
- 估计器：EMA + 离群剔除；下降阶段锁 XY（只允许 ≤1cm/步修正），避免 TF 跟着相机漂移
- 轨迹节流：发布后在 MOVE_TIME_SEC 内不再发下一条，降低 IK -31 风险
"""

from typing import Optional, Tuple, Dict, List
import math
import numpy as np
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


# =================== 运动限制 ===================
MAX_XY_STEP_PER_MOVE = 0.15   # m，单次 XY 平移上限（IBVS 与扫描均受限）
XY_SPEED_MAX = 0.05           # m/s，XY 限速
Z_SPEED_MAX  = 0.04           # m/s，Z 限速
MOVE_TIME_SEC = 3.0           # 每步轨迹时长（亦影响节流窗口）

# =================== 控制侧极性（仅影响 IBVS 控制，不影响 TF/几何） ===================
CTRL_SIGN_X = 1.0
CTRL_SIGN_Y = 1.0

# =================== 居中阶段的控制侧微调（仅作用在控制量上，不影响 TF） ===================
ENABLE_CENTER_BIAS = True
BIAS_CENTER_X = -0.10   # m，+X 让相机向 +X 平移
BIAS_CENTER_Y = -0.10   # m，+Y 让相机向 +Y 平移

# =================== TF 硬编码偏置（只影响发布的 object_position） ===================
ENABLE_TF_BIAS = True
TF_BIAS_X = -0.10       # m
TF_BIAS_Y = -0.10       # m
TF_BIAS_Z =  0.00       # m

# =================== 初始观察位姿 ===================
INIT_AT_START = True
UR5E_ORDER = [
    "shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
    "wrist_1_joint","wrist_2_joint","wrist_3_joint",
]
INIT_POS = [ 0.9239029288291931, -1.186562405233719, 1.1997712294207972,
            -1.5745235882201136, -1.5696094671832483, -0.579871956502096 ]
INIT_MOVE_TIME_SEC = 3.0
INIT_EXTRA_WAIT_SEC = 0.3

# =================== 话题/坐标 ===================
IMAGE_TOPIC = "/my_camera/pylon_ros2_camera_node/image_raw"
BASE_FRAME = "base_link"
TOOL_FRAME = "tool0"
POSE_FRAME = "base_link"
OBJECT_FRAME = "object_position"   # 估计/锁定后的 TF（控制用）
OBJECT_RAW_FRAME = "object_raw"    # 原始测量（可选）
PUBLISH_OBJECT_RAW = True
CONTROLLER_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"

# =================== 相机与手眼（与 minimal_clean 一致） ===================
FX = 2674.3803723910564
FY = 2667.4211254043507
CX = 954.5922081613583
CY = 1074.965947832258
# tool <- camera_(optical or link)
T_TOOL_CAM_XYZ  = np.array([-0.000006852374024, -0.099182661943126947, 0.02391824813032688], dtype=float)
T_TOOL_CAM_QUAT = np.array([-0.0036165657530785695, -0.000780788838366878, 0.7078681983794892, 0.7063348529868249], dtype=float)
HAND_EYE_FRAME  = "optical"  # 'optical' or 'link'
# camera_link <- camera_optical（REP-105）
R_CL_CO = np.array([[0.0,0.0,1.0],[-1.0,0.0,0.0],[0.0,-1.0,0.0]], dtype=float)

# =================== 虚拟平面与目标高度 ===================
Z_VIRT = 0.0
HOVER_ABOVE = 0.30           # 最终停在平面上方（tool0.z = Z_VIRT + HOVER_ABOVE）
DESCENT_STEP = 0.10          # 每次下降 10 cm

# =================== DINO & 判定 ===================
TEXT_PROMPT   = "yellow object ."
DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DINO_DEVICE   = "cuda"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD= 0.25

SCORE_OK      = 0.6          # 认为“看见”的置信度
SCORE_LOST    = 0.4          # 认为“丢失”的置信度
STABLE_FRAMES = 3            # 连续命中 N 帧才确认
MISS_FRAMES   = 3            # 连续丢失 N 帧才判丢
REACQ_FRAMES  = 2            # 复获判稳帧数
PIX_CENTER_TOL= 25.0         # 视为居中的像素阈值
FRAME_STRIDE  = 2            # 图像限频

# =================== 估计器参数 ===================
EST_ALPHA_XY = 0.2           # EMA 系数（平面内）
EST_XY_LOCK_CORR_MAX = 0.01  # m，下降阶段 XY 每步最大修正
EST_OUTLIER_XY = 0.08        # m，XY 跳变>8cm 视为离群，忽略

# =================== 其他 ===================
TF_REBROADCAST_HZ = 20.0
TF_TIME_MODE = "latest"
DEBUG_WINDOW = False
DRAW_BEST_BOX = False
DEBUG_HZ = 5.0
FLIP_X = False
FLIP_Y = False

# =================== 工具函数 ===================
def quat_to_rot(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def tfmsg_to_Rp(T: TransformStamped):
    q = T.transform.rotation
    t = T.transform.translation
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    p = np.array([t.x, t.y, t.z], dtype=float)
    return R, p

def clamp_xy(vec3: np.ndarray, cap: float) -> np.ndarray:
    """仅裁剪 XY 分量的长度到 cap（保留 z 不变）"""
    xy = vec3[:2]
    n = float(np.linalg.norm(xy))
    if n > cap and n > 1e-9:
        vec3 = vec3.copy()
        vec3[0] = xy[0] * (cap / n)
        vec3[1] = xy[1] * (cap / n)
    return vec3

# =================== 主节点 ===================
class SeeAnythingIBVSSeek(Node):
    def __init__(self):
        super().__init__("seeanything_ibvs_seek")

        # 可运行时覆盖的控制极性
        self.declare_parameter('ctrl_sign_x', CTRL_SIGN_X)
        self.declare_parameter('ctrl_sign_y', CTRL_SIGN_Y)
        self.ctrl_sign_x = float(self.get_parameter('ctrl_sign_x').value)
        self.ctrl_sign_y = float(self.get_parameter('ctrl_sign_y').value)

        # 图像
        qos_img = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        self.create_subscription(Image, IMAGE_TOPIC, self._cb_image, qos_img)

        # /joint_states
        if True:  # 可靠
            qos_js = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
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
            sys.path.append("/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything")
            from gripanything.core.detect_with_dino import GroundingDinoPredictor
        self.predictor = GroundingDinoPredictor(DINO_MODEL_ID, DINO_DEVICE)

        # hand-eye
        R_t_cam = quat_to_rot(*T_TOOL_CAM_QUAT.tolist())
        self.R_t_co = R_t_cam if HAND_EYE_FRAME.lower() == "optical" else (R_t_cam @ R_CL_CO)
        self.p_t_co = T_TOOL_CAM_XYZ

        # 控制 & IK
        self.pub_traj = self.create_publisher(JointTrajectory, CONTROLLER_TOPIC, 1)
        self._ik_client = None
        self._ik_candidates = ["/compute_ik", "/move_group/compute_ik"]
        self._inflight_ik = False
        self._pending_tag: Optional[str] = None

        # 缓存/状态
        self._frame_count = 0
        self._last_uv: Optional[Tuple[float, float]] = None
        self._last_score: float = float("nan")
        self._visible_streak = 0
        self._miss_streak = 0
        self._reacq_streak = 0
        self._pix_dist: float = float("inf")

        self._C_raw: Optional[np.ndarray] = None  # base 下测量交点
        self._est_xy: Optional[np.ndarray] = None # 估计的 XY
        self._last_good_tf: Optional[TransformStamped] = None

        self._scan_center_xy: Optional[np.ndarray] = None
        self._scan_z: Optional[float] = None
        self._scan_theta: float = 0.0

        self._z_step_goal_tool: Optional[float] = None
        self._mode = "INIT" if INIT_AT_START else "REACQUIRE"  # REACQUIRE / DESCEND / LOST_SCAN / DONE
        self._init_due_ns: Optional[int] = None
        self._init_published = False
        self._motion_busy_until_ns: int = 0  # 轨迹节流窗口
        self._warned = set()

        # 定时器
        if TF_REBROADCAST_HZ > 0:
            self.create_timer(1.0/TF_REBROADCAST_HZ, self._rebroadcast_tf)
        self.create_timer(0.1, self._tick)

        self.get_logger().info("seeanything_ibvs_seek: started.")

    # ===== 回调 =====
    def _on_js(self, msg: JointState):
        if self._last_js is None:
            self.get_logger().info(f"已收到 /joint_states（{len(msg.name)} 关节）。")
        self._last_js = msg

    def _cb_image(self, msg: Image):
        # 限帧
        self._frame_count += 1
        if FRAME_STRIDE > 1 and (self._frame_count % FRAME_STRIDE) != 0:
            return

        # 1) 解码
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().warn(f"图像解码失败: {e}")
            return
        pil = PILImage.fromarray(rgb)

        # 2) DINO
        try:
            out = self.predictor.predict(pil, TEXT_PROMPT, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)
        except Exception as e:
            self.get_logger().error(f"DINO 推理失败: {e}")
            return

        boxes, scores = [], []
        if isinstance(out, tuple) and len(out) == 3:
            boxes, labels, scores = out
        elif isinstance(out, tuple) and len(out) == 2:
            boxes, labels = out
            scores = [None] * len(boxes)

        if len(boxes) == 0:
            self._update_visibility(False, float("nan"))
            return

        s = np.array([float(si) if si is not None else -1.0 for si in scores], dtype=float)
        best = int(np.argmax(s))
        x0, y0, x1, y1 = (boxes[best].tolist() if hasattr(boxes[best], "tolist") else boxes[best])
        u = 0.5 * (x0 + x1)
        v = 0.5 * (y0 + y1)
        sc = float(s[best])
        self._pix_dist = math.hypot(u - CX, v - CY)

        # 3) TF：base<-tool
        t_query = Time.from_msg(msg.header.stamp) if TF_TIME_MODE == "image" else Time()
        try:
            T_bt = self.tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, t_query, timeout=RclDuration(seconds=0.2))
        except TransformException as ex:
            self.get_logger().warn(f"TF 查找失败（base<-tool0）：{ex}")
            self._update_visibility(False, sc)
            return
        R_bt, p_bt = tfmsg_to_Rp(T_bt)

        # 4) 相机位姿
        R_bc = R_bt @ self.R_t_co
        p_bc = R_bt @ self.p_t_co + p_bt

        # 5) u/v → 光学系方向
        x_n = (u - CX) / FX
        y_n = (v - CY) / FY
        if FLIP_X: x_n = -x_n
        if FLIP_Y: y_n = -y_n
        d_opt = np.array([x_n, y_n, 1.0], dtype=float)
        d_opt /= np.linalg.norm(d_opt)

        # 6) 射线到 base，与 z=Z_VIRT 求交（C_raw）
        d_base = R_bc @ d_opt
        d_norm = float(np.linalg.norm(d_base))
        if d_norm < 1e-9 or abs(float(d_base[2])) < 1e-6:
            self._update_visibility(False, sc)
            return
        d_base /= d_norm
        t_star = (Z_VIRT - float(p_bc[2])) / float(d_base[2])
        if t_star <= 0:
            self._update_visibility(False, sc)
            return
        C_raw = p_bc + t_star * d_base  # base 下交点（未经任何偏置）

        # 7) 估计器更新（仅 XY；Z 固定为 Z_VIRT）
        if self._est_xy is None:
            self._est_xy = C_raw[:2].copy()
        else:
            dxy = C_raw[:2] - self._est_xy
            dxy_norm = float(np.linalg.norm(dxy))
            if dxy_norm <= EST_OUTLIER_XY:
                if self._mode == "DESCEND":
                    # 降阶锁定：仅允许 ≤ 1cm/步 的微修正
                    if dxy_norm > EST_XY_LOCK_CORR_MAX:
                        dxy = dxy * (EST_XY_LOCK_CORR_MAX / max(dxy_norm, 1e-9))
                    self._est_xy += dxy
                else:
                    self._est_xy = (1.0 - EST_ALPHA_XY) * self._est_xy + EST_ALPHA_XY * C_raw[:2]
            # 否则离群，忽略更新

        # 8) 发布 RAW TF（可选）与偏置后的 object_position
        if PUBLISH_OBJECT_RAW:
            self._publish_tf(C_raw, OBJECT_RAW_FRAME)

        C_pub = C_raw.copy()
        if self._est_xy is not None:
            C_pub[0], C_pub[1] = float(self._est_xy[0]), float(self._est_xy[1])
        if ENABLE_TF_BIAS:
            C_pub[0] += TF_BIAS_X
            C_pub[1] += TF_BIAS_Y
            C_pub[2] += TF_BIAS_Z
        self._publish_tf(C_pub, OBJECT_FRAME)

        # 9) 可见性统计
        self._last_uv = (u, v)
        self._last_score = sc
        self._C_raw = C_raw
        self._update_visibility(sc >= SCORE_OK, sc)

        # 10) 可选画框
        if DEBUG_WINDOW and DRAW_BEST_BOX:
            dbg = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
            x0i, y0i, x1i, y1i = map(int, [x0, y0, x1, y1])
            cv2.rectangle(dbg, (x0i, y0i), (x1i, y1i), 2)
            cv2.circle(dbg, (int(round(u)), int(round(v))), 5, -1)
            cv2.imshow("DINO Debug", dbg)
            cv2.waitKey(1)

    # ===== 定时器主循环 =====
    def _tick(self):
        # 节流：轨迹执行窗口内不发新目标
        now_ns = self.get_clock().now().nanoseconds
        if now_ns < self._motion_busy_until_ns or self._inflight_ik:
            return

        # 初始位姿
        if self._mode == "INIT":
            self._publish_init_pose()
            self._init_due_ns = now_ns + int((INIT_MOVE_TIME_SEC + INIT_EXTRA_WAIT_SEC)*1e9)
            self._mode = "INIT_WAIT"
            return
        if self._mode == "INIT_WAIT":
            if self._init_due_ns is not None and now_ns >= self._init_due_ns:
                self.get_logger().info("初始位姿已到位，进入 REACQUIRE。")
                self._mode = "REACQUIRE"
            return

        # 获取 base<-tool，基础量
        try:
            T_bt = self.tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, Time(), timeout=RclDuration(seconds=0.2))
        except TransformException:
            return
        R_bt, p_bt = tfmsg_to_Rp(T_bt)
        q_bt = T_bt.transform.rotation
        R_bc = R_bt @ self.R_t_co
        p_bc = R_bt @ self.p_t_co + p_bt
        z_target_tool = Z_VIRT + HOVER_ABOVE

        # 模式转换判据
        if self._mode in ("REACQUIRE", "DESCEND"):
            if self._miss_streak >= MISS_FRAMES or (not (self._last_score >= SCORE_LOST)):
                # 长时间不可见 → 圆周搜索
                self._enter_scan(R_bt, p_bt)
                return

        # ========== REACQUIRE：小幅 XY 居中 ==========
        if self._mode == "REACQUIRE":
            # 满足条件 → 开始下降
            if (self._visible_streak >= STABLE_FRAMES) and (self._pix_dist <= PIX_CENTER_TOL):
                self._mode = "DESCEND"
                self._z_step_goal_tool = max(z_target_tool, float(p_bt[2]) - DESCENT_STEP)
                self.get_logger().info(f"进入 DESCEND：目标 tool0.z -> {self._z_step_goal_tool:.3f}")
                return

            # 否则做 IBVS 小步 XY（不改 z）
            if self._C_raw is None or self._last_uv is None:
                return
            u, v = self._last_uv
            du, dv = (u - CX), (v - CY)

            # 深度：把 C_raw 变换到相机帧，取 Z_cam
            C_cam = R_bc.T @ (self._C_raw - p_bc)
            Z_cam = float(C_cam[2])
            if Z_cam <= 1e-4:
                return

            # IBVS：像素误差 → 相机帧平移（仅由 CTRL_SIGN_* 控制极性）
            dX_cam = self.ctrl_sign_x * (du / FX) * Z_cam
            dY_cam = self.ctrl_sign_y * (dv / FY) * Z_cam
            # 适度增益
            IBVS_GAIN = 0.8
            d_cam  = np.array([IBVS_GAIN*dX_cam, IBVS_GAIN*dY_cam, 0.0], dtype=float)

            # → base
            d_base = R_bc @ d_cam

            # 限速 + 单步上限
            xy_step_cap = min(XY_SPEED_MAX * MOVE_TIME_SEC, MAX_XY_STEP_PER_MOVE)
            d_base = clamp_xy(d_base, xy_step_cap)

            # 控制侧微调（仅在 REACQUIRE 生效）
            if ENABLE_CENTER_BIAS:
                d_base[0] += BIAS_CENTER_X
                d_base[1] += BIAS_CENTER_Y
                d_base = clamp_xy(d_base, xy_step_cap)

            # 目标相机原点（保持 z）
            o_new = p_bc + d_base

            # → tool0 目标（姿态保持）
            p_bt_new = o_new - (R_bt @ self.p_t_co)
            if p_bt_new[2] < z_target_tool:
                p_bt_new[2] = z_target_tool

            self._send_pose(p_bt_new, q_bt, tag="reacq_xy")
            return

        # ========== DESCEND：优先向下，XY 锁定（仅微修） ==========
        if self._mode == "DESCEND":
            if self._C_raw is None or self._est_xy is None:
                # 没测量就回 REACQUIRE
                self._mode = "REACQUIRE"
                return

            # 下降目标（每步 10cm，限速）
            cam_offset_z = float((R_bt @ self.p_t_co)[2])
            oz_current = float(p_bc[2])
            oz_goal = float(max(z_target_tool, float(p_bt[2]) - DESCENT_STEP) + cam_offset_z) \
                      if self._z_step_goal_tool is None else float(self._z_step_goal_tool + cam_offset_z)

            z_step_cap = Z_SPEED_MAX * MOVE_TIME_SEC
            dz = oz_goal - oz_current
            dz = max(-z_step_cap, min(z_step_cap, dz))

            # 中心射线方向
            d_center = R_bc @ np.array([0.0, 0.0, 1.0], dtype=float)
            dzc = float(d_center[2])
            # 根据中心射线穿过“估计+偏置”的平面点，解算新的 ox,oy
            C_target_xy = self._est_xy.copy()
            if ENABLE_TF_BIAS:
                C_target_xy = C_target_xy + np.array([TF_BIAS_X, TF_BIAS_Y], dtype=float)

            oz_new = oz_current + dz
            if abs(dzc) < 1e-6:
                ox_new, oy_new = float(p_bc[0]), float(p_bc[1])  # 退化：保持 XY
            else:
                k = (Z_VIRT - oz_new) / dzc
                ox_new = float(C_target_xy[0] - k * float(d_center[0]))
                oy_new = float(C_target_xy[1] - k * float(d_center[1]))

            o_desired = np.array([ox_new, oy_new, oz_new], dtype=float)

            # 单步 XY 限幅（避免跨太远）
            d_xy = o_desired - p_bc
            d_xy = clamp_xy(d_xy, min(XY_SPEED_MAX * MOVE_TIME_SEC, MAX_XY_STEP_PER_MOVE))
            # Z 限幅（已在 dz 里做了），重新组装
            o_new = np.array([p_bc[0] + d_xy[0], p_bc[1] + d_xy[1], oz_new], dtype=float)

            # → tool0 目标
            p_bt_new = o_new - (R_bt @ self.p_t_co)
            if p_bt_new[2] < z_target_tool:
                p_bt_new[2] = z_target_tool

            # 完成判定
            if float(p_bt[2]) <= z_target_tool + 1e-4 and abs(oz_goal - oz_current) < 1e-4:
                self.get_logger().info(f"已到目标高度 {z_target_tool:.3f}，任务完成。")
                self._mode = "DONE"
                return

            # 若当前分数跌破阈值，转 REACQUIRE
            if not (self._last_score >= SCORE_LOST):
                self._mode = "REACQUIRE"
                return

            self._send_pose(p_bt_new, q_bt, tag="descend_step")
            return

        # ========== LOST_SCAN：慢速圆周 ==========
        if self._mode == "LOST_SCAN":
            # 复获后转 REACQUIRE
            if self._reacq_streak >= REACQ_FRAMES and self._visible_streak >= STABLE_FRAMES:
                self.get_logger().info("重获目标，进入 REACQUIRE。")
                self._mode = "REACQUIRE"
                self._z_step_goal_tool = None
                return

            # 扫描步进
            SCAN_RADIUS = 0.20
            SCAN_LINEAR_SPEED = 0.03
            arc_len = min(SCAN_LINEAR_SPEED * MOVE_TIME_SEC, MAX_XY_STEP_PER_MOVE)
            dtheta = arc_len / max(SCAN_RADIUS, 1e-6)
            self._scan_theta += dtheta

            cam_offset_z = float((R_bt @ self.p_t_co)[2])
            oz = float(self._scan_z + cam_offset_z)
            ox = float(self._scan_center_xy[0] + SCAN_RADIUS * math.cos(self._scan_theta))
            oy = float(self._scan_center_xy[1] + SCAN_RADIUS * math.sin(self._scan_theta))
            o_new = np.array([ox, oy, oz], dtype=float)

            p_bt_new = o_new - (R_bt @ self.p_t_co)
            if p_bt_new[2] < (Z_VIRT + HOVER_ABOVE):
                p_bt_new[2] = (Z_VIRT + HOVER_ABOVE)

            self._send_pose(p_bt_new, q_bt, tag="scan_arc")
            return

        # DONE
        if self._mode == "DONE":
            return

    # ===== 模式切换：进入圆周 =====
    def _enter_scan(self, R_bt, p_bt):
        p_bc = R_bt @ self.p_t_co + p_bt
        self._scan_center_xy = np.array([float(p_bc[0]), float(p_bc[1])], dtype=float)
        self._scan_z = float(p_bt[2])  # 保持当前高度
        self._scan_theta = 0.0
        self._z_step_goal_tool = None
        self.get_logger().info("目标丢失：进入圆周扫描（r=0.20m，慢速，单步XY≤15cm）。")
        self._mode = "LOST_SCAN"

    # ===== 可见性统计 =====
    def _update_visibility(self, visible: bool, score: float):
        if visible:
            self._visible_streak += 1
            self._reacq_streak += 1
            self._miss_streak = 0
        else:
            self._miss_streak += 1
            self._reacq_streak = 0

    # ===== 发送目标姿态（带节流） =====
    def _send_pose(self, p_bt_new: np.ndarray, q_bt, tag: str):
        if not self._ensure_ik_client():
            return
        seed = self._get_seed()
        if seed is None:
            return

        ps = PoseStamped()
        ps.header.frame_id = POSE_FRAME
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(p_bt_new[0])
        ps.pose.position.y = float(p_bt_new[1])
        ps.pose.position.z = float(p_bt_new[2])
        ps.pose.orientation = q_bt

        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.ik_link_name = "tool0"
        req.ik_request.pose_stamped = ps
        req.ik_request.avoid_collisions = False
        req.ik_request.robot_state.joint_state = seed
        req.ik_request.timeout = Duration(sec=2, nanosec=0)

        fut = self._ik_client.call_async(req)
        self._inflight_ik = True
        self._pending_tag = tag
        fut.add_done_callback(self._on_ik_done)

    def _on_ik_done(self, fut):
        self._inflight_ik = False
        tag = self._pending_tag
        self._pending_tag = None

        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"IK 异常[{tag}]: {e}")
            return
        if res is None or res.error_code.val != 1:
            code = None if res is None else res.error_code.val
            self.get_logger().error(f"IK 无解[{tag}], error_code={code}")
            return

        name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(res.solution.joint_state.name)}
        target_positions: List[float] = []
        for jn in UR5E_ORDER:
            if jn not in name_to_idx:
                self.get_logger().error(f"IK 结果缺少关节[{tag}]: {jn}")
                return
            target_positions.append(res.solution.joint_state.position[name_to_idx[jn]])

        traj = JointTrajectory()
        traj.joint_names = UR5E_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = target_positions
        pt.time_from_start = Duration(sec=int(MOVE_TIME_SEC), nanosec=int((MOVE_TIME_SEC % 1.0)*1e9))
        traj.points = [pt]
        self.pub_traj.publish(traj)

        # 轨迹节流窗口
        self._motion_busy_until_ns = self.get_clock().now().nanoseconds + int(MOVE_TIME_SEC * 1e9)
        self.get_logger().info(
            f"[{tag}] 已发布关节目标（步长XY≤{MAX_XY_STEP_PER_MOVE:.2f}m, T={MOVE_TIME_SEC:.2f}s, "
            f"ctrl_sign=({self.ctrl_sign_x:+.0f},{self.ctrl_sign_y:+.0f})）"
        )

    # ===== IK / 其它 =====
    def _ensure_ik_client(self) -> bool:
        if self._ik_client:
            return True
        for name in self._ik_candidates:
            if not name:
                continue
            cli = self.create_client(GetPositionIK, name)
            if cli.wait_for_service(timeout_sec=0.5):
                self._ik_client = cli
                self.get_logger().info(f"IK 服务可用：{name}")
                return True
        self._warn_once("wait_ik", f"等待 IK 服务…（尝试：{self._ik_candidates}）")
        return False

    def _get_seed(self) -> Optional[JointState]:
        if self._last_js:
            return self._last_js
        self._warn_once("wait_js", "等待 /joint_states 用作 IK 种子…")
        return None

    def _publish_tf(self, C_base: np.ndarray, child: str):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = BASE_FRAME
        tf_msg.child_frame_id = child
        tf_msg.transform.translation.x = float(C_base[0])
        tf_msg.transform.translation.y = float(C_base[1])
        tf_msg.transform.translation.z = float(C_base[2])
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tf_msg)
        if child == OBJECT_FRAME:
            self._last_good_tf = tf_msg

    def _rebroadcast_tf(self):
        if not self._last_good_tf:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._last_good_tf.header.frame_id
        t.child_frame_id = self._last_good_tf.child_frame_id
        t.transform = self._last_good_tf.transform
        self.tf_broadcaster.sendTransform(t)

    def _publish_init_pose(self):
        if self._init_published:
            return
        traj = JointTrajectory()
        traj.joint_names = UR5E_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = INIT_POS
        pt.time_from_start = Duration(sec=int(INIT_MOVE_TIME_SEC), nanosec=int((INIT_MOVE_TIME_SEC % 1.0)*1e9))
        traj.points = [pt]
        self.pub_traj.publish(traj)
        self._init_published = True
        self.get_logger().info("已发布初始观察位姿。")

    def _warn_once(self, key: str, text: str):
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warning(text)

    def destroy_node(self):
        try:
            if DEBUG_WINDOW:
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = SeeAnythingIBVSSeek()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
