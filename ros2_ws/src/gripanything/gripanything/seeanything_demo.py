#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything_minimal_clean.py — GroundingDINO + 虚拟平面投影（稳健版，含XY硬补偿 + UV径向补偿）

功能要点：
- 只使用“最高置信度”目标来计算并发布 TF（object_position），其余目标忽略
- 丢帧限速 + 最近一次有效 TF 周期性重广播，避免 RViz TF 断线
- 处理手眼外参坐标系差异（tool->camera_optical vs tool->camera_link）
- 在 base_link 下对求得的 3D 点做常量平移补偿（默认 X/Y 各 -0.05/-0.17 m）
- 新增：UV 径向/非等向补偿（离中心越远补偿越强，长边方向更强）以修正“靠近长边达不到”的误差
"""

# ====== 快速误差补偿（base 下的 XYZ 常量位移） ======
ENABLE_BASE_BIAS = True
BIAS_BASE_X = -0.05   # +X 前/右，单位米
BIAS_BASE_Y = -0.17   # +Y 左/侧，单位米
BIAS_BASE_Z = 0.00    # Z 平面误差（通常为 0）

# ====== UV 径向/非等向补偿（像素域→光学域之前；默认启用） ======
ENABLE_UV_RADIAL_FIX = True
# 等向（圆对称）分量：r^2 = x^2 + y^2，x=(u-CX)/FX, y=(v-CY)/FY
# 正常“长边达不到”→ 往外拉一丢丢：K1_ISO 取正且较小；K2_ISO 先置 0
K1_ISO = 0.1      # 0.00~0.08 之间调；正值 = 向外拉
K2_ISO = 0.00

# 非等向分量：对图像的“长边”方向增强补偿
ENABLE_UV_ANISO = True
LONG_EDGE_AXIS = 'auto'  # 'auto' | 'x' | 'y'；auto=根据图像宽高自动判断
R_ECC = 1.35             # 椭圆半径拉伸比例（>1 表示长边半径更“远”）
KX1 = 0.1               # x 方向附加系数（长边若为 x 建议更大）
KY1 = 0.02               # y 方向附加系数（短边方向较小）

# 安全限幅（避免过度拉伸/收缩）
UV_SCALE_MIN = 0.85      # 允许的最小缩放
UV_SCALE_MAX = 1.35      # 允许的最大放大

from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import math
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration as RclDuration
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf2_ros import TransformException
from cv_bridge import CvBridge
from PIL import Image as PILImage

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
DEBUG_WINDOW         = False      # 是否弹本地窗口（默认关）
DRAW_BEST_BOX        = False      # 只画“最佳”那个框（默认不画，以免卡顿）
DEBUG_HZ             = 5.0        # 调试窗口最大刷新 Hz
TF_REBROADCAST_HZ    = 20.0       # 最近一次有效 TF 的重广播频率
FLIP_X               = False      # 若左右镜像，可把这里改为 True
FLIP_Y               = False      # 若上下镜像，可把这里改为 True

# ================== 你的 DINO 封装 ==================
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except Exception:
    import sys
    sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
    from gripanything.core.detect_with_dino import GroundingDinoPredictor


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


# camera_link <- camera_optical 的固定旋转（REP-105）
R_CL_CO = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0]
], dtype=float)


def correct_uv_with_radial(u: float, v: float, cx: float, cy: float,
                           fx: float, fy: float,
                           img_w: int, img_h: int) -> Tuple[float, float]:
    """
    对 (u,v) 做半径相关的等向 + 非等向缩放补偿，返回校正后的 (u', v')。
    - 先将像素坐标归一化到光学平面 (x,y)，做缩放，再映回像素。
    - 等向项用 K1_ISO/K2_ISO；非等向项在判定的长边方向更强（KX1/KY1, R_ECC）。
    """
    if not ENABLE_UV_RADIAL_FIX:
        return u, v

    # 归一化
    x = (u - cx) / fx
    y = (v - cy) / fy

    # 等向半径
    r2 = x*x + y*y
    scale_iso = 1.0 + K1_ISO * r2 + K2_ISO * (r2*r2)

    # 非等向增强（沿长边方向更强）
    sx, sy = scale_iso, scale_iso
    if ENABLE_UV_ANISO:
        # 确定长边轴
        if LONG_EDGE_AXIS == 'x':
            long_x = True
        elif LONG_EDGE_AXIS == 'y':
            long_x = False
        else:  # auto
            long_x = (img_w >= img_h)

        # 定义椭圆半径：把长边方向半径拉伸 R_ECC 倍
        if long_x:
            r2_e = (x * R_ECC) ** 2 + (y ** 2)
            sx = scale_iso * (1.0 + KX1 * r2_e)
            sy = scale_iso * (1.0 + KY1 * r2_e)
        else:
            r2_e = (x ** 2) + (y * R_ECC) ** 2
            sx = scale_iso * (1.0 + KX1 * r2_e)  # 这里依然允许单独调 x/y
            sy = scale_iso * (1.0 + KY1 * r2_e)

    # 限幅
    sx = min(max(sx, UV_SCALE_MIN), UV_SCALE_MAX)
    sy = min(max(sy, UV_SCALE_MIN), UV_SCALE_MAX)

    # 应用缩放，再映回像素
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
        R_t_cam = quat_to_rot(*T_TOOL_CAM_QUAT.tolist())  # tool <- camera_(?)
        if HAND_EYE_FRAME.lower() == 'optical':
            self.R_t_co = R_t_cam
        else:  # 手眼是 link，则乘以 (link <- optical)
            self.R_t_co = R_t_cam @ R_CL_CO
        self.p_t_co = T_TOOL_CAM_XYZ  # 光学与 link 原点一致

        # 调试窗口
        self._last_debug_pub = self.get_clock().now()
        if DEBUG_WINDOW:
            try:
                cv2.namedWindow("DINO Debug", cv2.WINDOW_NORMAL)
            except Exception:
                self.get_logger().warn("无法创建可视化窗口。")

        # 最近一次有效 TF（用于重广播）
        self._last_good_tf: Optional[TransformStamped] = None
        if TF_REBROADCAST_HZ > 0:
            self.create_timer(1.0/TF_REBROADCAST_HZ, self._rebroadcast_tf)

        self._busy = False
        self._frame_count = 0

        self.get_logger().info(
            f"[seeanything_minimal_clean] topic={IMAGE_TOPIC}, prompt='{TEXT_PROMPT}', "
            f"tf_time_mode={TF_TIME_MODE}, stride={FRAME_STRIDE}, hand_eye={HAND_EYE_FRAME}; "
            f"UVfix={'on' if ENABLE_UV_RADIAL_FIX else 'off'} aniso={'on' if ENABLE_UV_ANISO else 'off'}"
        )

    # 定时重广播最近一次有效 TF（防断）
    def _rebroadcast_tf(self):
        if self._last_good_tf is None:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._last_good_tf.header.frame_id
        t.child_frame_id  = self._last_good_tf.child_frame_id
        t.transform = self._last_good_tf.transform
        self.tf_broadcaster.sendTransform(t)

    def _cb_image(self, msg: Image):
        # 丢帧限速
        self._frame_count += 1
        if FRAME_STRIDE > 1 and (self._frame_count % FRAME_STRIDE) != 0:
            return
        if self._busy:
            return
        self._busy = True
        try:
            # 1) 图像到 PIL
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil = PILImage.fromarray(rgb)

            # 2) DINO 推理：boxes, labels, (scores 可选)
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

            # 3) 选“最高置信度”一项
            s = np.array([float(s) if s is not None else -1.0 for s in scores])
            best = int(np.argmax(s))
            bx = boxes[best]
            x0, y0, x1, y1 = (bx.tolist() if hasattr(bx, 'tolist') else bx)
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)
            sc = s[best]

            # 3.5) UV 径向/非等向补偿（先校正 uv，再做后续）
            h, w = rgb.shape[:2]
            u, v = correct_uv_with_radial(u, v, CX, CY, FX, FY, w, h)

            # 4) 像素 → 光学系视线（先标准归一化）
            x_n0 = (u - CX) / FX
            y_n0 = (v - CY) / FY

            # ★ 与原脚本一致：在光学平面做 +90° 的轴旋转
            #   公式： (x, y) = ( +y0, -x0 )
            x_n =  y_n0
            y_n = -x_n0

            # 可选镜像修正
            if FLIP_X: x_n = -x_n
            if FLIP_Y: y_n = -y_n

            d_opt = np.array([x_n, y_n, 1.0], dtype=float)
            d_opt /= np.linalg.norm(d_opt)

            # 5) 取 TF（最新或图像时刻）
            t_query = rclpy.time.Time.from_msg(msg.header.stamp) if TF_TIME_MODE == 'image' else rclpy.time.Time()
            try:
                T_bt = self.tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, t_query,
                                                       timeout=RclDuration(seconds=0.2))
            except TransformException as ex:
                self.get_logger().warn(f"TF 查找失败（{TF_TIME_MODE}，base<-tool0）：{ex}")
                return
            R_bt, p_bt = tfmsg_to_Rp(T_bt)

            # 6) 相机在 base 下：T^B_C = T^B_T ∘ T^T_C(optical)
            R_bc = R_bt @ self.R_t_co
            p_bc = R_bt @ self.p_t_co + p_bt

            # 7) 射线 -> base，并与 z=Z_VIRT 求交
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

            # 7.5) 在 base_link 下做常量平移补偿（XYZ）
            C = C_raw.copy()
            if ENABLE_BASE_BIAS:
                C[0] += float(BIAS_BASE_X)
                C[1] += float(BIAS_BASE_Y)
                C[2] += float(BIAS_BASE_Z)

            # 8) 发布 TF（只这个最佳目标）
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
                f"[best only] score={(sc if sc>=0 else float('nan')):.3f} "
                f"uv_corr=({u:.1f},{v:.1f}) "
                f"C_raw=({C_raw[0]:.3f},{C_raw[1]:.3f},{C_raw[2]:.3f}) "
                f"C_corr=({C[0]:.3f},{C[1]:.3f},{C[2]:.3f}) -> {OBJECT_FRAME}"
            )

            # 9) 可选：只画“最佳”框（默认关闭）
            if DEBUG_WINDOW and DRAW_BEST_BOX:
                now = self.get_clock().now()
                if (now - self._last_debug_pub).nanoseconds >= int(1e9/DEBUG_HZ):
                    dbg = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
                    x0i, y0i, x1i, y1i = map(int, [x0, y0, x1, y1])
                    txt = f"{sc:.2f}" if sc >= 0 else ""
                    color = (0, 255, 0)
                    cv2.rectangle(dbg, (x0i, y0i), (x1i, y1i), color, 2)
                    if txt:
                        cv2.putText(dbg, txt, (x0i, max(0, y0i-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.circle(dbg, (int(round(u)), int(round(v))), 5, (255,0,0), -1)  # 画校正后的 uv
                    cv2.imshow("DINO Debug", dbg)
                    cv2.waitKey(1)
                    self._last_debug_pub = now

        except Exception as e:
            self.get_logger().error(f"处理失败：{e}")
        finally:
            self._busy = False

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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
