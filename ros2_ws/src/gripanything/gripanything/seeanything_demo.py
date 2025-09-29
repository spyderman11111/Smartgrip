#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything_minimal_clean.py — GroundingDINO + 虚拟平面投影（稳健版，含XY硬补偿）

功能要点：
- 只使用“最高置信度”目标来计算并发布 TF（object_position），其余目标忽略
- 丢帧限速 + 最近一次有效 TF 周期性重广播，避免 RViz TF 断线
- 处理手眼外参坐标系差异（tool->camera_optical vs tool->camera_link）
- 在 base_link 下对求得的 3D 点做常量平移补偿（默认 X/Y 各 +0.15 m，Z 不补偿）
"""

# ====== 快速误差补偿（默认启用，X/Y 各 +0.15 m；Z 不补偿） ======
ENABLE_BASE_BIAS = True
BIAS_BASE_X = -0.05   # +X 前/右，单位米
BIAS_BASE_Y = -0.17  # +Y 左/侧，单位米
BIAS_BASE_Z = 0.00   # Z 平面误差（通常为 0）

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
FLIP_X               = False       # 若左右镜像，可把这里改为 True
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
            f"tf_time_mode={TF_TIME_MODE}, stride={FRAME_STRIDE}, hand_eye={HAND_EYE_FRAME}"
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

            # 4) 像素 → 光学系视线
            x_n0 = (u - CX) / FX
            y_n0 = (v - CY) / FY

            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★ 仅此处修改：在光学平面做 +90° 旋转以修正“上->左, 左->下, 右->上, 下->右”的错位
            # ★ 公式： (x, y) = (-y0, x0)
            # ★ 说明：这是一致的二维线性旋转，不做镜像，不改其余任何逻辑。
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            x_n =  y_n0
            y_n =  -x_n0
            # （如你开启了 FLIP_X/FLIP_Y，下两行仍然保留原行为）
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
                f"uv=({u:.1f},{v:.1f}) "
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
                    cv2.rectangle(dbg, (x0i, y0i), (x1i, y1i), 2)
                    if txt:
                        cv2.putText(dbg, txt, (x0i, max(0, y0i-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.circle(dbg, (int(round(u)), int(round(v))), 5, -1)
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
