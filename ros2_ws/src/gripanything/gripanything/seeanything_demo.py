#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything_minimal.py — GroundingDINO + 虚拟平面投影 (object_position 版, 极简)
流程：
订阅图像 -> DINO -> 取首个框中心 (u,v)
(u,v) 在光学系反投影为单位视线 d_cam
用 TF(base_link <- camera_link) + 固定旋转(link->optical) 得到 base 下 (o, d)
与虚拟平面 z = Z_VIRT 求交 -> C_base
发布 TF: object_position（姿态与 base 对齐）
可选：发布 TF: object_hover = C_base + [0,0,HOVER_ABOVE]
日志仅打印：uv 与 object_position 坐标
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration as RclDuration

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf2_ros import TransformException

from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage

# ====== GroundingDINO（保持你的调用方式）======
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except ImportError:
    import sys
    sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
    from gripanything.core.detect_with_dino import GroundingDinoPredictor


# ====== 参数（集中放在顶部）======
@dataclass(frozen=True)
class Config:
    # 话题
    IMAGE_TOPIC: str = '/my_camera/pylon_ros2_camera_node/image_raw'
    CAMERA_INFO_TOPIC: str = '/my_camera/pylon_ros2_camera_node/camera_info'

    # 坐标系
    BASE_FRAME: str = 'base_link'
    CAMERA_LINK_FRAME: str = 'camera_link'
    OBJECT_FRAME: str = 'object_position'
    HOVER_FRAME: str = 'object_hover'     # 可选发布：物体上方 HOVER_ABOVE 的点

    # 虚拟平面高度（在 base_link 下）
    Z_VIRT: float = 0.0

    # 机械臂最终落点相对物体上方的高度（米）
    HOVER_ABOVE: float = 0.50
    PUBLISH_HOVER_TF: bool = True

    # DINO
    TEXT_PROMPT: str = 'yellow object .'
    DINO_MODEL_ID: str = 'IDEA-Research/grounding-dino-tiny'
    DINO_DEVICE: str = 'cuda'
    BOX_THRESHOLD: float = 0.20
    TEXT_THRESHOLD: float = 0.20

    # 相机内参
    USE_CAMERA_INFO: bool = True
    FX: float = 2674.3803723910564
    FY: float = 2667.4211254043507
    CX: float = 954.5922081613583
    CY: float = 1074.965947832258

    # 你的 TF 与光学系对齐设置
    CAMERA_LINK_IS_OPTICAL: bool = False  # 若 camera_link 本身已是光学系，则设 True 跳过固定旋转
    USE_LATEST_TF_ON_FAIL: bool = True    # 查不到图像时刻的 TF 是否回退到最新 Time(0)

CFG = Config()


# ====== 小工具 ======
def tfmsg_to_Rp(transform: TransformStamped) -> Tuple[np.ndarray, np.ndarray]:
    q = transform.transform.rotation
    t = transform.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)
    p = np.array([t.x, t.y, t.z], dtype=float)
    return R, p


def R_link_to_optical() -> np.ndarray:
    """
    固定旋转：camera_link -> camera_optical
    约定：link: X前/Y左/Z上；optical: Z前/X右/Y下
    X_opt = -Y_link, Y_opt = -Z_link, Z_opt = +X_link
    返回矩阵把 "optical向量" 表达到 "link坐标"：v_link = R * v_opt
    """
    return np.array([[ 0,  0, 1],
                     [-1,  0, 0],
                     [ 0, -1, 0]], dtype=float)


# ====== 主节点 ======
class SeeAnythingMinimal(Node):
    def __init__(self):
        super().__init__('seeanything_minimal')

        # 声明参数
        self.declare_parameter('image_topic', CFG.IMAGE_TOPIC)
        self.declare_parameter('camera_info_topic', CFG.CAMERA_INFO_TOPIC)

        self.declare_parameter('base_frame', CFG.BASE_FRAME)
        self.declare_parameter('camera_link_frame', CFG.CAMERA_LINK_FRAME)
        self.declare_parameter('object_frame', CFG.OBJECT_FRAME)
        self.declare_parameter('hover_frame', CFG.HOVER_FRAME)

        self.declare_parameter('z_virt', CFG.Z_VIRT)
        self.declare_parameter('hover_above', CFG.HOVER_ABOVE)
        self.declare_parameter('publish_hover_tf', CFG.PUBLISH_HOVER_TF)

        self.declare_parameter('text_prompt', CFG.TEXT_PROMPT)
        self.declare_parameter('model_id', CFG.DINO_MODEL_ID)
        self.declare_parameter('device', CFG.DINO_DEVICE)
        self.declare_parameter('box_threshold', CFG.BOX_THRESHOLD)
        self.declare_parameter('text_threshold', CFG.TEXT_THRESHOLD)

        self.declare_parameter('use_camera_info', CFG.USE_CAMERA_INFO)
        self.declare_parameter('fx', CFG.FX)
        self.declare_parameter('fy', CFG.FY)
        self.declare_parameter('cx', CFG.CX)
        self.declare_parameter('cy', CFG.CY)

        self.declare_parameter('camera_link_is_optical', CFG.CAMERA_LINK_IS_OPTICAL)
        self.declare_parameter('use_latest_tf_on_fail', CFG.USE_LATEST_TF_ON_FAIL)

        # 读取参数
        g = self.get_parameter
        self.image_topic = g('image_topic').value
        self.ci_topic = g('camera_info_topic').value

        self.base_frame = g('base_frame').value
        self.camera_link_frame = g('camera_link_frame').value
        self.object_frame = g('object_frame').value
        self.hover_frame = g('hover_frame').value

        self.z_virt = float(g('z_virt').value)
        self.hover_above = float(g('hover_above').value)
        self.publish_hover_tf = bool(g('publish_hover_tf').value)

        self.text_prompt = g('text_prompt').value
        self.model_id = g('model_id').value
        self.device = g('device').value
        self.box_th = float(g('box_threshold').value)
        self.text_th = float(g('text_threshold').value)

        self.use_ci = bool(g('use_camera_info').value)
        self.fx = float(g('fx').value); self.fy = float(g('fy').value)
        self.cx = float(g('cx').value); self.cy = float(g('cy').value)

        self.camera_link_is_optical = bool(g('camera_link_is_optical').value)
        self.use_latest_tf_on_fail = bool(g('use_latest_tf_on_fail').value)

        # 内参缓存
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0,       0,      1]], dtype=float)
        self.D: Optional[np.ndarray] = None
        self.dist_model: Optional[str] = None
        self._have_K = not self.use_ci

        # TF 与 DINO
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.predictor = GroundingDinoPredictor(self.model_id, self.device)

        # 订阅
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        if self.use_ci:
            self.sub_ci = self.create_subscription(CameraInfo, self.ci_topic, self._cb_ci, qos)
        self.sub_img = self.create_subscription(Image, self.image_topic, self._cb_image, qos)

        # 启动时简短打印
        self.get_logger().info(
            f"[seeanything_minimal] image_topic={self.image_topic}, "
            f"frames: base='{self.base_frame}', camera_link='{self.camera_link_frame}', "
            f"object='{self.object_frame}', hover='{self.hover_frame}', "
            f"Z_VIRT={self.z_virt:.3f}, HOVER_ABOVE={self.hover_above:.3f}, "
            f"camera_link_is_optical={self.camera_link_is_optical}"
        )

        self._busy = False

    def _cb_ci(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.K = K.copy()
        self.fx, self.fy, self.cx, self.cy = K[0,0], K[1,1], K[0,2], K[1,2]
        self.D = np.array(msg.d, dtype=float).reshape(-1) if (msg.d is not None and len(msg.d) > 0) else None
        self.dist_model = getattr(msg, 'distortion_model', None) or None
        self._have_K = True

    def _cb_image(self, msg: Image):
        if self._busy:
            return
        if not self._have_K:
            self.get_logger().warn('等待 /camera_info 提供内参/畸变。')
            return

        self._busy = True
        try:
            # 1) 图像 -> PIL
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil = PILImage.fromarray(rgb)

            # 2) DINO 预测
            boxes, labels = self.predictor.predict(
                pil, self.text_prompt, box_threshold=self.box_th, text_threshold=self.text_th
            )
            n = 0 if boxes is None else len(boxes)
            prompt_text = (self.text_prompt or "").strip().rstrip(" .,:;!?，。；？！")
            self.get_logger().info(f"检测出{n}个 {prompt_text}")
            if n == 0:
                return

            # 取首个框中心 (u,v)
            x0, y0, x1, y1 = boxes[0].tolist()
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)

            # 3) (u,v) -> 光学系单位视线 d_cam_opt
            if self.D is not None and self.dist_model in (None, '', 'plumb_bob', 'rational_polynomial'):
                pts = np.array([[[u, v]]], dtype=np.float32)
                undist = cv2.undistortPoints(pts, self.K, self.D, P=None)
                x_n, y_n = float(undist[0, 0, 0]), float(undist[0, 0, 1])
            else:
                x_n = (u - self.cx) / self.fx
                y_n = (v - self.cy) / self.fy
            d_cam_opt = np.array([x_n, y_n, 1.0], dtype=float)
            d_cam_opt /= np.linalg.norm(d_cam_opt)

            # 4) TF: base <- camera_link（按图像时间戳，必要时回退）
            t_img = rclpy.time.Time.from_msg(msg.header.stamp)
            try:
                Tmsg = self.tf_buffer.lookup_transform(
                    self.base_frame, self.camera_link_frame, t_img, timeout=RclDuration(seconds=0.2)
                )
            except TransformException:
                if not self.use_latest_tf_on_fail:
                    self.get_logger().warn("TF 查找失败（按图像时刻），丢弃该帧。")
                    return
                Tmsg = self.tf_buffer.lookup_transform(
                    self.base_frame, self.camera_link_frame, rclpy.time.Time()
                )

            R_base_clink, p_base_clink = tfmsg_to_Rp(Tmsg)

            # 5) 视线到 base
            if self.camera_link_is_optical:
                d_cam_link = d_cam_opt
            else:
                d_cam_link = R_link_to_optical() @ d_cam_opt   # v_link = R_link_opt * v_opt
            d_base = R_base_clink @ d_cam_link
            d_base /= np.linalg.norm(d_base)
            o_base = p_base_clink

            # 6) 与 z=Z_VIRT 求交
            rz = float(d_base[2])
            if abs(rz) < 1e-6:
                self.get_logger().warn('视线近水平（|d_z|≈0），无法与虚拟平面求交。')
                return
            t_star = (self.z_virt - float(o_base[2])) / rz
            if t_star < 0:
                self.get_logger().warn('交点在相机后方（t<0），忽略。')
                return
            C_base = o_base + t_star * d_base  # 交点（object_position）

            # 7) 发布 object_position TF（姿态与 base 对齐）
            now = self.get_clock().now().to_msg()
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self.base_frame
            t.child_frame_id = self.object_frame
            t.transform.translation.x = float(C_base[0])
            t.transform.translation.y = float(C_base[1])
            t.transform.translation.z = float(C_base[2])
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)

            # 可选：发布 object_hover（物体上方 HOVER_ABOVE）
            if self.publish_hover_tf and self.hover_above > 0.0:
                h = TransformStamped()
                h.header.stamp = now
                h.header.frame_id = self.base_frame
                h.child_frame_id = self.hover_frame
                h.transform.translation.x = float(C_base[0])
                h.transform.translation.y = float(C_base[1])
                h.transform.translation.z = float(C_base[2] + self.hover_above)
                h.transform.rotation.x = 0.0
                h.transform.rotation.y = 0.0
                h.transform.rotation.z = 0.0
                h.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(h)

            # 8) 最简日志：只打印 uv 与 object_position
            self.get_logger().info(
                f"uv=({u:.1f},{v:.1f})  object=({C_base[0]:.3f},{C_base[1]:.3f},{C_base[2]:.3f})"
            )

        except Exception as e:
            self.get_logger().warn(f"处理失败：{e}")
        finally:
            self._busy = False


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
