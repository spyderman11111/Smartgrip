#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
seeanything_minimal.py — GroundingDINO + 虚拟平面投影 (object_position 版, 极简)
流程：
订阅图像 -> DINO -> 取首个框中心 (u,v)
(u,v) 在光学系反投影为单位视线 d_cam
优先用 base<-tool0<-cam（手眼外参）得到 base 下 (o, d)，否则回退 base<-camera_link
与虚拟平面 z = Z_VIRT 求交 -> C_base
发布 TF: object_position（姿态与 base 对齐）

可视化（新增）：
- 一旦检测到目标，弹出窗口展示画框与置信度；关闭脚本时窗口自动退出。
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import re
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
from PIL import Image as PILImage
import cv2

# 你项目里的 DINO 封装（保持原路径不变）
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except Exception:
    import sys
    sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
    from gripanything.core.detect_with_dino import GroundingDinoPredictor


# ====== 配置 ======
@dataclass(frozen=True)
class Config:
    # 话题
    IMAGE_TOPIC: str = '/my_camera/pylon_ros2_camera_node/image_raw'
    CAMERA_INFO_TOPIC: str = '/my_camera/pylon_ros2_camera_node/camera_info'

    # 坐标系
    BASE_FRAME: str = 'base_link'
    CAMERA_LINK_FRAME: str = 'camera_link'
    OBJECT_FRAME: str = 'object_position'

    # 虚拟平面高度（在 base_link 下）
    Z_VIRT: float = 0.0

    # GroundingDINO
    TEXT_PROMPT: str = 'yellow object .'
    DINO_MODEL_ID: str = 'IDEA-Research/grounding-dino-tiny'
    DINO_DEVICE: str = 'cuda'
    BOX_THRESHOLD: float = 0.25
    TEXT_THRESHOLD: float = 0.25

    # 相机内参
    USE_CAMERA_INFO: bool = True
    FX: float = 2674.3803723910564
    FY: float = 2667.4211254043507
    CX: float = 954.5922081613583
    CY: float = 1074.965947832258

    # camera_link 是否已经是“光学系”
    CAMERA_LINK_IS_OPTICAL: bool = False
    USE_LATEST_TF_ON_FAIL: bool = True

    # 手眼外参与 tool 链（推荐）：base <- tool0 <- cam
    USE_TOOL_EXTRINSIC: bool = True
    TOOL_FRAME: str = 'tool0'
    # 把以下两项替换为你的手眼标定（tool->cam）
    T_TOOL_CAM_XYZ: tuple = (-0.000006852374024, -0.099182661943126947, 0.02391824813032688)
    T_TOOL_CAM_QUAT: tuple = (-0.0036165657530785695, -0.000780788838366878,
                              0.7078681983794892, 0.7063348529868249)  # qx,qy,qz,qw

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

def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def R_link_to_optical() -> np.ndarray:
    """
    固定旋转：camera_link -> camera_optical
    约定：link: X前/Y左/Z上；optical: Z前/X右/Y下
    返回矩阵把 "optical向量" 表达到 "link坐标"：v_link = R * v_opt
    """
    return np.array([[ 0,  0, 1],
                     [-1,  0, 0],
                     [ 0, -1, 0]], dtype=float)

def parse_conf(text: str) -> Optional[float]:
    """从标签字符串里尽力解析置信度（若无则返回 None）"""
    if not isinstance(text, str):
        return None
    # 取最后一个类似 0.87 / 87% 的数字
    m = re.findall(r'(\d+(?:\.\d+)?)%?', text)
    if not m:
        return None
    try:
        val = float(m[-1])
        return val/100.0 if '%' in text else val
    except Exception:
        return None


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
        self.declare_parameter('z_virt', CFG.Z_VIRT)

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

        # 手眼外参 + tool 链
        self.declare_parameter('use_tool_extrinsic', CFG.USE_TOOL_EXTRINSIC)
        self.declare_parameter('tool_frame', CFG.TOOL_FRAME)
        self.declare_parameter('t_tool_cam_xyz', list(CFG.T_TOOL_CAM_XYZ))
        self.declare_parameter('t_tool_cam_quat', list(CFG.T_TOOL_CAM_QUAT))

        self.declare_parameter('camera_link_is_optical', CFG.CAMERA_LINK_IS_OPTICAL)
        self.declare_parameter('use_latest_tf_on_fail', CFG.USE_LATEST_TF_ON_FAIL)

        # 读取参数
        g = self.get_parameter
        self.image_topic = g('image_topic').value
        self.ci_topic = g('camera_info_topic').value

        self.base_frame = g('base_frame').value
        self.camera_link_frame = g('camera_link_frame').value
        self.object_frame = g('object_frame').value
        self.z_virt = float(g('z_virt').value)

        self.text_prompt = g('text_prompt').value
        self.model_id = g('model_id').value
        self.device = g('device').value
        self.box_th = float(g('box_threshold').value)
        self.text_th = float(g('text_threshold').value)

        self.use_ci = bool(g('use_camera_info').value)
        self.fx = float(g('fx').value)
        self.fy = float(g('fy').value)
        self.cx = float(g('cx').value)
        self.cy = float(g('cy').value)

        # 手眼外参参数
        self.use_tool_extrinsic = g('use_tool_extrinsic').value
        self.tool_frame = g('tool_frame').value
        self.t_tool_cam_xyz = np.array(g('t_tool_cam_xyz').value, dtype=float)
        self.t_tool_cam_quat = np.array(g('t_tool_cam_quat').value, dtype=float)

        self.camera_link_is_optical = bool(g('camera_link_is_optical').value)
        self.use_latest_tf_on_fail = bool(g('use_latest_tf_on_fail').value)

        # TF 与 DINO
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.predictor = GroundingDinoPredictor(self.model_id, self.device)

        # 订阅
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()
        if self.use_ci:
            self.sub_ci = self.create_subscription(CameraInfo, self.ci_topic, self._cb_ci, qos)
        self.sub_img = self.create_subscription(Image, self.image_topic, self._cb_image, qos)

        self.get_logger().info(
            f"[seeanything_minimal] image_topic={self.image_topic}, "
            f"use_tool_extrinsic={self.use_tool_extrinsic}, tool_frame={self.tool_frame}"
        )

        # 内参缓存
        self._have_K = False
        self._busy = False
        self.D = None
        self.dist_model: Optional[str] = None

        # 可视化窗口状态
        self._win_name = "DINO Debug"
        self._win_created = False

    def _ensure_window(self):
        if not self._win_created:
            try:
                cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
                self._win_created = True
            except Exception as e:
                self.get_logger().warn(f"创建显示窗口失败：{e}")

    def _close_window(self):
        if self._win_created:
            try:
                cv2.destroyWindow(self._win_name)
            except Exception:
                pass
            self._win_created = False

    # CameraInfo 回调
    def _cb_ci(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]; self.cx = msg.k[2]; self.cy = msg.k[5]
        self.D = np.array(msg.d, dtype=float) if msg.d else None
        self.dist_model = msg.distortion_model
        self._have_K = True

    def _cb_image(self, msg: Image):
        if self._busy:
            return
        if not self._have_K and self.use_ci:
            self.get_logger().warn('等待 /camera_info 提供内参/畸变。')
            return

        self._busy = True
        try:
            # 1) 图像 -> PIL
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil = PILImage.fromarray(rgb)

            # 2) DINO 预测（兼容两种返回：boxes,labels 或 boxes,scores,labels）
            out = self.predictor.predict(
                pil, self.text_prompt, box_threshold=self.box_th, text_threshold=self.text_th
            )
            if isinstance(out, tuple) and len(out) == 3:
                boxes, scores, labels = out
            elif isinstance(out, tuple) and len(out) == 2:
                boxes, labels = out
                scores = [None] * len(boxes)
            else:
                # 最保守兼容：假设是 list[dict]
                try:
                    boxes = [o['box'] for o in out]
                    scores = [o.get('score') for o in out]
                    labels = [o.get('label', '') for o in out]
                except Exception:
                    self.get_logger().warn('DINO 返回格式不支持。')
                    return

            if len(boxes) == 0:
                self.get_logger().info('未检测到目标。')
                # 若之前有窗口也继续显示最近一帧；这里不强制关闭
                return

            # 3) 画框调试图（BGR）
            dbg = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
            H, W = dbg.shape[:2]
            u = v = None
            # 画全部框
            for i, b in enumerate(boxes):
                x0, y0, x1, y1 = map(int, b.tolist() if hasattr(b, 'tolist') else list(b))
                conf = scores[i]
                if conf is None and i < len(labels):
                    conf = parse_conf(labels[i])
                tag = labels[i] if i < len(labels) else ''
                if conf is not None:
                    tag = f"{tag} {conf:.2f}"
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(dbg, tag, (x0, max(0, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 4) 以第一个框为目标（保持原逻辑）
            x0, y0, x1, y1 = boxes[0].tolist() if hasattr(boxes[0], 'tolist') else boxes[0]
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)
            cv2.circle(dbg, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)

            # 5) (u,v) -> 光学系单位视线 d_cam_opt
            if self.D is not None and self.dist_model in (None, '', 'plumb_bob', 'rational_polynomial'):
                pts = np.array([[[u, v]]], dtype=np.float32)
                K = np.array([[self.fx, 0, self.cx],
                              [0, self.fy, self.cy],
                              [0, 0, 1]], dtype=float)
                undist = cv2.undistortPoints(pts, K, self.D, P=None)
                x_n, y_n = float(undist[0, 0, 0]), float(undist[0, 0, 1])
            else:
                x_n = (u - self.cx) / self.fx
                y_n = (v - self.cy) / self.fy
            d_cam_opt = np.array([x_n, y_n, 1.0], dtype=float)
            d_cam_opt /= np.linalg.norm(d_cam_opt)

            # 6) 相机位姿：优先走 base <- tool0 <- cam（手眼外参）
            t_img = rclpy.time.Time.from_msg(msg.header.stamp)
            if self.use_tool_extrinsic:
                try:
                    Tbt = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, t_img,
                                                          timeout=RclDuration(seconds=0.2))
                except TransformException:
                    if not self.use_latest_tf_on_fail:
                        self.get_logger().warn("TF 查找失败（按图像时刻，base<-tool0），丢弃该帧。")
                        return
                    Tbt = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame, rclpy.time.Time())
                Rbt, pbt = tfmsg_to_Rp(Tbt)

                qx, qy, qz, qw = self.t_tool_cam_quat.tolist()
                Rtc = quat_to_rot(qx, qy, qz, qw)
                ptc = self.t_tool_cam_xyz

                Rbc = Rbt @ Rtc
                pbc = Rbt @ ptc + pbt

                d_base = Rbc @ d_cam_opt
                d_base /= np.linalg.norm(d_base)
                o_base = pbc
            else:
                try:
                    Tmsg = self.tf_buffer.lookup_transform(
                        self.base_frame, self.camera_link_frame, t_img, timeout=RclDuration(seconds=0.2)
                    )
                except TransformException:
                    if not self.use_latest_tf_on_fail:
                        self.get_logger().warn("TF 查找失败（按图像时刻，base<-camera_link），丢弃该帧。")
                        return
                    Tmsg = self.tf_buffer.lookup_transform(self.base_frame, self.camera_link_frame, rclpy.time.Time())

                R_base_clink, p_base_clink = tfmsg_to_Rp(Tmsg)
                if self.camera_link_is_optical:
                    d_cam_link = d_cam_opt
                else:
                    d_cam_link = R_link_to_optical() @ d_cam_opt
                d_base = R_base_clink @ d_cam_link
                d_base /= np.linalg.norm(d_base)
                o_base = p_base_clink

            # 7) 与 z=Z_VIRT 求交
            rz = float(d_base[2])
            if abs(rz) < 1e-6:
                self.get_logger().warn('视线近水平（|d_z|≈0），无法与虚拟平面求交。')
                # 仍展示可视化
            else:
                t_star = (self.z_virt - float(o_base[2])) / rz
                if t_star >= 0:
                    C_base = o_base + t_star * d_base  # 交点（object_position）

                    # 发布 object_position TF（姿态与 base 对齐）
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

                    self.get_logger().info(
                        f"uv=({u:.1f},{v:.1f})  object=({C_base[0]:.3f},{C_base[1]:.3f},{C_base[2]:.3f})"
                    )

            # 8) 弹窗显示
            self._ensure_window()
            if self._win_created:
                cv2.imshow(self._win_name, dbg)
                # 不阻塞：1 ms
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f"处理失败：{e}")
        finally:
            self._busy = False

    def destroy_node(self):
        # 关闭窗口
        try:
            self._close_window()
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
        # 确保弹窗退出
        try:
            node.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == '__main__':
    main()
