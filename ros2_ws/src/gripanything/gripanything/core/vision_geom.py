"""
vision_geom.py — Single-shot detection + ray casting to z=Z_VIRT to get circle center.

Purpose:
- Run GroundingDINO once to detect the highest-score box for the given prompt.
- Convert pixel center to optical-ray, then to base frame using tool->camera extrinsics.
- Intersect the ray with a virtual plane z=Z_VIRT to get C, add bias, and build:
  (1) a hover pose over C with tool-Z-down, (2) object and circle TFs for visualization.
"""

from typing import Optional, Tuple
import numpy as np
from rclpy.duration import Duration as RclDuration
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import tf2_ros
from tf2_ros import TransformException

from .tf_ops import tfmsg_to_Rp, quat_to_rot, R_CL_CO, pose_from_pq

def _safe_float(x):
    try:
        import torch
        if hasattr(x, 'detach'):
            return float(x.detach().cpu().item())
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return -1.0

class SingleShotDetector:
    """Wrap one-shot detection and geometry for center C and hover pose."""
    def __init__(self, node, cfg, predictor):
        self._node = node
        self._cfg = cfg
        self._pred = predictor
        self._bridge = CvBridge()
        # precompute tool <- camera_optical
        qx, qy, qz, qw = cfg.cam.t_tool_cam_quat_xyzw
        R_t_cam = quat_to_rot(qx, qy, qz, qw)
        self.R_t_co = R_t_cam if cfg.cam.hand_eye_frame.lower() == 'optical' else (R_t_cam @ R_CL_CO)
        self.p_t_co = np.array(cfg.cam.t_tool_cam_xyz, dtype=float)

    def _pixel_to_dir_optical(self, u, v) -> np.ndarray:
        x_n0 = (u - self._cfg.cam.cx) / self._cfg.cam.fx
        y_n0 = (v - self._cfg.cam.cy) / self._cfg.cam.fy
        # optical-frame normalization (REP-105)
        x_n =  y_n0
        y_n = -x_n0
        if self._cfg.cam.flip_x: x_n = -x_n
        if self._cfg.cam.flip_y: y_n = -y_n
        d_opt = np.array([x_n, y_n, 1.0], dtype=float)
        return d_opt / (np.linalg.norm(d_opt) + 1e-12)

    def detect_once(self, img_msg: Image, tf_buffer: tf2_ros.Buffer) -> Optional[Tuple[np.ndarray, PoseStamped, TransformStamped, TransformStamped]]:
        """Return (C, hover_pose, tf_object, tf_circle) or None."""
        rgb = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        pil = PILImage.fromarray(rgb)

        out = self._pred.predict(
            pil, self._cfg.dino.text_prompt,
            box_threshold=self._cfg.dino.box_threshold,
            text_threshold=self._cfg.dino.text_threshold
        )
        if not isinstance(out, tuple) or len(out) < 2:
            self._node.get_logger().warn("Unsupported DINO return format.")
            return None
        boxes, labels = out[:2]
        scores = out[2] if len(out) >= 3 else [None] * len(boxes)
        if len(boxes) == 0:
            self._node.get_logger().info("No detections.")
            return None

        s = np.array([_safe_float(z) for z in scores], dtype=float)
        best = int(np.argmax(s))
        x0, y0, x1, y1 = (boxes[best].tolist() if hasattr(boxes[best], 'tolist') else boxes[best])
        u = 0.5 * (x0 + x1)
        v = 0.5 * (y0 + y1)

        # pixel -> optical ray
        d_opt = self._pixel_to_dir_optical(u, v)

        # TF(base<-tool)
        t_query = Time.from_msg(img_msg.header.stamp) if self._cfg.control.tf_time_mode == 'image' else Time()
        try:
            T_bt = tf_buffer.lookup_transform(self._cfg.frames.base_frame, self._cfg.frames.tool_frame, t_query,
                                              timeout=RclDuration(seconds=0.2))
        except TransformException as ex:
            self._node.get_logger().warn(f"TF lookup failed: {ex}")
            return None
        R_bt, p_bt = tfmsg_to_Rp(T_bt)

        # camera in base (optical)
        R_bc = R_bt @ self.R_t_co
        p_bc = R_bt @ self.p_t_co + p_bt

        # intersect with z = Z_VIRT
        d_base = R_bc @ d_opt
        dz = float(d_base[2])
        if abs(dz) < 1e-6:
            self._node.get_logger().warn("Ray nearly parallel to plane.")
            return None
        t_star = (self._cfg.frames.z_virt - float(p_bc[2])) / dz
        if t_star < 0:
            self._node.get_logger().warn("Intersection behind the camera.")
            return None

        C = p_bc + t_star * d_base
        if self._cfg.bias.enable:
            C[0] += self._cfg.bias.bx
            C[1] += self._cfg.bias.by
            C[2] += self._cfg.bias.bz

        # hover pose (tool-Z-down)
        hover = pose_from_pq(
            [float(C[0]), float(C[1]), float(C[2] + self._cfg.control.hover_above)],
            [0.0, 1.0, 0.0, 0.0],  # (w, x, y, z)=(0,1,0,0) = Rx(pi)
            self._cfg.frames.pose_frame
        )
        hover.header.stamp = self._node.get_clock().now().to_msg()

        # TFs
        tf_now = self._node.get_clock().now().to_msg()
        tf_obj = TransformStamped()
        tf_obj.header.stamp = tf_now
        tf_obj.header.frame_id = self._cfg.frames.base_frame
        tf_obj.child_frame_id  = self._cfg.frames.object_frame
        tf_obj.transform.translation.x, tf_obj.transform.translation.y, tf_obj.transform.translation.z = float(C[0]), float(C[1]), float(C[2])
        tf_obj.transform.rotation.w = 1.0

        tf_circle = TransformStamped()
        tf_circle.header.stamp = tf_now
        tf_circle.header.frame_id = self._cfg.frames.base_frame
        tf_circle.child_frame_id  = self._cfg.frames.circle_frame
        tf_circle.transform.translation.x, tf_circle.transform.translation.y, tf_circle.transform.translation.z = float(C[0]), float(C[1]), float(C[2])
        tf_circle.transform.rotation.w = 1.0

        self._node.get_logger().info(f"[detect once] C=({C[0]:.3f},{C[1]:.3f},{C[2]:.3f}), hover_z={hover.pose.position.z:.3f}")
        return C, hover, tf_obj, tf_circle
