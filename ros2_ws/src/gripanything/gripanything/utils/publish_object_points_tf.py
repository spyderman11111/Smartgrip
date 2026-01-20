#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
import tf2_ros


def _identity_quat() -> Tuple[float, float, float, float]:
    return (0.0, 0.0, 0.0, 1.0)


def _as_xyz(p) -> List[float]:
    if not isinstance(p, (list, tuple)) or len(p) != 3:
        raise RuntimeError(f"Point must be [x,y,z], got: {type(p)} {p}")
    return [float(p[0]), float(p[1]), float(p[2])]


def _load_corners8_base(obj: Dict) -> Optional[List[List[float]]]:
    """
    Return 8 corners in base_link if available.

    Priority:
      1) object.prism.corners_8.base_link   (new method)
      2) object.obb.corners_8.base_link     (legacy OBB)
      3) object.corners_base                (very old legacy fallback)
    """
    corners = None

    # New schema: prism
    prism = obj.get("prism", None)
    if isinstance(prism, dict):
        c8 = prism.get("corners_8", None)
        if isinstance(c8, dict):
            corners = c8.get("base_link", None)

    # Legacy schema: obb
    if corners is None:
        obb = obj.get("obb", None)
        if isinstance(obb, dict):
            c8 = obb.get("corners_8", None)
            if isinstance(c8, dict):
                corners = c8.get("base_link", None)

    # Very old fallback
    if corners is None:
        corners = obj.get("corners_base", None)

    if corners is None:
        return None
    if (not isinstance(corners, list)) or len(corners) != 8:
        return None

    return [_as_xyz(p) for p in corners]


def _top_bottom_center_from_corners(corners8: List[List[float]], which: str) -> List[float]:
    """
    Compute top/bottom center from corners by z-sorting:
      - top_center: mean of 4 corners with largest z
      - bottom_center: mean of 4 corners with smallest z
    Works for both prism corners (ordered) and obb corners (unordered).
    """
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in corners8]
    pts_sorted = sorted(pts, key=lambda t: t[2])  # ascending z
    if which == "bottom":
        sel = pts_sorted[:4]
    elif which == "top":
        sel = pts_sorted[-4:]
    else:
        raise RuntimeError(f"unknown which={which}")

    cx = sum(p[0] for p in sel) / 4.0
    cy = sum(p[1] for p in sel) / 4.0
    cz = sum(p[2] for p in sel) / 4.0
    return [float(cx), float(cy), float(cz)]


def _load_points_from_object_json(path: str) -> Tuple[List[float], List[List[float]]]:
    """
    Returns:
      center_b: [x,y,z] in base_link
      corners_b: 8x[x,y,z] in base_link

    Compatible with:
      - object.center.base_link  (preferred)
      - object.center_base       (legacy)

      - object.prism.corners_8.base_link  (new method)
      - object.obb.corners_8.base_link    (legacy)
      - object.corners_base               (very old legacy)
    """
    if not path or not os.path.exists(path):
        raise RuntimeError(f"object_json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    obj = data.get("object", {})
    if not isinstance(obj, dict):
        raise RuntimeError("Invalid JSON schema: key 'object' missing or not a dict.")

    # Center
    center = None
    c = obj.get("center", None)
    if isinstance(c, dict):
        center = c.get("base_link", None)
    if center is None:
        center = obj.get("center_base", None)

    if center is None:
        raise RuntimeError("Cannot find center: object.center.base_link (or legacy center_base).")

    center_b = _as_xyz(center)

    # Corners
    corners8 = _load_corners8_base(obj)
    if corners8 is None:
        raise RuntimeError(
            "Cannot find corners_8 in json. Expect one of:\n"
            "  object.prism.corners_8.base_link (new)\n"
            "  object.obb.corners_8.base_link (legacy)\n"
            "  object.corners_base (very old)"
        )

    return center_b, corners8


class ObjectPointsTFPublisher(Node):
    def __init__(self):
        super().__init__("object_points_tf_publisher")

        # ---- parameters ----
        self.declare_parameter("object_json", "")
        self.declare_parameter("parent_frame", "base_link")
        self.declare_parameter("prefix", "object")  # frames: <prefix>_center, <prefix>_corner_0..7

        self.declare_parameter("publish_rate_hz", 10.0)

        # Optional: also publish derived top/bottom centers (from corners)
        self.declare_parameter("publish_top_bottom_centers", True)

        # Optional: watch tool0 distance to a target point
        self.declare_parameter("report_tool0", True)
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("report_rate_hz", 2.0)
        self.declare_parameter("report_target", "center")  # "center" | "corner" | "top_center" | "bottom_center"
        self.declare_parameter("corner_index", 0)          # 0..7 (used if report_target="corner")

        # Optional: reload JSON if updated
        self.declare_parameter("reload_on_change", True)

        self.parent_frame = str(self.get_parameter("parent_frame").value)
        self.prefix = str(self.get_parameter("prefix").value)
        self.object_json = str(self.get_parameter("object_json").value)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # TF listener (for tool0 distance report)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # state
        self.center_b: Optional[List[float]] = None
        self.corners_b: Optional[List[List[float]]] = None
        self.top_center_b: Optional[List[float]] = None
        self.bottom_center_b: Optional[List[float]] = None

        self._json_mtime: Optional[float] = None

        # initial load
        self._reload_json(force=True)

        # timers
        pub_rate = float(self.get_parameter("publish_rate_hz").value)
        pub_period = 1.0 / max(1e-3, pub_rate)
        self.pub_timer = self.create_timer(pub_period, self._on_publish_timer)

        if bool(self.get_parameter("report_tool0").value):
            rep_rate = float(self.get_parameter("report_rate_hz").value)
            rep_period = 1.0 / max(1e-3, rep_rate)
            self.rep_timer = self.create_timer(rep_period, self._on_report_timer)

        self.get_logger().info(
            f"Publishing TF frames under '{self.parent_frame}': "
            f"{self.prefix}_center, {self.prefix}_corner_0..7"
        )

    def _reload_json(self, force: bool = False):
        reload_on_change = bool(self.get_parameter("reload_on_change").value)
        if not force and not reload_on_change:
            return

        if not self.object_json:
            return

        try:
            mtime = os.path.getmtime(self.object_json)
        except Exception:
            mtime = None

        if (not force) and (mtime is not None) and (self._json_mtime is not None) and (mtime <= self._json_mtime):
            return

        center_b, corners_b = _load_points_from_object_json(self.object_json)
        self.center_b = center_b
        self.corners_b = corners_b
        self._json_mtime = mtime

        # Derived
        try:
            self.top_center_b = _top_bottom_center_from_corners(corners_b, which="top")
            self.bottom_center_b = _top_bottom_center_from_corners(corners_b, which="bottom")
        except Exception:
            self.top_center_b = None
            self.bottom_center_b = None

        self.get_logger().info(
            f"Loaded points: center={self.center_b}, corner0={self.corners_b[0]}"
        )

    def _make_tf(self, child_frame: str, xyz: List[float]) -> TransformStamped:
        qx, qy, qz, qw = _identity_quat()
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.parent_frame
        msg.child_frame_id = child_frame
        msg.transform.translation.x = float(xyz[0])
        msg.transform.translation.y = float(xyz[1])
        msg.transform.translation.z = float(xyz[2])
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        return msg

    def _on_publish_timer(self):
        self._reload_json(force=False)

        if self.center_b is None or self.corners_b is None:
            return

        frames: List[TransformStamped] = []
        frames.append(self._make_tf(f"{self.prefix}_center", self.center_b))

        for i in range(8):
            frames.append(self._make_tf(f"{self.prefix}_corner_{i}", self.corners_b[i]))

        if bool(self.get_parameter("publish_top_bottom_centers").value):
            if self.top_center_b is not None:
                frames.append(self._make_tf(f"{self.prefix}_top_center", self.top_center_b))
            if self.bottom_center_b is not None:
                frames.append(self._make_tf(f"{self.prefix}_bottom_center", self.bottom_center_b))

        for tfm in frames:
            self.tf_broadcaster.sendTransform(tfm)

    def _pick_report_target(self) -> Tuple[str, List[float]]:
        if self.center_b is None or self.corners_b is None:
            raise RuntimeError("points not loaded")

        mode = str(self.get_parameter("report_target").value).strip().lower()

        if mode == "center":
            return (f"{self.prefix}_center", self.center_b)

        if mode == "corner":
            idx = int(self.get_parameter("corner_index").value)
            idx = max(0, min(7, idx))
            return (f"{self.prefix}_corner_{idx}", self.corners_b[idx])

        if mode == "top_center":
            if self.top_center_b is None:
                raise RuntimeError("top_center not available")
            return (f"{self.prefix}_top_center", self.top_center_b)

        if mode == "bottom_center":
            if self.bottom_center_b is None:
                raise RuntimeError("bottom_center not available")
            return (f"{self.prefix}_bottom_center", self.bottom_center_b)

        raise RuntimeError(f"unknown report_target: {mode}")

    def _on_report_timer(self):
        if self.center_b is None or self.corners_b is None:
            return

        tool_frame = str(self.get_parameter("tool_frame").value)
        try:
            target_name, target_xyz = self._pick_report_target()
        except Exception as e:
            self.get_logger().warn(f"report target error: {e}")
            return

        # lookup parent_frame -> tool0
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.parent_frame, tool_frame, rclpy.time.Time()
            )
            tx = float(tfm.transform.translation.x)
            ty = float(tfm.transform.translation.y)
            tz = float(tfm.transform.translation.z)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed ({self.parent_frame} -> {tool_frame}): {e}")
            return

        dx = tx - float(target_xyz[0])
        dy = ty - float(target_xyz[1])
        dz = tz - float(target_xyz[2])
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        self.get_logger().info(
            f"[tool0 check] tool=({tx:.4f},{ty:.4f},{tz:.4f}) "
            f"target={target_name}=({target_xyz[0]:.4f},{target_xyz[1]:.4f},{target_xyz[2]:.4f}) "
            f"delta=({dx:.4f},{dy:.4f},{dz:.4f}) |dist|={dist:.4f} m"
        )


def main():
    rclpy.init()
    node = ObjectPointsTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
