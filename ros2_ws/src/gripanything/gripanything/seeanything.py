#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seeanything.py — Main ROS 2 node (thin controller) for:
- INIT pose -> single GroundingDINO detection (interactive prompt) ->
- hover above detected center -> N-vertex circular path via IK (non-closed) ->
- return to INIT and exit.

Key additions in this revision:
- Interactive text prompt at startup (terminal input) so the user specifies the target.
  You can disable interactivity by passing: --ros-args -p require_prompt:=false -p text_prompt:="your target"
- Clean separation of concerns via core/ modules.
"""

from typing import Optional, List
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration as RclDuration
from rclpy.time import Time
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped

import tf2_ros

# Your existing DINO wrapper
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except Exception:
    import sys
    sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
    from gripanything.core.detect_with_dino import GroundingDinoPredictor

from gripanything.core.config import load_from_ros_params
from gripanything.core.vision_geom import SingleShotDetector
from gripanything.core.ik_and_traj import MotionContext, TrajectoryPublisher, IKClient
from gripanything.core.polygon_path import make_polygon_vertices
from gripanything.core.tf_ops import tfmsg_to_Rp

class SeeAnythingNode(Node):
    def __init__(self):
        super().__init__('seeanything_minimal_clean')

        # Load config (with some ROS param overrides)
        self.cfg = load_from_ros_params(self)

        # Interactive prompt (unless disabled)
        if self.cfg.runtime.require_prompt:
            user_prompt = input("Enter the target text prompt (e.g., 'orange object'): ").strip()
            if user_prompt:
                # update both config and the ROS parameter value
                self.cfg.dino.text_prompt = user_prompt
                self.set_parameters([Parameter('text_prompt', value=user_prompt)])
                self.get_logger().info(f"Using user prompt: \"{user_prompt}\"")
            else:
                self.get_logger().warn(f"No input given; using default prompt: \"{self.cfg.dino.text_prompt}\"")
        else:
            self.get_logger().info(f"Interactive prompt disabled. Using: \"{self.cfg.dino.text_prompt}\"")

        # Image subscription (trigger only)
        qos_img = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, self.cfg.frames.image_topic, self._on_image, qos_img)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_brd = tf2_ros.TransformBroadcaster(self)

        # DINO + detector
        self.predictor = GroundingDinoPredictor(self.cfg.dino.model_id, self.cfg.dino.device)
        self.detector = SingleShotDetector(self, self.cfg, self.predictor)

        # Trajectory / IK / joint_states
        self.motion = MotionContext(self, self.cfg.control.joint_order, self.cfg.control.vel_eps, self.cfg.control.require_stationary)
        self.traj   = TrajectoryPublisher(self, self.cfg.control.joint_order, self.cfg.control.controller_topic)
        self.ik     = IKClient(self, self.cfg.control.group_name, self.cfg.control.ik_link_name,
                               self.cfg.control.joint_order, self.cfg.control.ik_timeout,
                               self.cfg.jump.ignore_joints, self.cfg.jump.max_safe_jump, self.cfg.jump.max_warn_jump)

        # State
        self._phase = 'init_needed'
        self._inflight = False
        self._done = False
        self._fixed_hover: Optional[PoseStamped] = None
        self._circle_center: Optional[np.ndarray] = None
        self._ring_z: Optional[float] = None
        self._start_yaw: Optional[float] = None
        self._poly_wps: List[PoseStamped] = []
        self._poly_idx: int = 0
        self._poly_dwell_due_ns: Optional[int] = None
        self._last_obj_tf: Optional[TransformStamped] = None
        self._last_circle_tf: Optional[TransformStamped] = None
        self._frame_count = 0

        # TF rebroadcast
        if self.cfg.control.tf_rebroadcast_hz > 0:
            self.create_timer(1.0 / self.cfg.control.tf_rebroadcast_hz, self._rebroadcast_tfs)

        # Main loop
        self.create_timer(0.05, self._tick)

        self.get_logger().info(
            f"[seeanything] topic={self.cfg.frames.image_topic}, hover={self.cfg.control.hover_above:.3f}m, "
            f"bias=({self.cfg.bias.bx:.3f},{self.cfg.bias.by:.3f},{self.cfg.bias.bz:.3f}); "
            f"N={self.cfg.circle.n_vertices}, R={self.cfg.circle.radius:.3f}m, orient={self.cfg.circle.orient_mode}, dir={self.cfg.circle.poly_dir}"
        )

    # ---------- TF rebroadcast ----------
    def _rebroadcast_tfs(self):
        now = self.get_clock().now().to_msg()
        if self._last_obj_tf is not None:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self._last_obj_tf.header.frame_id
            t.child_frame_id  = self._last_obj_tf.child_frame_id
            t.transform = self._last_obj_tf.transform
            self.tf_brd.sendTransform(t)
        if self._last_circle_tf is not None:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self._last_circle_tf.header.frame_id
            t.child_frame_id  = self._last_circle_tf.child_frame_id
            t.transform = self._last_circle_tf.transform
            self.tf_brd.sendTransform(t)

    # ---------- current tool yaw (XY) ----------
    def _get_tool_yaw_xy(self) -> Optional[float]:
        try:
            T_bt = self.tf_buffer.lookup_transform(self.cfg.frames.base_frame, self.cfg.frames.tool_frame, Time(),
                                                   timeout=RclDuration(seconds=0.2))
            R_bt, _ = tfmsg_to_Rp(T_bt)
            ex = R_bt[:, 0]
            return float(math.atan2(float(ex[1]), float(ex[0])))
        except Exception as ex:
            self.get_logger().warn(f"Read tool yaw failed: {ex}")
            return None

    # ---------- FSM ----------
    def _tick(self):
        if self._done or self._inflight:
            return

        if self._phase == 'init_needed':
            self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
            self.motion.set_seed_hint(self.cfg.control.init_pos)
            self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
            self._phase = 'init_moving'
            return

        if self._phase == 'init_moving':
            if self.motion.is_stationary():
                self._phase = 'wait_detect'
                self.get_logger().info("INIT reached. Waiting for detection…")
            return

        if self._phase == 'wait_detect' and self._fixed_hover is not None:
            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                self.get_logger().warn("Waiting for /joint_states …")
                return
            self._inflight = True
            ref = (self.motion._last_js.position if self.motion._last_js else self.cfg.control.init_pos)
            self.ik.request_async(self._fixed_hover, seed, list(ref), self._on_hover_ik)
            return

        if self._phase == 'hover_to_center':
            if not self.motion.is_stationary():
                return
            self._start_yaw = self._get_tool_yaw_xy()
            if self._circle_center is not None and self._ring_z is not None and self._start_yaw is not None:
                self._poly_wps = make_polygon_vertices(
                    self.get_clock().now().to_msg,
                    self._circle_center, self._ring_z, self._start_yaw,
                    self.cfg.frames.pose_frame,
                    self.cfg.circle.n_vertices, self.cfg.circle.num_turns, self.cfg.circle.poly_dir,
                    self.cfg.circle.orient_mode, self.cfg.circle.start_dir_offset_deg,
                    self.cfg.circle.radius, self.cfg.circle.tool_z_sign
                )
                self._poly_idx = 0
                self.get_logger().info(f"Generated vertices: {len(self._poly_wps)} (start_yaw={self._start_yaw:.3f} rad).")
                self._phase = 'poly_prepare'
            else:
                self._phase = 'return_init'
            return

        if self._phase == 'poly_prepare':
            if not self._poly_wps:
                self._phase = 'return_init'; return
            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None: return
            self._inflight = True
            ref = (self.motion._last_js.position if self.motion._last_js else self.cfg.control.init_pos)
            self.ik.request_async(self._poly_wps[0], seed, list(ref), self._on_poly_ik)
            self._poly_idx = 1
            self._phase = 'poly_moving'
            return

        if self._phase == 'poly_moving':
            now = self.get_clock().now().nanoseconds
            if not self.motion.is_stationary():
                return
            # dwell at vertex
            if self._poly_dwell_due_ns is None:
                self._poly_dwell_due_ns = now + int(self.cfg.circle.dwell_time * 1e9)
                self.get_logger().info("At vertex, dwell for capture…")
                return
            if now < self._poly_dwell_due_ns:
                return
            self._poly_dwell_due_ns = None

            if self._poly_idx >= len(self._poly_wps):
                self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
                self.motion.set_seed_hint(self.cfg.control.init_pos)
                self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
                self._phase = 'return_init'
                return

            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None: return
            self._inflight = True
            ref = (self.motion._last_js.position if self.motion._last_js else self.cfg.control.init_pos)
            self.ik.request_async(self._poly_wps[self._poly_idx], seed, list(ref), self._on_poly_ik)
            self._poly_idx += 1
            return

        if self._phase == 'return_init':
            if not self.motion.is_stationary():
                return
            self._done = True
            self.get_logger().info("Circle finished. Returned to INIT. Exiting.")
            rclpy.shutdown()
            return

    # ---------- IK callbacks ----------
    def _on_hover_ik(self, joint_positions):
        self.traj.publish_positions(joint_positions, self.cfg.control.move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.control.move_time, 0.3)
        self._inflight = False
        self._phase = 'hover_to_center'

    def _on_poly_ik(self, joint_positions):
        self.traj.publish_positions(joint_positions, self.cfg.circle.edge_move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.circle.edge_move_time, 0.3)
        self._inflight = False

    # ---------- image callback: single-shot detection ----------
    def _on_image(self, msg: Image):
        if self._done or self._inflight:
            return
        if self._phase != 'wait_detect':
            return
        if not self.motion.is_stationary():
            return
        self._frame_count += 1
        if self.cfg.runtime.frame_stride > 1 and (self._frame_count % self.cfg.runtime.frame_stride) != 0:
            return

        out = self.detector.detect_once(msg, self.tf_buffer)
        if out is None:
            return
        C, hover, tf_obj, tf_circle = out

        self._fixed_hover = hover
        self._circle_center = C.copy()
        self._ring_z = float(C[2] + self.cfg.control.hover_above)

        self.tf_brd.sendTransform(tf_obj); self._last_obj_tf = tf_obj
        self.tf_brd.sendTransform(tf_circle); self._last_circle_tf = tf_circle

def main():
    rclpy.init()
    node = SeeAnythingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
