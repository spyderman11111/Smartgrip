#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seeanything.py — Two-pass detection + circular path + snapshots

Pipeline:
1) INIT pose -> wait stationary
2) Stage-1 detection (coarse): compute C1, build pose that changes XY to C1 while keeping current tool Z; IK + move
3) Wait stationary
4) Stage-2 detection (fine): compute C2, set hover pose (tool-Z-down) with Z = C2.z + hover_above; IK + move
5) Wait stationary -> compute start_yaw -> generate N-vertex circular path (non-closed)
   * NEW: use configurable sweep_deg (< 360°) to avoid joint limit at the last vertex (default: 355°)
6) Vertex-by-vertex IK; at each dwell: snapshot + save {joint positions + camera pose}
   - If IK abnormal jump on a vertex: skip that vertex and continue (no dwell)
7) Done -> return to INIT -> exit

JSON log (image_jointstates.json):
{
  "note": "Per-image joint positions and camera pose at capture time.",
  "prompt": "<text prompt>",
  "created_at": "<ISO time>",
  "joint_names": [ ... ],
  "shots": { "image_1": { "position": [...], "camera_pose": {...} }, ... }
}
"""

from typing import Optional, List, Dict, Any
import math
import os
import json
from datetime import datetime

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration as RclDuration
from rclpy.time import Time
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge

import tf2_ros

# DINO wrapper
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
# === 复用你提供的 tf_ops 工具 ===
from gripanything.core.tf_ops import tfmsg_to_Rp, R_CL_CO, quat_to_rot


class SeeAnythingNode(Node):
    def __init__(self):
        super().__init__('seeanything_minimal_clean')

        # Load config (with some ROS param overrides)
        self.cfg = load_from_ros_params(self)

        # 环绕角度（度），默认 355°，用于替代满圈 360°
        # 若配置文件/ROS 参数中未提供 circle.sweep_deg，则使用默认值 355.0
        self._sweep_deg: float = float(getattr(getattr(self.cfg, "circle", object()), "sweep_deg", 355.0))

        # Interactive prompt (unless disabled)
        if self.cfg.runtime.require_prompt:
            user_prompt = input("Enter the target text prompt (e.g., 'orange object'): ").strip()
            if user_prompt:
                self.cfg.dino.text_prompt = user_prompt
                self.set_parameters([Parameter('text_prompt', value=user_prompt)])
                self.get_logger().info(f"Using user prompt: \"{user_prompt}\"")
            else:
                self.get_logger().warn(f"No input given; using default prompt: \"{self.cfg.dino.text_prompt}\"")
        else:
            self.get_logger().info(f"Interactive prompt disabled. Using: \"{self.cfg.dino.text_prompt}\"")

        # Prepare paths
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        self._img_dir = os.path.join(self._script_dir, 'ur5image')
        os.makedirs(self._img_dir, exist_ok=True)

        # JSON log (simplified)
        self._js_path = os.path.join(self._script_dir, "image_jointstates.json")
        self._joint_log: Dict[str, Any] = {
            "note": "Per-image joint positions and camera pose at capture time.",
            "prompt": self.cfg.dino.text_prompt,
            "created_at": datetime.now().isoformat(),
            "joint_names": [],
            "shots": {}
        }
        self._flush_joint_log()

        # Rolling camera frame buffer (BGR for cv2.imwrite) + last image stamp
        self._bridge = CvBridge()
        self._latest_bgr: Optional[np.ndarray] = None
        self._last_img_stamp = None

        # Image subscription (trigger + buffer)
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
        self.motion = MotionContext(
            self,
            self.cfg.control.joint_order,
            self.cfg.control.vel_eps,
            self.cfg.control.require_stationary,
        )
        self.traj = TrajectoryPublisher(
            self, self.cfg.control.joint_order, self.cfg.control.controller_topic
        )
        self.ik = IKClient(
            self,
            self.cfg.control.group_name,
            self.cfg.control.ik_link_name,
            self.cfg.control.joint_order,
            self.cfg.control.ik_timeout,
            self.cfg.jump.ignore_joints,
            self.cfg.jump.max_safe_jump,
            self.cfg.jump.max_warn_jump,
        )

        # State
        self._phase = 'init_needed'
        self._inflight = False
        self._done = False

        # Stage-1 / Stage-2 cached poses and detection
        self._pose_stage1: Optional[PoseStamped] = None
        self._fixed_hover: Optional[PoseStamped] = None
        self._circle_center: Optional[np.ndarray] = None
        self._ring_z: Optional[float] = None

        # Circular path state
        self._start_yaw: Optional[float] = None
        self._poly_wps: List[PoseStamped] = []
        self._poly_idx: int = 0
        self._poly_dwell_due_ns: Optional[int] = None
        self._skip_last_vertex = False

        # TF rebroadcast
        self._last_obj_tf: Optional[TransformStamped] = None
        self._last_circle_tf: Optional[TransformStamped] = None
        if self.cfg.control.tf_rebroadcast_hz > 0:
            self.create_timer(1.0 / self.cfg.control.tf_rebroadcast_hz, self._rebroadcast_tfs)

        # Main loop
        self.create_timer(0.05, self._tick)

        self._frame_count = 0
        self.get_logger().info(
            f"[seeanything] topic={self.cfg.frames.image_topic}, hover={self.cfg.control.hover_above:.3f}m, "
            f"bias=({self.cfg.bias.bx:.3f},{self.cfg.bias.by:.3f},{self.cfg.bias.bz:.3f}); "
            f"N={self.cfg.circle.n_vertices}, R={self.cfg.circle.radius:.3f}m, orient={self.cfg.circle.orient_mode}, "
            f"dir={self.cfg.circle.poly_dir}, sweep={self._sweep_deg:.1f}°"
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

    # ---------- helpers ----------
    def _get_tool_z_now(self) -> Optional[float]:
        """Read current tool Z in base frame."""
        try:
            T_bt = self.tf_buffer.lookup_transform(
                self.cfg.frames.base_frame, self.cfg.frames.tool_frame, Time(),
                timeout=RclDuration(seconds=0.2)
            )
            _, p_bt = tfmsg_to_Rp(T_bt)
            return float(p_bt[2])
        except Exception as ex:
            self.get_logger().warn(f"Read tool Z failed: {ex}")
            return None

    def _calc_camera_pose_for_last_image(self) -> Optional[Dict[str, Any]]:
        """
        Compute base->camera_optical pose at the timestamp of the last received image.
        Uses cfg.cam.{t_tool_cam_xyz, t_tool_cam_quat_xyzw, hand_eye_frame}.
        """
        if self._last_img_stamp is None:
            return None

        # 1) TF base<-tool0 at the image timestamp (try exact; fallback to latest on failure)
        try:
            t_query = Time.from_msg(self._last_img_stamp)
            T_bt = self.tf_buffer.lookup_transform(
                self.cfg.frames.base_frame, self.cfg.frames.tool_frame, t_query,
                timeout=RclDuration(seconds=0.3)
            )
        except Exception as ex:
            self.get_logger().warn(f"TF lookup at image stamp failed ({ex}); using latest.")
            try:
                T_bt = self.tf_buffer.lookup_transform(
                    self.cfg.frames.base_frame, self.cfg.frames.tool_frame, Time(),
                    timeout=RclDuration(seconds=0.2)
                )
            except Exception as ex2:
                self.get_logger().warn(f"TF latest lookup also failed: {ex2}")
                return None

        R_bt, p_bt = tfmsg_to_Rp(T_bt)

        # 2) tool->camera_optical
        qx, qy, qz, qw = self.cfg.cam.t_tool_cam_quat_xyzw
        R_t_cam = quat_to_rot(qx, qy, qz, qw)
        if str(self.cfg.cam.hand_eye_frame).lower() == 'optical':
            R_t_co = R_t_cam
        else:
            # convert camera_link to camera_optical
            R_t_co = R_t_cam @ R_CL_CO
        p_t_co = np.array(self.cfg.cam.t_tool_cam_xyz, dtype=float)

        # 3) base->camera_optical
        R_bc = R_bt @ R_t_co
        p_bc = R_bt @ p_t_co + p_bt

        return {
            "parent_frame": self.cfg.frames.base_frame,
            "child_frame": "camera_optical",
            "stamp": {
                "sec": int(self._last_img_stamp.sec) if self._last_img_stamp else None,
                "nanosec": int(self._last_img_stamp.nanosec) if self._last_img_stamp else None
            },
            "R": R_bc.astype(float).round(12).tolist(),
            "t": [float(p_bc[0]), float(p_bc[1]), float(p_bc[2])]
        }

    # ---------- image callback: two-pass detection triggers ----------
    def _on_image(self, msg: Image):
        # update rolling buffer + stamp
        try:
            self._latest_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass
        self._last_img_stamp = msg.header.stamp

        if self._done or self._inflight:
            return
        if self._phase not in ('wait_detect_stage1', 'wait_detect_stage2'):
            return
        if not self.motion.is_stationary():
            return

        # stride throttle
        self._frame_count += 1
        if self.cfg.runtime.frame_stride > 1 and (self._frame_count % self.cfg.runtime.frame_stride) != 0:
            return

        out = self.detector.detect_once(msg, self.tf_buffer)
        if out is None:
            return
        C, hover, tf_obj, tf_circle = out

        # rebroadcast for RViz
        self.tf_brd.sendTransform(tf_obj);    self._last_obj_tf = tf_obj
        self.tf_brd.sendTransform(tf_circle); self._last_circle_tf = tf_circle

        if self._phase == 'wait_detect_stage1':
            # Stage-1: change XY to C, keep current Z (tool-Z-down)
            z_keep = self._get_tool_z_now()
            if z_keep is None:
                self.get_logger().warn("Cannot read current Z; waiting next frame.")
                return
            ps = PoseStamped()
            ps.header.frame_id = self.cfg.frames.pose_frame
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = float(C[0])
            ps.pose.position.y = float(C[1])
            ps.pose.position.z = float(z_keep)
            # tool-Z-down: (w,x,y,z) = (0,1,0,0)
            ps.pose.orientation.w = 0.0
            ps.pose.orientation.x = 1.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = 0.0
            self._pose_stage1 = ps
            self.get_logger().info(
                f"[Stage-1] Move XY→({C[0]:.3f},{C[1]:.3f}), keep Z={z_keep:.3f}"
            )

        elif self._phase == 'wait_detect_stage2':
            # Stage-2: set hover over C (XY = C, Z = C.z + hover_above)
            self._fixed_hover = hover
            self._circle_center = C.copy()
            self._ring_z = float(hover.pose.position.z)  # equals C.z + hover_above
            self.get_logger().info(
                f"[Stage-2] Move XY→({C[0]:.3f},{C[1]:.3f}), Z→{self._ring_z:.3f}"
            )

    # ---------- helper: write the joint log to disk ----------
    def _flush_joint_log(self):
        try:
            with open(self._js_path, "w", encoding="utf-8") as f:
                json.dump(self._joint_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().error(f"Failed to write joint log JSON: {e}")

    # ---------- Save snapshot ----------
    def _save_snapshot(self, vertex_index_zero_based: int):
        idx1 = int(vertex_index_zero_based) + 1  # 1-based numbering for files
        if self._latest_bgr is None:
            self.get_logger().warn("No camera frame available to save.")
            return
        fname = f"pose_{idx1}_image.png"
        fpath = os.path.join(self._img_dir, fname)
        try:
            ok = cv2.imwrite(fpath, self._latest_bgr)
            if ok:
                self.get_logger().info(f"Saved snapshot: {fpath}")
            else:
                self.get_logger().error(f"cv2.imwrite returned False for: {fpath}")
        except Exception as e:
            self.get_logger().error(f"Failed to save snapshot to {fpath}: {e}")

    # ---------- Save joint positions (+ camera pose) into ONE JSON ----------
    def _save_joint_positions_and_cam(self, vertex_index_zero_based: int):
        idx1 = int(vertex_index_zero_based) + 1
        key = f"image_{idx1}"

        js = self.motion._last_js  # latest joint state on the topic
        if js is None or not js.position or not js.name:
            self.get_logger().warn("No /joint_states available to log positions.")
            return

        # Fill joint_names once
        if not self._joint_log["joint_names"]:
            self._joint_log["joint_names"] = list(js.name)

        entry = {
            "position": [float(x) for x in js.position],
            "camera_pose": self._calc_camera_pose_for_last_image()
        }

        self._joint_log["shots"][key] = entry
        self._flush_joint_log()
        self.get_logger().info(f'Saved positions & camera pose under key "{key}" -> {self._js_path}')

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
                self._phase = 'wait_detect_stage1'
                self.get_logger().info("INIT reached. Waiting for Stage-1 detection (XY only)…")
            return

        # ---- Stage-1 move (XY only, keep Z) ----
        if self._phase == 'wait_detect_stage1' and self._pose_stage1 is not None:
            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                self.get_logger().warn("Waiting for /joint_states …")
                return
            self._inflight = True
            self.ik.request_async(self._pose_stage1, seed, self._on_ik_stage1)
            return

        if self._phase == 'stage1_moving':
            if self.motion.is_stationary():
                self._pose_stage1 = None
                self._phase = 'wait_detect_stage2'
                self.get_logger().info("Stage-1 done. Waiting for Stage-2 detection (XY + descend)…")
            return

        # ---- Stage-2 move (hover over center) ----
        if self._phase == 'wait_detect_stage2' and self._fixed_hover is not None:
            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                self.get_logger().warn("Waiting for /joint_states …")
                return
            self._inflight = True
            self.ik.request_async(self._fixed_hover, seed, self._on_hover_ik)
            return

        if self._phase == 'hover_to_center':
            if not self.motion.is_stationary():
                return
            # At hover; compute start yaw and make polygon vertices
            self._start_yaw = self._get_tool_yaw_xy()
            if self._circle_center is not None and self._ring_z is not None and self._start_yaw is not None:
                # 1) 先按原逻辑生成整圈顶点（可能是 360° * num_turns）
                all_wps = make_polygon_vertices(
                    self.get_clock().now().to_msg,
                    self._circle_center, self._ring_z, self._start_yaw,
                    self.cfg.frames.pose_frame,
                    self.cfg.circle.n_vertices, self.cfg.circle.num_turns, self.cfg.circle.poly_dir,
                    self.cfg.circle.orient_mode, self.cfg.circle.start_dir_offset_deg,
                    self.cfg.circle.radius, self.cfg.circle.tool_z_sign
                )
                # 2) NEW: 按 sweep_deg 裁剪顶点数，避免到达满圈末端触限位
                total_deg = 360.0 * float(self.cfg.circle.num_turns)
                sweep_deg = max(0.0, min(float(self._sweep_deg), total_deg))  # clamp
                if sweep_deg < total_deg and len(all_wps) > 1:
                    keep = max(1, int(math.floor(len(all_wps) * (sweep_deg / total_deg))))
                    # 确保至少保留 1 个顶点，且小于等于总数
                    keep = min(keep, len(all_wps))
                    self._poly_wps = all_wps[:keep]
                    self.get_logger().info(
                        f"Generated vertices: {len(all_wps)} -> trimmed to {len(self._poly_wps)} "
                        f"for sweep {sweep_deg:.1f}° / {total_deg:.1f}°."
                    )
                else:
                    self._poly_wps = all_wps
                    self.get_logger().info(f"Generated vertices: {len(self._poly_wps)} (full sweep).")

                self._poly_idx = 0
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
            self.ik.request_async(self._poly_wps[0], seed, self._on_poly_ik)
            self._poly_idx = 1
            self._phase = 'poly_moving'
            return

        if self._phase == 'poly_moving':
            # If the last IK was skipped due to abnormal jump, jump straight to next vertex (no dwell)
            if self._skip_last_vertex:
                self._skip_last_vertex = False
                if self._poly_idx >= len(self._poly_wps):
                    self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
                    self.motion.set_seed_hint(self.cfg.control.init_pos)
                    self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
                    self._phase = 'return_init'
                    return
                if not self.ik.ready():
                    return
                seed = self.motion.make_seed()
                if seed is None:
                    return
                self._inflight = True
                self.ik.request_async(self._poly_wps[self._poly_idx], seed, self._on_poly_ik)
                self._poly_idx += 1
                return

            now = self.get_clock().now().nanoseconds
            if not self.motion.is_stationary():
                return
            # dwell at current vertex -> snapshot + (positions + camera pose)
            if self._poly_dwell_due_ns is None:
                self._poly_dwell_due_ns = now + int(self.cfg.circle.dwell_time * 1e9)
                curr_vertex0 = max(0, self._poly_idx - 1)  # zero-based
                self.get_logger().info(f"At vertex {curr_vertex0 + 1}, dwell for capture…")
                self._save_snapshot(curr_vertex0)
                self._save_joint_positions_and_cam(curr_vertex0)
                # 如果这是最后一个顶点，驻留期间就准备直接回 INIT（驻留结束立即返回）
                self._at_last_vertex = (self._poly_idx >= len(self._poly_wps))
                return

            if now < self._poly_dwell_due_ns:
                return
            self._poly_dwell_due_ns = None

            # 如果已经到达最后一个点并完成拍照 -> 直接回 INIT
            if getattr(self, "_at_last_vertex", False):
                self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
                self.motion.set_seed_hint(self.cfg.control.init_pos)
                self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
                self._phase = 'return_init'
                return

            if self._poly_idx >= len(self._poly_wps):
                # 保险分支
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
            self.ik.request_async(self._poly_wps[self._poly_idx], seed, self._on_poly_ik)
            self._poly_idx += 1
            return

        if self._phase == 'return_init':
            if not self.motion.is_stationary():
                return
            self._done = True
            self.get_logger().info("Circle finished. Returned to INIT. Exiting.")
            rclpy.shutdown()
            return

    # ---------- current tool yaw (XY) ----------
    def _get_tool_yaw_xy(self) -> Optional[float]:
        try:
            T_bt = self.tf_buffer.lookup_transform(
                self.cfg.frames.base_frame, self.cfg.frames.tool_frame, Time(),
                timeout=RclDuration(seconds=0.2)
            )
            R_bt, _ = tfmsg_to_Rp(T_bt)
            ex = R_bt[:, 0]
            return float(math.atan2(float(ex[1]), float(ex[0])))
        except Exception as ex:
            self.get_logger().warn(f"Read tool yaw failed: {ex}")
            return None

    # ---------- IK callbacks ----------
    def _on_ik_stage1(self, joint_positions: Optional[List[float]]):
        self._inflight = False
        if joint_positions is None:
            # Stage-1 failed or abnormal jump: go back to INIT & exit
            self.get_logger().warn("Stage-1 IK skipped/failed. Returning to INIT and exiting.")
            self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
            self.motion.set_seed_hint(self.cfg.control.init_pos)
            self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
            self._phase = 'return_init'
            return

        self.traj.publish_positions(joint_positions, self.cfg.control.move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.control.move_time, 0.3)
        self._phase = 'stage1_moving'

    def _on_hover_ik(self, joint_positions: Optional[List[float]]):
        self._inflight = False
        if joint_positions is None:
            # Hover IK could not be realized — exit gracefully
            self.get_logger().warn("Stage-2 hover IK skipped/failed. Returning to INIT and exiting.")
            self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
            self.motion.set_seed_hint(self.cfg.control.init_pos)
            self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
            self._phase = 'return_init'
            return

        self.traj.publish_positions(joint_positions, self.cfg.control.move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.control.move_time, 0.3)
        self._phase = 'hover_to_center'

    def _on_poly_ik(self, joint_positions: Optional[List[float]]):
        if joint_positions is None:
            # Skip this vertex and continue with the next one without dwell/snapshot
            self.get_logger().warn("Vertex IK skipped after abnormal jump. Moving to next vertex.")
            self._inflight = False
            self._poly_dwell_due_ns = None
            self._skip_last_vertex = True
            return

        self.traj.publish_positions(joint_positions, self.cfg.circle.edge_move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.circle.edge_move_time, 0.3)
        self._inflight = False


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
