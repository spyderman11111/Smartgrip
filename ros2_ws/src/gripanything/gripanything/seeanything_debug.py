#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seeanything.py — Two-pass detection + circular path + snapshots + VGGT + postprocess + final goto

Key fix (time-binding stability):
- Atomically bind {image, image_stamp} at capture time
- Compute TF strictly at that image_stamp (NO fallback-to-latest)
- Save image + json entry together in ONE atomic "shot" function
- If TF at stamp is unavailable: skip saving that shot (avoid image<->pose mismatch)

Pipeline
1) Move to INIT pose -> wait until stationary
2) Stage-1 detection (coarse): compute C1; build a pose that updates XY to C1 while keeping current tool Z; IK + move
3) Wait until stationary
4) Stage-2 detection (fine): compute C2; set hover pose (tool-Z-down) with Z = C2.z + hover_above; IK + move
5) Wait until stationary -> compute start_yaw -> generate an N-vertex polygon/circular path (non-closed)
   - Use sweep_deg (< 360°) to avoid joint limits near the last vertex
6) Vertex-by-vertex IK; at each dwell: atomically capture {image + joint positions + camera pose @ image stamp}
   - If IK indicates an abnormal jump on a vertex: skip that vertex and continue (no dwell)
7) Return to INIT
8) Offline pipeline:
   - Run VGGT reconstruction from captured images
   - Post-process + align to base_link
   - Export object center + shape vertices in base_link (new: prism/vertices, no OBB)
   - Apply an additional OFFLINE bias to exported coordinates (center + vertices)
9) Final goto:
   - After object center is available, move to (offset above) object center (tool-Z-down)
   - Wait until stationary, then exit
"""

from typing import Optional, List, Dict, Any
import math
import os
import json
import threading
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


# -----------------------------------------------------------------------------
# Output paths (fixed)
# -----------------------------------------------------------------------------
OUTPUT_ROOT = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output"
OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "ur5image")
OUTPUT_JSON_PATH = os.path.join(OUTPUT_ROOT, "ur5camerajointstates.json")
OUTPUT_VGGT_DIR = os.path.join(OUTPUT_ROOT, "offline_output")
OUTPUT_OBJECT_JSON_NAME = "object_in_base_link.json"


# -----------------------------------------------------------------------------
# Offline pipeline toggles (recommended defaults for ROS)
# -----------------------------------------------------------------------------
RUN_OFFLINE_PIPELINE = True
VGGT_AUTO_VISUALIZE = False          # Avoid Open3D window blocking the ROS node
POSTPROCESS_VISUALIZE = False        # Avoid Open3D window blocking the ROS node

# VGGT reconstruction parameters (keep explicit and stable for reproducibility)
VGGT_BATCH_SIZE = 30
VGGT_MAX_POINTS = 1_500_000
VGGT_RESOLUTION = 518
VGGT_CONF_THRESH = 1.5

# Alignment mode for VGGT-world -> base_link ("sim3" estimates scale; "se3" forces scale=1)
POST_ALIGN_METHOD = "sim3"

# IMPORTANT: Our VGGT cameras.json stores "cam_T_world" (cam <- world).
# The postprocess loader expects world_T_cam, so we set vggt_pose_is_world_T_cam accordingly.
POST_VGGT_POSE_IS_WORLD_T_CAM = True


# -----------------------------------------------------------------------------
# Final goto (after offline pipeline)
# -----------------------------------------------------------------------------
FINAL_GOTO_ENABLE = True
FINAL_GOTO_Z_OFFSET = 0.15          # meters above object center
FINAL_GOTO_MOVE_TIME = 3.0          # seconds
FINAL_GOTO_Z_MIN = 0.05             # safety clamp
FINAL_GOTO_Z_MAX = 2.00             # safety clamp


# -----------------------------------------------------------------------------
# DINO wrapper (robust import)
# -----------------------------------------------------------------------------
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except Exception:
    import sys
    sys.path.append("/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything")
    from gripanything.core.detect_with_dino import GroundingDinoPredictor

from gripanything.core.config import load_from_ros_params
from gripanything.core.vision_geom import SingleShotDetector
from gripanything.core.ik_and_traj import MotionContext, TrajectoryPublisher, IKClient
from gripanything.core.polygon_path import make_polygon_vertices

# Reuse TF utilities
from gripanything.core.tf_ops import tfmsg_to_Rp, R_CL_CO, quat_to_rot


# -----------------------------------------------------------------------------
# Offline pipeline imports (VGGT + point cloud postprocess)
# -----------------------------------------------------------------------------
def _import_offline_modules():
    """
    Imports offline pipeline modules based on the current repository tree.

    Expected:
      - gripanything.core.vggtreconstruction provides: VGGTConfig, VGGTReconstructor
      - gripanything.core.point_processing provides: Config, process_pointcloud
    """
    try:
        from gripanything.core.vggtreconstruction import VGGTConfig as _VGGTConfig, VGGTReconstructor as _VGGTReconstructor
    except Exception:
        import sys
        sys.path.append("/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything")
        from gripanything.core.vggtreconstruction import VGGTConfig as _VGGTConfig, VGGTReconstructor as _VGGTReconstructor

    try:
        from gripanything.core.point_processing import Config as _PostConfig, process_pointcloud as _process_pointcloud
    except Exception:
        import sys
        sys.path.append("/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything")
        from gripanything.core.point_processing import Config as _PostConfig, process_pointcloud as _process_pointcloud

    return _VGGTConfig, _VGGTReconstructor, _PostConfig, _process_pointcloud


def _safe_remove(path: str) -> None:
    """Remove a file if it exists; do nothing on failure."""
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _cleanup_outputs_for_new_run() -> None:
    """
    Ensure that a new run overwrites outputs cleanly:
    - Remove old pose_* images so leftover frames do not contaminate the next run
    - Remove old JSON log
    - Remove old VGGT outputs that we generate (points/cameras/lines/object json)
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VGGT_DIR, exist_ok=True)

    # Images from previous run
    try:
        for fn in os.listdir(OUTPUT_IMG_DIR):
            if fn.startswith("pose_") and ("_image." in fn or fn.endswith(".png") or fn.endswith(".jpg")):
                _safe_remove(os.path.join(OUTPUT_IMG_DIR, fn))
    except Exception:
        pass

    # Main shot log
    _safe_remove(OUTPUT_JSON_PATH)

    # VGGT outputs
    _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "points.ply"))
    _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "cameras.json"))
    _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "cameras_lines.ply"))
    _safe_remove(os.path.join(OUTPUT_VGGT_DIR, OUTPUT_OBJECT_JSON_NAME))


def _stamp_to_ns(stamp_msg) -> int:
    try:
        return int(stamp_msg.sec) * 1_000_000_000 + int(stamp_msg.nanosec)
    except Exception:
        return -1


class SeeAnythingNode(Node):
    def __init__(self):
        super().__init__("seeanything_minimal_clean")

        # Always clean outputs at node startup to guarantee overwriting behavior.
        _cleanup_outputs_for_new_run()

        # Load config (with some ROS param overrides)
        self.cfg = load_from_ros_params(self)

        # Sweep angle in degrees; used to trim the full 360° loop to avoid joint limits.
        # If cfg.circle.sweep_deg is not available, the fallback default is 120.0.
        self._sweep_deg: float = float(getattr(getattr(self.cfg, "circle", object()), "sweep_deg", 120.0))

        # Interactive prompt (unless disabled)
        if self.cfg.runtime.require_prompt:
            user_prompt = input("Enter the target text prompt (e.g., 'orange object'): ").strip()
            if user_prompt:
                self.cfg.dino.text_prompt = user_prompt
                self.set_parameters([Parameter("text_prompt", value=user_prompt)])
                self.get_logger().info(f'Using user prompt: "{user_prompt}"')
            else:
                self.get_logger().warn(f'No input given; using default prompt: "{self.cfg.dino.text_prompt}"')
        else:
            self.get_logger().info(f'Interactive prompt disabled. Using: "{self.cfg.dino.text_prompt}"')

        # Fixed output locations
        self._img_dir = OUTPUT_IMG_DIR
        self._js_path = OUTPUT_JSON_PATH

        # JSON log (single file, always overwrite)
        self._joint_log: Dict[str, Any] = {
            "note": "Per-image joint positions and camera pose at capture time.",
            "prompt": self.cfg.dino.text_prompt,
            "created_at": datetime.now().isoformat(),
            "joint_names": [],
            "shots": {}
        }
        self._flush_joint_log()

        # Rolling camera frame buffer (BGR) + last image stamp (atomic via lock)
        self._bridge = CvBridge()
        self._img_lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_img_stamp = None   # msg.header.stamp

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

        # FSM state
        self._phase = "init_needed"
        self._inflight = False
        self._done = False

        # Offline pipeline state (avoid running twice)
        self._offline_ran = False

        # Stage-1 / Stage-2 cached poses and detection outputs
        self._pose_stage1: Optional[PoseStamped] = None
        self._fixed_hover: Optional[PoseStamped] = None
        self._circle_center: Optional[np.ndarray] = None
        self._ring_z: Optional[float] = None

        # Circular/polygon path state
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

        # Final goto state
        self._final_point_base: Optional[np.ndarray] = None   # (x,y,z) in base_link
        self._final_pose: Optional[PoseStamped] = None
        self._final_goto_requested = False

        # Main loop
        self.create_timer(0.05, self._tick)

        self._frame_count = 0

        ob = self.cfg.offline_bias
        self.get_logger().info(
            f"[seeanything] topic={self.cfg.frames.image_topic}, hover={self.cfg.control.hover_above:.3f}m, "
            f"online_bias=({self.cfg.bias.bx:.3f},{self.cfg.bias.by:.3f},{self.cfg.bias.bz:.3f}); "
            f"offline_bias(enable={ob.enable}, ox={ob.ox:.3f}, oy={ob.oy:.3f}, oz={ob.oz:.3f}); "
            f"N={self.cfg.circle.n_vertices}, R={self.cfg.circle.radius:.3f}m, orient={self.cfg.circle.orient_mode}, "
            f"dir={self.cfg.circle.poly_dir}, sweep={self._sweep_deg:.1f}deg; "
            f"out_img_dir={self._img_dir}, out_json={self._js_path}; "
            f"vggt_out_dir={OUTPUT_VGGT_DIR}"
        )

    # ---------- TF rebroadcast ----------
    def _rebroadcast_tfs(self):
        now = self.get_clock().now().to_msg()
        if self._last_obj_tf is not None:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self._last_obj_tf.header.frame_id
            t.child_frame_id = self._last_obj_tf.child_frame_id
            t.transform = self._last_obj_tf.transform
            self.tf_brd.sendTransform(t)
        if self._last_circle_tf is not None:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self._last_circle_tf.header.frame_id
            t.child_frame_id = self._last_circle_tf.child_frame_id
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

    def _calc_camera_pose_at_stamp(self, stamp_msg) -> Optional[Dict[str, Any]]:
        """
        Compute base->camera_optical pose strictly at the provided image stamp.
        NO fallback-to-latest. If TF is not available at this stamp, return None.

        Uses cfg.cam.{t_tool_cam_xyz, t_tool_cam_quat_xyzw, hand_eye_frame}.
        """
        if stamp_msg is None:
            return None

        t_query = Time.from_msg(stamp_msg)

        # 1) TF base<-tool0 at the image timestamp (exact)
        try:
            T_bt = self.tf_buffer.lookup_transform(
                self.cfg.frames.base_frame, self.cfg.frames.tool_frame, t_query,
                timeout=RclDuration(seconds=0.3)
            )
        except Exception as ex:
            self.get_logger().warn(f"TF lookup at image stamp failed (no fallback): {ex}")
            return None

        R_bt, p_bt = tfmsg_to_Rp(T_bt)

        # 2) tool->camera_optical
        qx, qy, qz, qw = self.cfg.cam.t_tool_cam_quat_xyzw
        R_t_cam = quat_to_rot(qx, qy, qz, qw)

        if str(self.cfg.cam.hand_eye_frame).lower() == "optical":
            R_t_co = R_t_cam
        else:
            # convert camera_link to camera_optical
            R_t_co = R_t_cam @ R_CL_CO

        p_t_co = np.array(self.cfg.cam.t_tool_cam_xyz, dtype=float)

        # 3) base->camera_optical
        R_bc = R_bt @ R_t_co
        p_bc = R_bt @ p_t_co + p_bt

        stamp_ns = _stamp_to_ns(stamp_msg)

        return {
            "parent_frame": self.cfg.frames.base_frame,
            "child_frame": "camera_optical",
            "stamp": {
                "sec": int(stamp_msg.sec),
                "nanosec": int(stamp_msg.nanosec),
                "stamp_ns": int(stamp_ns),
            },
            "R": R_bc.astype(float).round(12).tolist(),
            "t": [float(p_bc[0]), float(p_bc[1]), float(p_bc[2])]
        }

    def _apply_offline_bias_to_object_json(self, obj_json_path: str, offset_xyz: np.ndarray) -> bool:
        """
        Apply an additive offset (in base_link) to the exported object JSON.

        New schema support:
          - object.center.base_link
          - object.prism.*.base_link (e.g., vertices_8/corners_8/footprint_corners_4/top_corners_4/bottom_corners_4)
        Backward compatibility:
          - object.obb.corners_8.base_link

        The file is overwritten in-place.

        Returns True if the file was modified and saved.
        """
        try:
            if not obj_json_path or (not os.path.isfile(obj_json_path)):
                return False

            with open(obj_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            obj = data.get("object", {})
            if not isinstance(obj, dict):
                return False

            ox, oy, oz = float(offset_xyz[0]), float(offset_xyz[1]), float(offset_xyz[2])

            def _is_num(x) -> bool:
                return isinstance(x, (int, float))

            def _is_vec3(p) -> bool:
                return isinstance(p, (list, tuple)) and len(p) == 3 and all(_is_num(v) for v in p)

            def _is_list_vec3(ps) -> bool:
                return isinstance(ps, list) and len(ps) > 0 and all(_is_vec3(p) for p in ps)

            def _add3(p):
                return [float(p[0]) + ox, float(p[1]) + oy, float(p[2]) + oz]

            changed = False

            # 1) Center (explicit)
            c = obj.get("center", None)
            if isinstance(c, dict) and _is_vec3(c.get("base_link", None)):
                c["base_link"] = _add3(c["base_link"])
                changed = True
            elif _is_vec3(obj.get("center_base", None)):
                obj["center_base"] = _add3(obj["center_base"])
                changed = True
            elif _is_vec3(obj.get("center_base_link", None)):
                obj["center_base_link"] = _add3(obj["center_base_link"])
                changed = True

            # 2) Backward-compatible OBB corners if present
            obb = obj.get("obb", None)
            if isinstance(obb, dict):
                c8 = obb.get("corners_8", None)
                if isinstance(c8, dict) and _is_list_vec3(c8.get("base_link", None)):
                    c8["base_link"] = [_add3(p) for p in c8["base_link"]]
                    changed = True

            # 3) New prism/shape vertices: apply to any point-list under object subtree with key "base_link"
            #    but ONLY when it is a list of 3D points (list[list[3]]). This avoids touching extents/scalars.
            def _walk_apply_points(node):
                nonlocal changed
                if isinstance(node, dict):
                    for k, v in node.items():
                        if k == "base_link" and _is_list_vec3(v):
                            node[k] = [_add3(p) for p in v]
                            changed = True
                        else:
                            _walk_apply_points(v)
                elif isinstance(node, list):
                    for it in node:
                        _walk_apply_points(it)

            _walk_apply_points(obj)

            if not changed:
                return False

            with open(obj_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as ex:
            self.get_logger().warn(f"[offline] Failed to apply offline bias to object JSON: {ex}")
            return False

    # ---------- image callback: two-pass detection triggers ----------
    def _on_image(self, msg: Image):
        # Update rolling buffer + stamp atomically
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        with self._img_lock:
            self._latest_bgr = bgr
            self._latest_img_stamp = msg.header.stamp

        if self._done or self._inflight:
            return
        if self._phase not in ("wait_detect_stage1", "wait_detect_stage2"):
            return
        if not self.motion.is_stationary():
            return

        # Stride throttle
        self._frame_count += 1
        if self.cfg.runtime.frame_stride > 1 and (self._frame_count % self.cfg.runtime.frame_stride) != 0:
            return

        out = self.detector.detect_once(msg, self.tf_buffer)
        if out is None:
            return
        C, hover, tf_obj, tf_circle = out

        # Broadcast TF for RViz
        self.tf_brd.sendTransform(tf_obj)
        self._last_obj_tf = tf_obj
        self.tf_brd.sendTransform(tf_circle)
        self._last_circle_tf = tf_circle

        if self._phase == "wait_detect_stage1":
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
            # tool-Z-down quaternion: (w,x,y,z) = (0,1,0,0)
            ps.pose.orientation.w = 0.0
            ps.pose.orientation.x = 1.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = 0.0
            self._pose_stage1 = ps
            self.get_logger().info(f"[Stage-1] Move XY->({C[0]:.3f},{C[1]:.3f}), keep Z={z_keep:.3f}")

        elif self._phase == "wait_detect_stage2":
            # Stage-2: hover over C (XY = C, Z = C.z + hover_above)
            self._fixed_hover = hover
            self._circle_center = C.copy()
            self._ring_z = float(hover.pose.position.z)  # equals C.z + hover_above
            self.get_logger().info(f"[Stage-2] Move XY->({C[0]:.3f},{C[1]:.3f}), Z->{self._ring_z:.3f}")

    # ---------- helper: write the joint log to disk (always overwrite) ----------
    def _flush_joint_log(self):
        try:
            with open(self._js_path, "w", encoding="utf-8") as f:
                json.dump(self._joint_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().error(f"Failed to write joint log JSON: {e}")

    # ---------- Atomic capture: image + joint positions + camera pose at image stamp ----------
    def _capture_and_log_shot(self, vertex_index_zero_based: int) -> bool:
        """
        Atomically bind:
          - image file (pose_{k}_image.png)
          - image stamp
          - camera_pose computed at that stamp (NO fallback)
          - joint positions (latest) + joint stamp for debugging

        If TF at the exact image stamp is not available, skip saving this shot to avoid mismatched pairing.
        """
        idx1 = int(vertex_index_zero_based) + 1
        key = f"image_{idx1}"
        fname = f"pose_{idx1}_image.png"
        fpath = os.path.join(self._img_dir, fname)

        # 1) Freeze (bgr + stamp) atomically
        with self._img_lock:
            if self._latest_bgr is None or self._latest_img_stamp is None:
                self.get_logger().warn("No camera frame/stamp available to capture.")
                return False
            bgr = self._latest_bgr.copy()
            stamp_msg = self._latest_img_stamp

        # 2) Compute camera pose at THIS image stamp (no fallback)
        cam_pose = self._calc_camera_pose_at_stamp(stamp_msg)
        if cam_pose is None:
            self.get_logger().warn(f"[capture] Skip idx={idx1}: TF not available at this image stamp (no fallback).")
            return False

        # 3) Get joint state (latest). We log joint_stamp_ns for post-mortem checking.
        js = self.motion._last_js
        if js is None or (not js.position) or (not js.name):
            self.get_logger().warn("[capture] No /joint_states available; skip shot.")
            return False

        if not self._joint_log["joint_names"]:
            self._joint_log["joint_names"] = list(js.name)

        joint_stamp_ns = -1
        try:
            joint_stamp_ns = _stamp_to_ns(js.header.stamp)
        except Exception:
            joint_stamp_ns = -1

        # 4) Save image first (VGGT needs it), then write JSON entry.
        try:
            ok = cv2.imwrite(fpath, bgr)
            if not ok:
                self.get_logger().error(f"[capture] cv2.imwrite returned False for: {fpath}")
                return False
        except Exception as e:
            self.get_logger().error(f"[capture] Failed to save snapshot to {fpath}: {e}")
            return False

        entry = {
            "image_file": fname,
            "image_stamp_ns": int(cam_pose["stamp"].get("stamp_ns", -1)),
            "joint_stamp_ns": int(joint_stamp_ns),
            "position": [float(x) for x in js.position],
            "camera_pose": cam_pose,
        }

        self._joint_log["shots"][key] = entry
        self._flush_joint_log()
        self.get_logger().info(
            f"[capture] Saved idx={idx1}: {fpath} | img_stamp_ns={entry['image_stamp_ns']} | joint_stamp_ns={entry['joint_stamp_ns']}"
        )
        return True

    # ---------- Backward-compatible wrappers (keep your call sites intact) ----------
    def _save_snapshot(self, vertex_index_zero_based: int):
        # Capture atomically instead of saving image only
        self._capture_and_log_shot(vertex_index_zero_based)

    def _save_joint_positions_and_cam(self, vertex_index_zero_based: int):
        # No-op: captured together with _save_snapshot() for atomic binding
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

    # ---------- Parse object center from exported JSON ----------
    def _read_center_from_object_json(self, obj_json_path: str) -> Optional[np.ndarray]:
        try:
            if not obj_json_path or (not os.path.isfile(obj_json_path)):
                return None
            with open(obj_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            obj = data.get("object", {})

            p = None

            # Preferred format: object.center.base_link
            if isinstance(obj, dict):
                c = obj.get("center", {})
                if isinstance(c, dict):
                    p = c.get("base_link", None)

            # Backward/alternate formats
            if p is None and isinstance(obj, dict):
                p = obj.get("center_base", None)
            if p is None and isinstance(obj, dict):
                p = obj.get("center_base_link", None)
            if p is None and isinstance(obj, dict):
                p = obj.get("center_b", None)

            if p is None or (not isinstance(p, (list, tuple))) or len(p) != 3:
                return None
            return np.asarray([float(p[0]), float(p[1]), float(p[2])], dtype=float).reshape(3)
        except Exception:
            return None

    # ---------- Build final hover pose (offset above object center) ----------
    def _build_final_hover_pose(self, center_b: np.ndarray) -> PoseStamped:
        x, y, z = float(center_b[0]), float(center_b[1]), float(center_b[2])
        z_hover = z + float(FINAL_GOTO_Z_OFFSET)
        z_hover = max(float(FINAL_GOTO_Z_MIN), min(float(FINAL_GOTO_Z_MAX), z_hover))

        ps = PoseStamped()
        ps.header.frame_id = self.cfg.frames.pose_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z_hover
        # tool-Z-down quaternion
        ps.pose.orientation.w = 0.0
        ps.pose.orientation.x = 1.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = 0.0
        return ps

    # ---------- Offline pipeline: VGGT + postprocess ----------
    def _run_offline_pipeline_once(self):
        if not RUN_OFFLINE_PIPELINE:
            self.get_logger().info("[offline] RUN_OFFLINE_PIPELINE=False, skipping VGGT + postprocess.")
            return
        if self._offline_ran:
            return

        self._offline_ran = True

        # Block control while running offline pipeline
        self._inflight = True

        # Ensure old outputs are removed (explicit overwrite behavior)
        _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "points.ply"))
        _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "cameras.json"))
        _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "cameras_lines.ply"))
        _safe_remove(os.path.join(OUTPUT_VGGT_DIR, OUTPUT_OBJECT_JSON_NAME))

        try:
            VGGTConfig, VGGTReconstructor, PostConfig, process_pointcloud = _import_offline_modules()
        except Exception as e:
            self.get_logger().error(f"[offline] Failed to import offline modules: {e}")
            self._inflight = False
            return

        # 1) VGGT reconstruction
        try:
            self.get_logger().info("[offline] Running VGGT reconstruction...")
            vcfg = VGGTConfig(
                images_dir=self._img_dir,
                out_dir=OUTPUT_VGGT_DIR,
                batch_size=VGGT_BATCH_SIZE,
                max_points=VGGT_MAX_POINTS,
                resolution=VGGT_RESOLUTION,
                conf_thresh=VGGT_CONF_THRESH,
                img_limit=None,
                auto_visualize=VGGT_AUTO_VISUALIZE,
                seed=42,
            )
            recon = VGGTReconstructor(vcfg)

            # Lazy import to avoid ROS startup issues and to fix "torch not defined"
            try:
                import torch
            except Exception as e:
                self.get_logger().error(f"[offline] torch import failed: {e}")
                self._inflight = False
                return

            with torch.inference_mode():
                vout = recon.run()

            points_ply = vout.get("points_ply", os.path.join(OUTPUT_VGGT_DIR, "points.ply"))
            cameras_json = vout.get("cameras_json", os.path.join(OUTPUT_VGGT_DIR, "cameras.json"))
            self.get_logger().info(f"[offline] VGGT done: points_ply={points_ply}, cameras_json={cameras_json}")
        except Exception as e:
            self.get_logger().error(f"[offline] VGGT reconstruction failed: {e}")
            self._inflight = False
            return

        # 2) Post-process + align to base_link
        obj_json = os.path.join(OUTPUT_VGGT_DIR, OUTPUT_OBJECT_JSON_NAME)
        try:
            self.get_logger().info("[offline] Running point cloud postprocess + alignment to base_link...")
            pcfg = PostConfig(
                ply_path=points_ply,
                vggt_cameras_json=cameras_json,
                robot_shots_json=self._js_path,
                out_dir=OUTPUT_VGGT_DIR,
                visualize=POSTPROCESS_VISUALIZE,
                export_object_json=True,
                object_json_name=OUTPUT_OBJECT_JSON_NAME,
                align_method=POST_ALIGN_METHOD,
                vggt_pose_is_world_T_cam=POST_VGGT_POSE_IS_WORLD_T_CAM,
            )
            pres = process_pointcloud(pcfg)

            obj_json = pres.get("outputs", {}).get("object_json", obj_json)
            center_b = pres.get("center_b", None)

            # Parse center
            center_np: Optional[np.ndarray] = None
            if center_b is not None:
                try:
                    center_np = np.asarray(center_b, dtype=float).reshape(3)
                except Exception:
                    center_np = None
            if center_np is None:
                center_np = self._read_center_from_object_json(obj_json)

            # Apply OFFLINE bias to exported coordinates (center + vertices)
            if bool(getattr(self.cfg, "offline_bias", None) and self.cfg.offline_bias.enable):
                ob = self.cfg.offline_bias
                offset = np.asarray([float(ob.ox), float(ob.oy), float(ob.oz)], dtype=float).reshape(3)

                # Write back into object JSON to keep downstream tools consistent
                changed = self._apply_offline_bias_to_object_json(obj_json, offset)
                if changed:
                    self.get_logger().info(
                        f"[offline] Applied offline_bias to object JSON: "
                        f"({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})"
                    )
                else:
                    self.get_logger().warn("[offline] Offline bias enabled, but object JSON was not modified (schema not matched).")

                # Re-read center from JSON to guarantee consistency
                center_np = self._read_center_from_object_json(obj_json)

            # Cache final point for the "final goto" step
            self._final_point_base = center_np

            if center_np is not None:
                self.get_logger().info(
                    f"[offline] Object center in base_link (after offline bias if enabled): "
                    f"({center_np[0]:.4f}, {center_np[1]:.4f}, {center_np[2]:.4f})"
                )
            else:
                self.get_logger().warn("[offline] Object center not found (parse failed).")

            self.get_logger().info(f"[offline] Postprocess done. Exported: {obj_json}")
        except Exception as e:
            self.get_logger().error(f"[offline] Postprocess/alignment failed: {e}")
        finally:
            self._inflight = False

    # ---------- Final goto: request IK and move to hover above object ----------
    def _request_final_goto(self):
        if not FINAL_GOTO_ENABLE:
            return
        if self._final_goto_requested:
            return
        if self._final_point_base is None:
            self.get_logger().warn("[final] No object center available; skipping final goto.")
            self._final_goto_requested = True
            return
        if not self.ik.ready():
            return

        self._final_pose = self._build_final_hover_pose(self._final_point_base)
        seed = self.motion.make_seed()
        if seed is None:
            self.get_logger().warn("[final] Waiting for /joint_states seed...")
            return

        self._final_goto_requested = True
        self._inflight = True
        self.get_logger().info(
            f"[final] Request IK for hover above object: "
            f"({self._final_pose.pose.position.x:.3f}, "
            f"{self._final_pose.pose.position.y:.3f}, "
            f"{self._final_pose.pose.position.z:.3f})"
        )
        self.ik.request_async(self._final_pose, seed, self._on_final_ik)

    def _on_final_ik(self, joint_positions: Optional[List[float]]):
        self._inflight = False
        if joint_positions is None:
            self.get_logger().warn("[final] IK failed/skipped for final goto. Exiting.")
            self._phase = "shutdown"
            return

        self.traj.publish_positions(joint_positions, float(FINAL_GOTO_MOVE_TIME))
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(float(FINAL_GOTO_MOVE_TIME), 0.3)
        self._phase = "final_moving"

    # ---------- FSM ----------
    def _tick(self):
        if self._done:
            return

        # Prevent control FSM from running while IK/offline is inflight
        if self._inflight and self._phase not in ("offline_pipeline",):
            return

        if self._phase == "init_needed":
            self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
            self.motion.set_seed_hint(self.cfg.control.init_pos)
            self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
            self._phase = "init_moving"
            return

        if self._phase == "init_moving":
            if self.motion.is_stationary():
                self._phase = "wait_detect_stage1"
                self.get_logger().info("INIT reached. Waiting for Stage-1 detection (XY only)...")
            return

        # Stage-1 move (XY only, keep Z)
        if self._phase == "wait_detect_stage1" and self._pose_stage1 is not None:
            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                self.get_logger().warn("Waiting for /joint_states ...")
                return
            self._inflight = True
            self.ik.request_async(self._pose_stage1, seed, self._on_ik_stage1)
            return

        if self._phase == "stage1_moving":
            if self.motion.is_stationary():
                self._pose_stage1 = None
                self._phase = "wait_detect_stage2"
                self.get_logger().info("Stage-1 done. Waiting for Stage-2 detection (XY + descend)...")
            return

        # Stage-2 move (hover over center)
        if self._phase == "wait_detect_stage2" and self._fixed_hover is not None:
            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                self.get_logger().warn("Waiting for /joint_states ...")
                return
            self._inflight = True
            self.ik.request_async(self._fixed_hover, seed, self._on_hover_ik)
            return

        if self._phase == "hover_to_center":
            if not self.motion.is_stationary():
                return

            # At hover; compute start yaw and build polygon vertices
            self._start_yaw = self._get_tool_yaw_xy()
            if self._circle_center is not None and self._ring_z is not None and self._start_yaw is not None:
                # 1) Generate full vertices (potentially 360deg * num_turns)
                all_wps = make_polygon_vertices(
                    self.get_clock().now().to_msg,
                    self._circle_center, self._ring_z, self._start_yaw,
                    self.cfg.frames.pose_frame,
                    self.cfg.circle.n_vertices, self.cfg.circle.num_turns, self.cfg.circle.poly_dir,
                    self.cfg.circle.orient_mode, self.cfg.circle.start_dir_offset_deg,
                    self.cfg.circle.radius, self.cfg.circle.tool_z_sign
                )

                # 2) Trim vertices according to sweep_deg to avoid hitting joint limits
                total_deg = 360.0 * float(self.cfg.circle.num_turns)
                sweep_deg = max(0.0, min(float(self._sweep_deg), total_deg))
                if sweep_deg < total_deg and len(all_wps) > 1:
                    keep = max(1, int(math.floor(len(all_wps) * (sweep_deg / total_deg))))
                    keep = min(keep, len(all_wps))
                    self._poly_wps = all_wps[:keep]
                    self.get_logger().info(
                        f"Generated vertices: {len(all_wps)} -> trimmed to {len(self._poly_wps)} "
                        f"for sweep {sweep_deg:.1f}deg / {total_deg:.1f}deg."
                    )
                else:
                    self._poly_wps = all_wps
                    self.get_logger().info(f"Generated vertices: {len(self._poly_wps)} (full sweep).")

                self._poly_idx = 0
                self._phase = "poly_prepare"
            else:
                self._phase = "return_init"
            return

        if self._phase == "poly_prepare":
            if not self._poly_wps:
                self._phase = "return_init"
                return
            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                return
            self._inflight = True
            self.ik.request_async(self._poly_wps[0], seed, self._on_poly_ik)
            self._poly_idx = 1
            self._phase = "poly_moving"
            return

        if self._phase == "poly_moving":
            # If the last IK was skipped due to abnormal jump, move to the next vertex without dwell
            if self._skip_last_vertex:
                self._skip_last_vertex = False
                if self._poly_idx >= len(self._poly_wps):
                    self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
                    self.motion.set_seed_hint(self.cfg.control.init_pos)
                    self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
                    self._phase = "return_init"
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

            # Dwell at current vertex -> atomic capture
            if self._poly_dwell_due_ns is None:
                self._poly_dwell_due_ns = now + int(self.cfg.circle.dwell_time * 1e9)
                curr_vertex0 = max(0, self._poly_idx - 1)
                self.get_logger().info(f"At vertex {curr_vertex0 + 1}, dwell for capture...")

                # Atomic capture (image + pose at exact image stamp). No fallback.
                self._capture_and_log_shot(curr_vertex0)

                self._at_last_vertex = (self._poly_idx >= len(self._poly_wps))
                return

            if now < self._poly_dwell_due_ns:
                return
            self._poly_dwell_due_ns = None

            # If already at the last point -> return to INIT
            if getattr(self, "_at_last_vertex", False):
                self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
                self.motion.set_seed_hint(self.cfg.control.init_pos)
                self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
                self._phase = "return_init"
                return

            if self._poly_idx >= len(self._poly_wps):
                self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
                self.motion.set_seed_hint(self.cfg.control.init_pos)
                self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
                self._phase = "return_init"
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

        # After returning to INIT, run offline pipeline and then optionally do final goto
        if self._phase == "return_init":
            if not self.motion.is_stationary():
                return

            if RUN_OFFLINE_PIPELINE and not self._offline_ran:
                self._phase = "offline_pipeline"
                self.get_logger().info("Capture finished and INIT reached. Starting offline pipeline (VGGT + postprocess)...")
                self._run_offline_pipeline_once()

                if FINAL_GOTO_ENABLE:
                    self._phase = "final_goto_needed"
                else:
                    self._phase = "shutdown"
                return

            self._phase = "shutdown"
            return

        if self._phase == "final_goto_needed":
            if not self.motion.is_stationary():
                return
            self._request_final_goto()
            if self._final_goto_requested and self._inflight:
                return
            if self._final_goto_requested and (self._final_point_base is None):
                self._phase = "shutdown"
            return

        if self._phase == "final_moving":
            if not self.motion.is_stationary():
                return
            self.get_logger().info("[final] Final hover reached. Exiting.")
            self._phase = "shutdown"
            return

        if self._phase == "shutdown":
            self._done = True
            self.get_logger().info("All done. Exiting.")
            rclpy.shutdown()
            return

    # ---------- IK callbacks ----------
    def _on_ik_stage1(self, joint_positions: Optional[List[float]]):
        self._inflight = False
        if joint_positions is None:
            self.get_logger().warn("Stage-1 IK skipped/failed. Returning to INIT and exiting.")
            self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
            self.motion.set_seed_hint(self.cfg.control.init_pos)
            self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
            self._phase = "return_init"
            return

        self.traj.publish_positions(joint_positions, self.cfg.control.move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.control.move_time, 0.3)
        self._phase = "stage1_moving"

    def _on_hover_ik(self, joint_positions: Optional[List[float]]):
        self._inflight = False
        if joint_positions is None:
            self.get_logger().warn("Stage-2 hover IK skipped/failed. Returning to INIT and exiting.")
            self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
            self.motion.set_seed_hint(self.cfg.control.init_pos)
            self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
            self._phase = "return_init"
            return

        self.traj.publish_positions(joint_positions, self.cfg.control.move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.control.move_time, 0.3)
        self._phase = "hover_to_center"

    def _on_poly_ik(self, joint_positions: Optional[List[float]]):
        if joint_positions is None:
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


if __name__ == "__main__":
    main()
