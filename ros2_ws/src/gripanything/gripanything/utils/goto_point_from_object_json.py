#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
goto_point_from_object_json.py

Reusable "go-to-point" helper for UR robots via MoveIt IK + JointTrajectory.

Primary use cases:
1) Standalone tool:
   ros2 run gripanything goto_point_from_object_json --ros-args -p object_json:=... -p z_offset:=0.20

2) Import and call from another ROS2 node (e.g., seeanything.py):
   from gripanything.utils.goto_point_from_object_json import GotoPointConfig, GotoPointRunner
   runner = GotoPointRunner(node, GotoPointConfig(object_json="/path/to/object_in_base_link.json", z_offset=0.20))
   runner.start()  # async; publishes a trajectory once IK returns
"""

import os
import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState


# ------------------------- Math helpers -------------------------
def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Quaternion (x,y,z,w) from intrinsic XYZ (roll, pitch, yaw)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def duration_from_sec(sec: float) -> Duration:
    sec_i = int(sec)
    nsec = int(max(0.0, sec - sec_i) * 1e9)
    return Duration(sec=sec_i, nanosec=nsec)


# ------------------------- JSON loader -------------------------
def read_target_point_from_object_json(
    object_json_path: str,
    use_point: str = "center",      # "center" or "corner"
    corner_index: int = 0,
) -> List[float]:
    """
    Read a target point from your exported object_in_base_link.json.

    Supported layouts (compatible with your previous versions):
    - center: object.center.base_link or center_base
    - corners: object.obb.corners_8.base_link or corners_base
    """
    if not object_json_path or not os.path.exists(object_json_path):
        raise RuntimeError(f"object_json not found: {object_json_path}")

    with open(object_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    obj = data.get("object", {})

    if use_point == "center":
        p = obj.get("center", {}).get("base_link", None)
        if p is None:
            p = obj.get("center_base", None)
        if p is None:
            raise RuntimeError("Cannot find center in json (object.center.base_link or center_base).")
        return [float(p[0]), float(p[1]), float(p[2])]

    if use_point == "corner":
        corners = None
        if "obb" in obj and "corners_8" in obj["obb"] and "base_link" in obj["obb"]["corners_8"]:
            corners = obj["obb"]["corners_8"]["base_link"]
        elif "corners_base" in obj:
            corners = obj["corners_base"]

        if corners is None or len(corners) != 8:
            raise RuntimeError("Cannot find corners (expect 8) in json.")

        idx = int(corner_index)
        idx = max(0, min(7, idx))
        p = corners[idx]
        return [float(p[0]), float(p[1]), float(p[2])]

    raise RuntimeError(f"Unknown use_point: {use_point}")


# ------------------------- Config -------------------------
@dataclass
class GotoPointConfig:
    # Input
    object_json: str = ""

    # Which point to use from the json
    use_point: str = "center"     # "center" or "corner"
    corner_index: int = 0         # used when use_point="corner"

    # Hover height above the selected point (default: 0.20 m)
    z_offset: float = 0.20
    z_min: float = 0.05
    z_max: float = 2.00

    # MoveIt IK settings
    ik_service: str = "/compute_ik"          # sometimes "/move_group/compute_ik"
    move_group: str = "ur_manipulator"
    ik_frame: str = "base_link"
    eef_link: str = "tool0"
    ik_timeout: float = 2.0
    avoid_collisions: bool = False

    # Controller topic
    traj_topic: str = "/scaled_joint_trajectory_controller/joint_trajectory"
    joint_names: List[str] = field(default_factory=lambda: [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ])
    time_from_start: float = 3.0

    # End-effector orientation target (roll, pitch, yaw)
    # Default: pointing "down" (may need adjustment for your URDF / tool convention)
    target_rpy: List[float] = field(default_factory=lambda: [math.pi, 0.0, 0.0])

    # If set, runner will wait for /joint_states and use it as an IK seed
    require_joint_states_seed: bool = True


# ------------------------- Runner (import-friendly) -------------------------
class GotoPointRunner:
    """
    Import-friendly, non-blocking runner.

    Typical usage in another node:
        runner = GotoPointRunner(node, cfg)
        runner.start(on_done=...)
    """

    def __init__(self, node: Node, cfg: GotoPointConfig):
        self.node = node
        self.cfg = cfg

        self._traj_pub = self.node.create_publisher(JointTrajectory, self.cfg.traj_topic, 10)
        self._ik_cli = self.node.create_client(GetPositionIK, self.cfg.ik_service)

        self._last_joint_state: Optional[JointState] = None
        self._ik_future = None
        self._done = False

        # Optional subscription for IK seed
        if self.cfg.require_joint_states_seed:
            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.node.create_subscription(JointState, "/joint_states", self._on_joint_state, qos)

    @property
    def done(self) -> bool:
        return self._done

    def _on_joint_state(self, msg: JointState):
        self._last_joint_state = msg

    def _publish_trajectory(self, joint_state: JointState):
        """Publish a JointTrajectory with the IK solution, ordered by cfg.joint_names."""
        name_to_pos = {n: p for n, p in zip(joint_state.name, joint_state.position)}
        missing = [n for n in self.cfg.joint_names if n not in name_to_pos]
        if missing:
            raise RuntimeError(f"IK solution missing joints: {missing}")

        traj = JointTrajectory()
        traj.joint_names = list(self.cfg.joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = [float(name_to_pos[n]) for n in self.cfg.joint_names]
        pt.time_from_start = duration_from_sec(float(self.cfg.time_from_start))

        traj.points = [pt]
        self._traj_pub.publish(traj)

    def _build_target_pose(self) -> PoseStamped:
        """Load point from json, apply hover z_offset, and create a PoseStamped for IK."""
        p = read_target_point_from_object_json(
            self.cfg.object_json,
            use_point=self.cfg.use_point,
            corner_index=self.cfg.corner_index,
        )

        x, y, z = float(p[0]), float(p[1]), float(p[2])
        z_hover = z + float(self.cfg.z_offset)
        z_hover = max(float(self.cfg.z_min), min(float(self.cfg.z_max), z_hover))

        self.node.get_logger().info(
            f"[goto_point] target(base_link) raw=({x:.4f},{y:.4f},{z:.4f}) -> hover_z={z_hover:.4f} (z_offset={self.cfg.z_offset:.3f})"
        )

        pose = PoseStamped()
        pose.header.frame_id = self.cfg.ik_frame
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z_hover

        r, p_, y_ = float(self.cfg.target_rpy[0]), float(self.cfg.target_rpy[1]), float(self.cfg.target_rpy[2])
        qx, qy, qz, qw = quat_from_rpy(r, p_, y_)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def start(self, on_done: Optional[Callable[[bool, str], None]] = None) -> None:
        """
        Start an async IK request and publish trajectory on success.
        - on_done(ok, message) is called exactly once.
        """
        if self._done:
            if on_done is not None:
                on_done(True, "Already done.")
            return

        if self._ik_future is not None:
            return  # already running

        if self.cfg.require_joint_states_seed and self._last_joint_state is None:
            # Do not block; caller can call start() again later (or disable require_joint_states_seed).
            self.node.get_logger().info("[goto_point] Waiting for /joint_states to use as IK seed...")
            return

        if not self._ik_cli.wait_for_service(timeout_sec=0.1):
            self.node.get_logger().warn(f"[goto_point] IK service not available yet: {self.cfg.ik_service}")
            return

        pose = self._build_target_pose()

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.cfg.move_group
        req.ik_request.pose_stamped = pose
        req.ik_request.ik_link_name = self.cfg.eef_link
        req.ik_request.avoid_collisions = bool(self.cfg.avoid_collisions)
        req.ik_request.timeout = duration_from_sec(float(self.cfg.ik_timeout))

        # Seed with current robot state if available
        if self.cfg.require_joint_states_seed and self._last_joint_state is not None:
            req.ik_request.robot_state.joint_state = self._last_joint_state

        self._ik_future = self._ik_cli.call_async(req)

        def _cb(fut):
            self._ik_future = None
            try:
                res = fut.result()
            except Exception as e:
                msg = f"IK call raised: {e}"
                self.node.get_logger().error(f"[goto_point] {msg}")
                self._done = True
                if on_done is not None:
                    on_done(False, msg)
                return

            if res is None:
                msg = "IK failed (no response)."
                self.node.get_logger().error(f"[goto_point] {msg}")
                self._done = True
                if on_done is not None:
                    on_done(False, msg)
                return

            if res.error_code.val != res.error_code.SUCCESS:
                msg = f"IK failed, error_code={res.error_code.val}"
                self.node.get_logger().error(f"[goto_point] {msg}")
                self._done = True
                if on_done is not None:
                    on_done(False, msg)
                return

            try:
                self._publish_trajectory(res.solution.joint_state)
            except Exception as e:
                msg = f"Publish trajectory failed: {e}"
                self.node.get_logger().error(f"[goto_point] {msg}")
                self._done = True
                if on_done is not None:
                    on_done(False, msg)
                return

            msg = "OK: published trajectory to hover above the selected point."
            self.node.get_logger().info(f"[goto_point] {msg}")
            self._done = True
            if on_done is not None:
                on_done(True, msg)

        self._ik_future.add_done_callback(_cb)


# ------------------------- Standalone node wrapper -------------------------
class GotoPointNode(Node):
    """Standalone wrapper around GotoPointRunner, keeping your ros2-run usage."""
    def __init__(self):
        super().__init__("goto_point_from_object_json")

        # Parameters (keep compatible with your original script)
        self.declare_parameter("object_json", "")
        self.declare_parameter("use_point", "center")
        self.declare_parameter("corner_index", 0)
        self.declare_parameter("z_offset", 0.20)   # default changed to 0.20 m
        self.declare_parameter("z_min", 0.05)
        self.declare_parameter("z_max", 2.00)

        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("move_group", "ur_manipulator")
        self.declare_parameter("ik_frame", "base_link")
        self.declare_parameter("eef_link", "tool0")

        self.declare_parameter("traj_topic", "/scaled_joint_trajectory_controller/joint_trajectory")
        self.declare_parameter("joint_names", [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ])
        self.declare_parameter("time_from_start", 3.0)
        self.declare_parameter("target_rpy", [math.pi, 0.0, 0.0])

        self.declare_parameter("ik_timeout", 2.0)
        self.declare_parameter("avoid_collisions", False)
        self.declare_parameter("exit_after_publish", True)
        self.declare_parameter("require_joint_states_seed", True)

        cfg = GotoPointConfig(
            object_json=str(self.get_parameter("object_json").value),
            use_point=str(self.get_parameter("use_point").value),
            corner_index=int(self.get_parameter("corner_index").value),
            z_offset=float(self.get_parameter("z_offset").value),
            z_min=float(self.get_parameter("z_min").value),
            z_max=float(self.get_parameter("z_max").value),
            ik_service=str(self.get_parameter("ik_service").value),
            move_group=str(self.get_parameter("move_group").value),
            ik_frame=str(self.get_parameter("ik_frame").value),
            eef_link=str(self.get_parameter("eef_link").value),
            traj_topic=str(self.get_parameter("traj_topic").value),
            joint_names=list(self.get_parameter("joint_names").value),
            time_from_start=float(self.get_parameter("time_from_start").value),
            target_rpy=list(self.get_parameter("target_rpy").value),
            ik_timeout=float(self.get_parameter("ik_timeout").value),
            avoid_collisions=bool(self.get_parameter("avoid_collisions").value),
            require_joint_states_seed=bool(self.get_parameter("require_joint_states_seed").value),
        )

        self.runner = GotoPointRunner(self, cfg)

        # Polling timer: call runner.start() until it fires (it will wait for joint_states/service readiness)
        self._timer = self.create_timer(0.2, self._tick)

    def _tick(self):
        if self.runner.done:
            if bool(self.get_parameter("exit_after_publish").value):
                self.get_logger().info("Exiting node.")
                rclpy.shutdown()
            return

        self.runner.start(on_done=self._on_done)

    def _on_done(self, ok: bool, msg: str):
        if ok:
            self.get_logger().info(msg)
        else:
            self.get_logger().error(msg)


def main():
    rclpy.init()
    node = GotoPointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
