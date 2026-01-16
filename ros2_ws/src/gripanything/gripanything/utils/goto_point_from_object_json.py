#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
from typing import List, Optional, Dict, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    # x-y-z intrinsic (roll, pitch, yaw)
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


def _duration_from_sec(sec: float) -> Duration:
    sec_i = int(sec)
    nsec = int((sec - sec_i) * 1e9)
    return Duration(sec=sec_i, nanosec=nsec)


class GotoPoint(Node):
    def __init__(self):
        super().__init__("goto_point_from_object_json")

        # ---------------- Params ----------------
        self.declare_parameter("object_json", "")
        self.declare_parameter("use_point", "center")  # "center" or "corner"
        self.declare_parameter("corner_index", 0)      # 0..7
        self.declare_parameter("z_offset", 0.15)       # meters
        self.declare_parameter("z_min", 0.05)          # meters, safety clamp
        self.declare_parameter("z_max", 2.00)          # meters

        # IK / group config (按你 MoveIt 配置改)
        self.declare_parameter("ik_service", "/compute_ik")  # 有些系统是 /move_group/compute_ik
        self.declare_parameter("move_group", "ur_manipulator")
        self.declare_parameter("ik_frame", "base_link")
        self.declare_parameter("eef_link", "tool0")  # 你的 MoveIt eef link

        # Controller topic
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

        # Orientation: 默认“向下”姿态；不对就改
        self.declare_parameter("target_rpy", [math.pi, 0.0, 0.0])  # roll, pitch, yaw

        # IK behavior
        self.declare_parameter("ik_timeout", 2.0)
        self.declare_parameter("avoid_collisions", False)  # True 可能要求 planning_scene 更完整
        self.declare_parameter("exit_after_publish", True)

        # ---------------- IO ----------------
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            self.get_parameter("traj_topic").value,
            10
        )

        self.ik_cli = self.create_client(GetPositionIK, self.get_parameter("ik_service").value)
        if not self.ik_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"IK service not available: {self.get_parameter('ik_service').value}")

        self.last_joint_state: Optional[JointState] = None
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, qos)

        # do once
        self.done = False
        self._ik_future = None
        self.timer = self.create_timer(0.2, self._run_once)

    def _on_joint_state(self, msg: JointState):
        self.last_joint_state = msg

    def _read_target_point(self) -> List[float]:
        path = self.get_parameter("object_json").value
        if not path or not os.path.exists(path):
            raise RuntimeError(f"object_json not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        obj = data.get("object", {})
        use_point = self.get_parameter("use_point").value

        if use_point == "center":
            p = obj.get("center", {}).get("base_link", None)
            if p is None:
                # 兼容旧结构
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

            idx = int(self.get_parameter("corner_index").value)
            idx = max(0, min(7, idx))
            p = corners[idx]
            return [float(p[0]), float(p[1]), float(p[2])]

        raise RuntimeError(f"Unknown use_point: {use_point}")

    def _publish_trajectory(self, joint_state: JointState):
        name_to_pos = {n: p for n, p in zip(joint_state.name, joint_state.position)}
        joint_names: List[str] = list(self.get_parameter("joint_names").value)

        missing = [n for n in joint_names if n not in name_to_pos]
        if missing:
            raise RuntimeError(f"IK solution missing joints: {missing}")

        traj = JointTrajectory()
        traj.joint_names = joint_names

        pt = JointTrajectoryPoint()
        pt.positions = [float(name_to_pos[n]) for n in joint_names]
        T = float(self.get_parameter("time_from_start").value)
        pt.time_from_start = _duration_from_sec(T)

        traj.points = [pt]
        self.traj_pub.publish(traj)

    def _request_ik_async(self, pose: PoseStamped):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.get_parameter("move_group").value
        req.ik_request.pose_stamped = pose
        req.ik_request.ik_link_name = self.get_parameter("eef_link").value
        req.ik_request.avoid_collisions = bool(self.get_parameter("avoid_collisions").value)

        # 关键：seed（参考你 ik_and_traj.py）
        if self.last_joint_state is not None:
            req.ik_request.robot_state.joint_state = self.last_joint_state

        timeout_s = float(self.get_parameter("ik_timeout").value)
        req.ik_request.timeout = _duration_from_sec(timeout_s)

        self._ik_future = self.ik_cli.call_async(req)
        self._ik_future.add_done_callback(self._on_ik_done)

    def _on_ik_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"IK call raised: {e}")
            self._finish()
            return

        if res is None:
            self.get_logger().error("IK failed (no response).")
            self._finish()
            return

        if res.error_code.val != res.error_code.SUCCESS:
            self.get_logger().error(f"IK failed, error_code={res.error_code.val}")
            self._finish()
            return

        self.get_logger().info("IK success. Publishing trajectory...")
        try:
            self._publish_trajectory(res.solution.joint_state)
        except Exception as e:
            self.get_logger().error(f"Publish trajectory failed: {e}")
            self._finish()
            return

        self.get_logger().info("Done. Robot should move to hover above the chosen point.")
        self._finish()

    def _finish(self):
        if self.done:
            return
        self.done = True

        if bool(self.get_parameter("exit_after_publish").value):
            # 让 ros2 run 自动结束
            self.get_logger().info("Exiting node.")
            rclpy.shutdown()

    def _run_once(self):
        if self.done:
            return
        if self._ik_future is not None:
            return  # 已经发出 IK 请求了

        # 等 /joint_states 更稳（seed）
        if self.last_joint_state is None:
            self.get_logger().info("Waiting for /joint_states ...")
            return

        # 1) load target point
        p = self._read_target_point()
        z_offset = float(self.get_parameter("z_offset").value)
        z_min = float(self.get_parameter("z_min").value)
        z_max = float(self.get_parameter("z_max").value)

        x, y, z = p
        z_hover = z + z_offset
        z_hover = max(z_min, min(z_max, z_hover))

        self.get_logger().info(
            f"Target point (base_link): raw={p}, then hover_z={z_hover:.4f} (z_offset={z_offset})"
        )

        # 2) build pose
        pose = PoseStamped()
        pose.header.frame_id = self.get_parameter("ik_frame").value
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z_hover)

        rpy = self.get_parameter("target_rpy").value
        qx, qy, qz, qw = quat_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        # 3) IK async
        self._request_ik_async(pose)


def main():
    rclpy.init()
    node = GotoPoint()
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
