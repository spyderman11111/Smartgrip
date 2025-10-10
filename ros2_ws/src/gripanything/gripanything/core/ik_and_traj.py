#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ik_and_traj.py — MoveIt IK client, joint_state tracking, and trajectory publishing.

Purpose:
- Encapsulate /joint_states handling and "is stationary" checks.
- Provide a robust async IK request wrapper with jump guards (name-aligned reference).
- Publish JointTrajectory messages for init and point-to-point moves.

Changes (2025-10-07):
- IK reference angles are matched by joint NAME (prefer seed.joint_state), not by list order.
- When a large/abnormal jump is detected, log the JOINT NAME and Δ (ref -> target), and call done(None).
"""

from typing import List, Optional, Callable, Tuple
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from builtin_interfaces.msg import Duration as MsgDuration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped

from .tf_ops import wrap_to_near


class MotionContext:
    """Subscribe to /joint_states, provide stationary detection and seed-hint storage."""
    def __init__(self, node, joint_order: List[str], vel_eps: float, require_stationary: bool):
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE)
        self._node = node
        self._last_js: Optional[JointState] = None
        self._seed_hint: Optional[np.ndarray] = None
        self._motion_due_ns: Optional[int] = None
        self._joint_order = list(joint_order)
        self._vel_eps = float(vel_eps)
        self._require_stationary = bool(require_stationary)
        self._node.create_subscription(JointState, '/joint_states', self._on_js, qos)

    def _on_js(self, msg: JointState):
        if self._last_js is None:
            self._node.get_logger().info(f"Received /joint_states ({len(msg.name)} joints).")
        self._last_js = msg

    def set_motion_due(self, seconds: float, extra_wait: float = 0.3):
        now = self._node.get_clock().now().nanoseconds
        self._motion_due_ns = now + int((seconds + extra_wait) * 1e9)

    def is_stationary(self) -> bool:
        now_ns = self._node.get_clock().now().nanoseconds
        if self._motion_due_ns is not None and now_ns < self._motion_due_ns:
            return False
        if not self._require_stationary:
            return True
        if self._last_js is None or not self._last_js.velocity:
            return True
        try:
            return all(abs(float(v)) <= self._vel_eps for v in self._last_js.velocity)
        except Exception:
            return True

    def make_seed(self) -> Optional[JointState]:
        """Prefer the last seed hint we published; otherwise fall back to /joint_states."""
        if self._seed_hint is not None:
            js = JointState()
            js.name = self._joint_order
            js.position = [float(a) for a in self._seed_hint]
            return js
        return self._last_js

    def set_seed_hint(self, positions: List[float]):
        self._seed_hint = np.array(positions, dtype=float)


class TrajectoryPublisher:
    """Minimal wrapper to publish JointTrajectory commands."""
    def __init__(self, node, joint_order: List[str], controller_topic: str):
        self._node = node
        self._joint_order = list(joint_order)
        self._pub = node.create_publisher(JointTrajectory, controller_topic, 1)

    def publish_init(self, init_positions: List[float], move_time: float):
        traj = JointTrajectory()
        traj.joint_names = self._joint_order
        pt = JointTrajectoryPoint()
        pt.positions = [float(a) for a in init_positions]
        pt.time_from_start = MsgDuration(sec=int(move_time), nanosec=int((move_time % 1.0) * 1e9))
        traj.points = [pt]
        self._pub.publish(traj)
        self._node.get_logger().info(f"Published INIT joint pose (T={move_time:.1f}s).")

    def publish_positions(self, positions: List[float], move_time: float):
        traj = JointTrajectory()
        traj.joint_names = self._joint_order
        pt = JointTrajectoryPoint()
        pt.positions = [float(a) for a in positions]
        pt.time_from_start = MsgDuration(sec=int(move_time), nanosec=int((move_time % 1.0) * 1e9))
        traj.points = [pt]
        self._pub.publish(traj)
        self._node.get_logger().info(
            "Published joint goal: [" + ", ".join(f"{v:.6f}" for v in positions) + f"], T={move_time:.1f}s"
        )


class IKClient:
    """Async MoveIt IK client with jump-guarding and reference-angle wrapping (name-aligned)."""
    def __init__(self, node, group_name: str, ik_link_name: str, joint_order: List[str],
                 timeout_s: float, ignore_joints: List[str], max_safe_jump: float, max_warn_jump: float):
        self._node = node
        self._group = group_name
        self._link = ik_link_name
        self._joint_order = list(joint_order)
        self._timeout = float(timeout_s)
        self._ignore = set(ignore_joints)
        self._max_safe = float(max_safe_jump)
        self._max_warn = float(max_warn_jump)
        self._client = None
        self._candidates = ['/compute_ik', '/move_group/compute_ik']
        self._warned = False

    def ready(self) -> bool:
        if self._client:
            return True
        for name in self._candidates:
            cli = self._node.create_client(GetPositionIK, name)
        # noqa: E701
            if cli.wait_for_service(timeout_sec=0.5):
                self._client = cli
                self._node.get_logger().info(f"IK service available: {name}")
                return True
        if not self._warned:
            self._warned = True
            self._node.get_logger().warn(f"Waiting for IK service… (tried: {self._candidates})")
        return False

    def request_async(
        self,
        pose: PoseStamped,
        seed: JointState,
        done: Callable[[Optional[List[float]]], None],
        ref_positions: Optional[List[float]] = None,
    ):
        """
        Asynchronously request IK. Reference angles are matched by JOINT NAME:
        - Prefer seed.joint_state (name/position) as reference.
        - If absent, fall back to (self._joint_order, ref_positions).
        On abnormal jump (> max_warn), log and call done(None).
        """
        req = GetPositionIK.Request()
        req.ik_request.group_name = self._group
        req.ik_request.ik_link_name = self._link
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = False
        req.ik_request.robot_state.joint_state = seed
        req.ik_request.timeout = MsgDuration(
            sec=int(self._timeout),
            nanosec=int((self._timeout % 1.0) * 1e9),
        )

        fut = self._client.call_async(req)

        def _cb(f):
            try:
                res = f.result()
            except Exception as e:
                self._node.get_logger().error(f"IK call raised: {e}")
                done(None)
                return

            if res is None or res.error_code.val != 1:
                code = None if res is None else res.error_code.val
                self._node.get_logger().error(f"IK failed (error_code={code}).")
                done(None)
                return

            name_to_idx = {n: i for i, n in enumerate(res.solution.joint_state.name)}

            # Build reference-by-name
            ref_by_name = {}
            if seed and seed.name and seed.position:
                for n, p in zip(seed.name, seed.position):
                    ref_by_name[n] = float(p)
            elif ref_positions is not None:
                for n, a in zip(self._joint_order, ref_positions):
                    ref_by_name[n] = float(a)

            targets: List[float] = []
            jumps: List[Tuple[str, float, float, float]] = []  # (joint, delta, ref, near)
            missing = []

            for jn in self._joint_order:
                if jn not in name_to_idx:
                    missing.append(jn)
                    continue
                raw = float(res.solution.joint_state.position[name_to_idx[jn]])
                ref = ref_by_name.get(jn, raw)
                near = wrap_to_near(raw, ref)
                targets.append(near)
                if jn not in self._ignore:
                    jumps.append((jn, abs(near - ref), ref, near))

            if missing:
                self._node.get_logger().error(f"IK result missing joints: {missing}")
                done(None)
                return

            worst = max(jumps, key=lambda x: x[1]) if jumps else None
            if worst is not None:
                jn_w, dj, ref_w, near_w = worst
                if dj > self._max_warn:
                    self._node.get_logger().warn(
                        f"Abnormal joint jump: joint={jn_w}, Δ={dj:.3f} rad (> {self._max_warn:.3f}), "
                        f"ref={ref_w:.3f} -> target={near_w:.3f}. Skipped."
                    )
                    done(None)
                    return
                if dj > self._max_safe:
                    self._node.get_logger().warn(
                        f"Large joint jump: joint={jn_w}, Δ={dj:.3f} rad (> {self._max_safe:.3f}). Publishing anyway."
                    )

            done(targets)

        fut.add_done_callback(_cb)
