#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
goto_hover_once.py — ROS 2 Humble / UR5e

一次性动作：
1) TF 读取 object_position → 计算 hover 位姿（+Z 提升 hover_above）
2) 末端朝下 (RotX(pi))，可选绕 Z 加 yaw
3) 异步调用 /compute_ik 求解
4) 发布 JointTrajectory 到 scaled_joint_trajectory_controller

注意：
- 不在定时器里阻塞等待 IK；使用 call_async + 回调，避免“卡自己导致超时”的问题。
"""

import math
from typing import Optional, Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration as RclDuration
from rclpy.time import Time

import tf2_ros
from tf2_ros import TransformException

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from moveit_msgs.srv import GetPositionIK


UR5E_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


# ---------- quaternion helpers ----------
def quat_mul(q1, q2):
    """Hamilton product, (x,y,z,w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    )


def quat_from_yaw(yaw_rad: float):
    s, c = math.sin(0.5 * yaw_rad), math.cos(0.5 * yaw_rad)
    return (0.0, 0.0, s, c)  # (x,y,z,w)


class GoToHoverOnce(Node):
    def __init__(self):
        super().__init__("goto_hover_once")

        # ---------- 参数 ----------
        self.declare_parameter("object_frame", "object_position")
        self.declare_parameter("pose_frame", "base_link")      # MoveIt 规划坐标系（常为 base_link）
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("ik_link_name", "tool0")

        self.declare_parameter("hover_above", 0.2)            # 提升高度（米）
        self.declare_parameter("yaw_deg", 0.0)                 # 绕 +Z 的 yaw（度）

        self.declare_parameter("ik_service", "")               # 留空自动选择
        self.declare_parameter("ik_timeout", 2.0)              # 秒（给足 2s）

        self.declare_parameter("controller_topic", "/scaled_joint_trajectory_controller/joint_trajectory")
        self.declare_parameter("move_time", 3.0)               # 秒

        self.declare_parameter("require_js", True)
        self.declare_parameter("js_wait_timeout", 2.0)
        self.declare_parameter("fallback_zero_if_timeout", False)
        self.declare_parameter("zero_seed", [0, 0, 0, 0, 0, 0])
        self.declare_parameter("js_reliability", "reliable")   # reliable / best_effort

        # 读取参数
        self.object_frame = str(self.get_parameter("object_frame").value)
        self.pose_frame = str(self.get_parameter("pose_frame").value)
        self.group_name = str(self.get_parameter("group_name").value)
        self.ik_link_name = str(self.get_parameter("ik_link_name").value)

        self.hover_above = float(self.get_parameter("hover_above").value)
        self.yaw_deg = float(self.get_parameter("yaw_deg").value)

        self.ik_service_param = str(self.get_parameter("ik_service").value or "")
        self.ik_timeout = float(self.get_parameter("ik_timeout").value)

        self.controller_topic = str(self.get_parameter("controller_topic").value)
        self.move_time = float(self.get_parameter("move_time").value)

        self.require_js = bool(self.get_parameter("require_js").value)
        self.js_wait_timeout = float(self.get_parameter("js_wait_timeout").value)
        self.fallback_zero_if_timeout = bool(self.get_parameter("fallback_zero_if_timeout").value)
        self.zero_seed = [float(v) for v in self.get_parameter("zero_seed").value]
        self.js_reliability = str(self.get_parameter("js_reliability").value).strip().lower()

        # ---------- 通信 ----------
        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # /joint_states
        if self.js_reliability == "best_effort":
            qos_js = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
            )
        else:
            qos_js = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
            )
        self._last_js: Optional[JointState] = None
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_js)

        # 轨迹发布
        self.pub_traj = self.create_publisher(JointTrajectory, self.controller_topic, 1)

        # IK 客户端
        self._ik_client = None
        self._ik_service_name = None
        self._ik_candidates = ["/compute_ik", "/move_group/compute_ik"]

        # 状态
        self._warned = set()
        self._deadline_js: Optional[int] = None
        self._inflight = False
        self._done = False

        self.get_logger().info(
            f"goto_hover_once 启动: pose_frame={self.pose_frame}, object_frame={self.object_frame}, "
            f"hover={self.hover_above:.3f} m, move_time={self.move_time:.1f}s"
        )

        self.create_timer(0.2, self._tick)

    # ---------- 回调 ----------
    def _on_js(self, msg: JointState):
        if self._last_js is None:
            self.get_logger().info(f"已收到 /joint_states（{len(msg.name)} 关节）。")
        self._last_js = msg

    # ---------- 工具 ----------
    def _warn_once(self, key: str, text: str):
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warning(text)

    def _ensure_ik_client(self) -> bool:
        if self._ik_client:
            return True
        names = [self.ik_service_param] if self.ik_service_param else self._ik_candidates
        for name in names:
            if not name:
                continue
            cli = self.create_client(GetPositionIK, name)
            if cli.wait_for_service(timeout_sec=0.5):
                self._ik_client = cli
                self._ik_service_name = name
                self.get_logger().info(f"IK 服务可用：{name}")
                return True
        self._warn_once("wait_ik", f"等待 IK 服务…（尝试：{names}）")
        return False

    def _get_seed(self) -> Optional[JointState]:
        if self._last_js:
            return self._last_js
        now_ns = self.get_clock().now().nanoseconds
        if self._deadline_js is None:
            self._deadline_js = now_ns + int(self.js_wait_timeout * 1e9)
            self._warn_once("wait_js", "等待 /joint_states …")
            return None
        if now_ns < self._deadline_js:
            return None
        if not self.fallback_zero_if_timeout and self.require_js:
            self._warn_once("js_timeout", "等待 /joint_states 超时，仍在等待（可设置 fallback_zero_if_timeout:=true 使用零位种子）。")
            return None
        # 使用零位种子
        self._warn_once("use_zero_seed", "使用零位种子。")
        js = JointState()
        js.name = UR5E_JOINT_ORDER.copy()
        js.position = self.zero_seed[:6]
        js.header.stamp = self.get_clock().now().to_msg()
        return js

    def _lookup_hover_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self.tf_buffer.lookup_transform(self.pose_frame, self.object_frame, Time(),
                                                 timeout=RclDuration(seconds=0.5))
        except TransformException as ex:
            self._warn_once("tf_object", f"TF 未就绪：{self.pose_frame} <- {self.object_frame} ：{ex}")
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        z = tf.transform.translation.z + self.hover_above

        # 末端朝下（RotX(pi)），再绕 Z 加 yaw
        q_down = (1.0, 0.0, 0.0, 0.0)
        q_yaw = quat_from_yaw(math.radians(self.yaw_deg))
        q = quat_mul(q_yaw, q_down)

        ps = PoseStamped()
        ps.header.frame_id = self.pose_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        return ps

    # ---------- IK 异步调用 ----------
    def _request_ik(self, pose: PoseStamped, seed: JointState):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ik_link_name
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = False
        req.ik_request.robot_state.joint_state = seed
        req.ik_request.timeout = Duration(
            sec=int(self.ik_timeout),
            nanosec=int((self.ik_timeout % 1.0) * 1e9),
        )

        self._inflight = True
        fut = self._ik_client.call_async(req)
        fut.add_done_callback(self._on_ik_done)

    def _on_ik_done(self, fut):
        self._inflight = False
        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"IK 调用异常：{e}")
            return
        if res is None:
            self.get_logger().error("IK 调用失败（无返回）。")
            return
        if res.error_code.val != 1:
            self.get_logger().error(f"IK 未找到解，error_code={res.error_code.val}")
            return

        # 将返回关节顺序映射到 UR5E_JOINT_ORDER
        name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(res.solution.joint_state.name)}
        target_positions: List[float] = []
        missing = []
        for jn in UR5E_JOINT_ORDER:
            if jn not in name_to_idx:
                missing.append(jn)
            else:
                target_positions.append(res.solution.joint_state.position[name_to_idx[jn]])
        if missing:
            self.get_logger().error(f"IK 结果缺少关节: {missing}")
            return

        # 发布关节轨迹（单点）
        traj = JointTrajectory()
        traj.joint_names = UR5E_JOINT_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = target_positions
        pt.time_from_start = Duration(
            sec=int(self.move_time),
            nanosec=int((self.move_time % 1.0) * 1e9)
        )
        traj.points = [pt]
        self.pub_traj.publish(traj)

        self.get_logger().info(
            "已发布关节目标（平滑到位）："
            + "[" + ", ".join(f"{v:.6f}" for v in target_positions) + f"], T={self.move_time:.1f}s"
        )
        self._done = True
        self.create_timer(0.5, self._shutdown_once)

    # ---------- 主循环 ----------
    def _tick(self):
        if self._done or self._inflight:
            return
        if not self._ensure_ik_client():
            return
        seed = self._get_seed()
        if seed is None:
            return
        pose = self._lookup_hover_pose()
        if pose is None:
            return

        self.get_logger().info(
            f"目标位姿: frame={pose.header.frame_id}, "
            f"p=({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},{pose.pose.position.z:.3f}), "
            f"yaw={self.yaw_deg:.1f}°, hover={self.hover_above:.3f}m"
        )
        self._request_ik(pose, seed)

    # ---------- 收尾 ----------
    def _shutdown_once(self):
        self.get_logger().info("goto_hover_once 完成，退出。")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = GoToHoverOnce()
    try:
        rclpy.spin(node)  # 单线程执行器即可；我们不在回调里阻塞
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
