#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GoTo P_top（一次性）via MoveIt /compute_ik  — ROS 2 Humble
- 订阅 /p_top (PoseStamped)
- 调用 /compute_ik 求解关节位姿
- 预览：发布到 /display_planned_path (DisplayTrajectory)
- 执行：发布到 /scaled_joint_trajectory_controller/joint_trajectory (JointTrajectory)

说明：
- 不依赖 moveit_py / moveit_commander。
- 需要你的 MoveIt bringup 已启动（/compute_ik 存在），且 /joint_states 可用。
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState, DisplayTrajectory, RobotTrajectory

# UR5(e) 关节顺序（与 scaled_joint_trajectory_controller 一致）
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

class GoToPTopViaIK(Node):
    def __init__(self):
        super().__init__('goto_p_top_via_ik')

        # 可调参数
        self.declare_parameter('topic', '/p_top')                                   # 目标位姿话题
        self.declare_parameter('group_name', 'ur_manipulator')                      # MoveIt 规划组
        self.declare_parameter('eef_link', 'tool0')                                 # 末端链接
        self.declare_parameter('reference_frame', 'base_link')                      # 参考系
        self.declare_parameter('ik_timeout', 1.0)                                   # IK 超时(s)
        self.declare_parameter('move_time', 3.0)                                    # 执行总时长(s)
        self.declare_parameter('traj_topic', '/scaled_joint_trajectory_controller/joint_trajectory')
        self.declare_parameter('avoid_collisions', False)                           # IK 是否避碰
        self.declare_parameter('single_shot', True)                                 # 成功一次后退出
        # 预览/执行开关（均可独立打开）
        self.declare_parameter('preview', True)                                     # 是否发布 DisplayTrajectory
        self.declare_parameter('execute', False)                                    # 是否执行关节轨迹
        self.declare_parameter('preview_topic', '/display_planned_path')            # 预览话题

        # 读取参数
        self.topic = self.get_parameter('topic').value
        self.group = self.get_parameter('group_name').value
        self.eef = self.get_parameter('eef_link').value
        self.ref = self.get_parameter('reference_frame').value
        self.ik_timeout = float(self.get_parameter('ik_timeout').value)
        self.move_time = float(self.get_parameter('move_time').value)
        self.traj_topic = self.get_parameter('traj_topic').value
        self.avoid_collisions = bool(self.get_parameter('avoid_collisions').value)
        self.single_shot = bool(self.get_parameter('single_shot').value)
        self.preview = bool(self.get_parameter('preview').value)
        self.execute = bool(self.get_parameter('execute').value)
        self.preview_topic = self.get_parameter('preview_topic').value

        # 通信
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(PoseStamped, self.topic, self._on_pose, qos)
        self.create_subscription(JointState, '/joint_states', self._on_js, qos)
        self.traj_pub = self.create_publisher(JointTrajectory, self.traj_topic, 10)
        self.preview_pub = self.create_publisher(DisplayTrajectory, self.preview_topic, 10)
        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')

        self._js = {}   # 最新关节状态缓存
        self._done = False

        self.get_logger().info(
            f"节点已启动: 订阅 {self.topic}, 参考系={self.ref}, 组={self.group}, eef={self.eef}, "
            f"move_time={self.move_time}s, 避碰={self.avoid_collisions}, preview={self.preview}, execute={self.execute}"
        )

    def _on_js(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self._js[n] = float(p)

    def _on_pose(self, msg: PoseStamped):
        if self.single_shot and self._done:
            return

        if msg.header.frame_id and msg.header.frame_id != self.ref:
            self.get_logger().warning(
                f"收到 frame_id={msg.header.frame_id} 与 reference_frame={self.ref} 不一致；"
                f"本节点不做 TF 变换，请在上游保证一致。"
            )

        p = msg.pose.position
        q = msg.pose.orientation
        self.get_logger().info(
            f"收到 /p_top: p=({p.x:.3f},{p.y:.3f},{p.z:.3f}), q=({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f})"
        )

        # 1) 等待 IK 服务
        if not self.ik_cli.service_is_ready():
            self.get_logger().info("等待 /compute_ik 服务...")
            if not self.ik_cli.wait_for_service(timeout_sec=5.0):
                self.get_logger().error("等待 /compute_ik 超时，确认 move_group 是否已启动")
                return

        # 2) 组装种子（当前关节；若未收到 joint_states，则用 0）
        seed = RobotState()
        seed.joint_state.name = JOINT_NAMES
        if self._js:
            seed.joint_state.position = [self._js.get(n, 0.0) for n in JOINT_NAMES]
        else:
            seed.joint_state.position = [0.0] * len(JOINT_NAMES)
            self.get_logger().warn("尚未收到 /joint_states，使用零位作为 IK 种子，解可能欠佳")

        # 3) IK 请求
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group
        req.ik_request.ik_link_name = self.eef
        req.ik_request.pose_stamped = msg
        req.ik_request.robot_state = seed
        req.ik_request.avoid_collisions = self.avoid_collisions
        req.ik_request.timeout = Duration(
            sec=int(self.ik_timeout),
            nanosec=int((self.ik_timeout - math.floor(self.ik_timeout)) * 1e9)
        )

        future = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.ik_timeout + 1.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("IK 调用未返回或失败")
            return

        res = future.result()
        if res.error_code.val != 1:
            self.get_logger().error(f"IK 失败，MoveItErrorCodes={res.error_code.val}")
            return

        # 4) 解析解
        sol = res.solution.joint_state
        name2pos = dict(zip(sol.name, sol.position))
        q_cur = [self._js.get(n, 0.0) for n in JOINT_NAMES]
        q_sol = [float(name2pos.get(n, q_cur[i])) for i, n in enumerate(JOINT_NAMES)]

        did_anything = False

        # 5) 预览（DisplayTrajectory）
        if self.preview:
            disp = DisplayTrajectory()
            disp.model_id = self.group
            # 轨迹起点：当前关节
            disp.trajectory_start = RobotState()
            disp.trajectory_start.joint_state.name = JOINT_NAMES
            disp.trajectory_start.joint_state.position = q_cur

            rt = RobotTrajectory()
            rt.joint_trajectory.joint_names = JOINT_NAMES

            p0 = JointTrajectoryPoint()
            p0.positions = q_cur
            p0.time_from_start = Duration(sec=0)

            p1 = JointTrajectoryPoint()
            p1.positions = q_sol
            p1.time_from_start = Duration(sec=int(self.move_time),
                                          nanosec=int((self.move_time % 1) * 1e9))

            rt.joint_trajectory.points = [p0, p1]
            disp.trajectory.append(rt)

            self.preview_pub.publish(disp)
            self.get_logger().info("已发布 DisplayTrajectory（RViz → MotionPlanning 面板可预览）")
            did_anything = True

        # 6) 执行（关节轨迹）
        if self.execute:
            jt = JointTrajectory()
            jt.joint_names = JOINT_NAMES
            pt = JointTrajectoryPoint()
            pt.positions = q_sol
            pt.time_from_start = Duration(sec=int(self.move_time),
                                          nanosec=int((self.move_time % 1) * 1e9))
            jt.points.append(pt)
            self.traj_pub.publish(jt)
            self.get_logger().info("已发布 JointTrajectory（执行）")
            did_anything = True

        if not did_anything:
            self.get_logger().warn("preview=false 且 execute=false：未执行任何操作")

        self._done = True
        if self.single_shot:
            self.create_timer(0.8, self._shutdown_once)

    def _shutdown_once(self):
        self.get_logger().info("一次性执行完成，即将退出节点")
        rclpy.shutdown()

def main():
    rclpy.init()
    node = GoToPTopViaIK()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()


# 3) 仅预览（不执行）
#ros2 run gripanything goto_p_top --ros-args -p preview:=true -p execute:=false

# 4) 仅执行（不预览）
#ros2 run gripanything goto_p_top --ros-args -p preview:=false -p execute:=true

# 5) 先预览再执行
#ros2 run gripanything goto_p_top --ros-args -p preview:=true -p execute:=true