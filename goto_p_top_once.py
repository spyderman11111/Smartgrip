#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped

class GoToPTopOnce(Node):
    def __init__(self):
        super().__init__('goto_p_top_once')
        # 参数
        self.declare_parameter('topic', '/p_top')
        self.declare_parameter('group_name', 'ur_manipulator')
        self.declare_parameter('eef_link', 'tool0')
        self.declare_parameter('reference_frame', 'base_link')
        self.declare_parameter('max_vel_scale', 0.2)
        self.declare_parameter('max_acc_scale', 0.2)

        self.topic = self.get_parameter('topic').value
        self.group_name = self.get_parameter('group_name').value
        self.eef_link = self.get_parameter('eef_link').value
        self.reference_frame = self.get_parameter('reference_frame').value
        self.max_vel = float(self.get_parameter('max_vel_scale').value)
        self.max_acc = float(self.get_parameter('max_acc_scale').value)

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(PoseStamped, self.topic, self._cb, qos)

        # 初始化 MoveIt
        try:
            import moveit_commander
            moveit_commander.roscpp_initialize([])
            from moveit_commander import MoveGroupCommander
            self.move_group = MoveGroupCommander(self.group_name)
            self.move_group.set_pose_reference_frame(self.reference_frame)
            self.move_group.set_end_effector_link(self.eef_link)
            self.move_group.set_max_velocity_scaling_factor(self.max_vel)
            self.move_group.set_max_acceleration_scaling_factor(self.max_acc)
            self._has_moveit = True
            self.get_logger().info(f"MoveIt OK: group={self.group_name}, eef={self.eef_link}, frame={self.reference_frame}")
        except Exception as e:
            self._has_moveit = False
            self.get_logger().error(f"MoveIt 初始化失败: {e}")

        self._done = False

    def _cb(self, msg: PoseStamped):
        if self._done or not self._has_moveit:
            return
        try:
            self.get_logger().info(f"收到 {self.topic}: "
                                   f"p=({msg.pose.position.x:.3f},{msg.pose.position.y:.3f},{msg.pose.position.z:.3f})")
            self.move_group.set_pose_target(msg)
            plan = self.move_group.plan()
            ok = plan and hasattr(plan, "joint_trajectory") and len(plan.joint_trajectory.points) > 0
            if not ok:
                self.get_logger().warn("规划为空，尝试直接 go(wait=True)")
            self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.get_logger().info("已到达 /p_top（一次性执行完成）")
        except Exception as e:
            self.get_logger().error(f"移动失败: {e}")
        finally:
            self._done = True
            # 稍等片刻再退出，避免日志被截断
            self.create_timer(0.5, self._shutdown)

    def _shutdown(self):
        rclpy.shutdown()

def main():
    rclpy.init()
    node = GoToPTopOnce()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if hasattr(node, "move_group"):
                node.move_group.stop()
                node.move_group.clear_pose_targets()
        except Exception:
            pass

if __name__ == '__main__':
    main()
