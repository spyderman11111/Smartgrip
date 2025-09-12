#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
goto_hover_once.py — ROS 2 Humble / UR5e
功能：
- 监听 TF: object_position（在 base_link 下）
- 计算悬停点 P_hover = object_position + [0, 0, HOVER_ABOVE]
- 设定末端与桌面平行（-Z_tool // +Z_base），可选在 Z 轴上加 yaw 角
- 调用 MoveIt /compute_ik 求解一次 IK
- 将解发布到 /scaled_joint_trajectory_controller/joint_trajectory（一次性）

依赖：
- 已运行 UR 驱动与 scaled_joint_trajectory_controller
- 已运行 MoveIt bringup（提供 /compute_ik）
- 你的 seeanything_minimal.py 正在发布 TF: object_position

用法（示例）：
  ros2 run your_pkg goto_hover_once
  # 或带参数：
  ros2 run your_pkg goto_hover_once --ros-args -p hover_above:=0.30 -p yaw_deg:=0.0 -p move_time:=3.0
"""

import math
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.duration import Duration as RclDuration
from rclpy.time import Time

import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from moveit_msgs.srv import GetPositionIK


def quat_from_rx_ry_rz(rx: float, ry: float, rz: float):
    """XYZ欧拉角 -> 四元数 (x,y,z,w)"""
    cx, cy, cz = math.cos(rx/2), math.cos(ry/2), math.cos(rz/2)
    sx, sy, sz = math.sin(rx/2), math.sin(ry/2), math.sin(rz/2)
    qw = cx*cy*cz + sx*sy*sz
    qx = sx*cy*cz - cx*sy*sz
    qy = cx*sy*cz + sx*cy*sz
    qz = cx*cy*sz - sx*sy*cz
    return (qx, qy, qz, qw)


class GoToHoverOnce(Node):
    def __init__(self):
        super().__init__('goto_hover_once')

        # ------ 参数 ------
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('object_frame', 'object_position')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('group_name', 'ur_manipulator')   # 你的 MoveIt 规划组名
        self.declare_parameter('ik_timeout', 0.2)                 # s
        self.declare_parameter('hover_above', 0.30)               # m
        self.declare_parameter('yaw_deg', 0.0)                    # 末端绕 Z 的偏航
        self.declare_parameter('move_time', 3.0)                  # 轨迹时间
        self.declare_parameter('controller_topic', '/scaled_joint_trajectory_controller/joint_trajectory')

        self.base_frame = self.get_parameter('base_frame').value
        self.object_frame = self.get_parameter('object_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.group_name = self.get_parameter('group_name').value
        self.ik_timeout = float(self.get_parameter('ik_timeout').value)
        self.hover_above = float(self.get_parameter('hover_above').value)
        self.yaw_deg = float(self.get_parameter('yaw_deg').value)
        self.move_time = float(self.get_parameter('move_time').value)
        self.controller_topic = self.get_parameter('controller_topic').value

        # ------ TF / IK / 发布器 ------
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')
        self.pub_traj = self.create_publisher(JointTrajectory, self.controller_topic, QoSProfile(depth=1))

        # joint_states（用于 IK 初始位姿）
        self._last_js = None
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)

        # 主循环：定时尝试一次
        self._done = False
        self._timer = self.create_timer(0.2, self._tick)

        self.get_logger().info(f'GoToHoverOnce 启动：object={self.object_frame}, hover={self.hover_above:.3f} m, move_time={self.move_time:.1f}s')

    def _on_js(self, msg: JointState):
        self._last_js = msg

    def _lookup_object(self):
        try:
            tfmsg = self.tf_buffer.lookup_transform(self.base_frame, self.object_frame, Time(),
                                                    timeout=RclDuration(seconds=0.2))
            t = tfmsg.transform.translation
            return (t.x, t.y, t.z)
        except TransformException as e:
            self.get_logger().warn_once(f'等待 TF {self.base_frame} <- {self.object_frame} ...')
            return None

    def _wait_ik_service(self):
        if not self.ik_cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn_once('等待 /compute_ik 服务 ...')
            return False
        return True

    @staticmethod
    def _build_name_index(names: List[str]) -> Dict[str, int]:
        return {n: i for i, n in enumerate(names)}

    def _tick(self):
        if self._done:
            return
        if not self._wait_ik_service():
            return

        obj = self._lookup_object()
        if obj is None:
            return
        x, y, z = obj
        P_hover = (x, y, z + self.hover_above)

        # 末端朝下，与桌面平行：Rx(pi)，再加一个绕 Z 的 yaw
        rx = math.pi       # 翻转让 z_tool 指向 -Z_base
        ry = 0.0
        rz = math.radians(self.yaw_deg)
        qx, qy, qz, qw = quat_from_rx_ry_rz(rx, ry, rz)

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = P_hover[0]
        pose.pose.position.y = P_hover[1]
        pose.pose.position.z = P_hover[2]
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        # 组织 IK 请求
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.tool_frame
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout = Duration(sec=int(self.ik_timeout), nanosec=int((self.ik_timeout % 1.0) * 1e9))
        req.ik_request.attempts = 1

        # 当前关节作为种子
        if self._last_js is not None:
            req.ik_request.robot_state.joint_state = self._last_js
        else:
            self.get_logger().warn('未收到 /joint_states，仍尝试 IK（可能较难收敛）')

        # 调用 IK
        fut = self.ik_cli.call_async(req)
        rclpy.task.Future.add_done_callback(fut, lambda _: None)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)

        if not fut.done() or fut.result() is None:
            self.get_logger().error('IK 调用失败/超时。')
            return

        res = fut.result()
        if res.error_code.val != 1:
            self.get_logger().error(f'IK 未找到解，error_code={res.error_code.val}')
            return

        js = res.solution.joint_state
        name_idx = self._build_name_index(js.name)

        # UR5e 常见关节顺序（按你的控制器）
        ur_order = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        positions = []
        for jn in ur_order:
            if jn not in name_idx:
                self.get_logger().error(f'IK 结果缺少关节 {jn}，实际包含：{js.name}')
                return
            positions.append(js.position[name_idx[jn]])

        # 发布一次性轨迹
        traj = JointTrajectory()
        traj.joint_names = ur_order
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = Duration(sec=int(self.move_time), nanosec=int((self.move_time % 1.0) * 1e9))
        traj.points.append(pt)
        self.pub_traj.publish(traj)

        self.get_logger().info(
            f'已发送到悬停点 P_hover=({P_hover[0]:.3f},{P_hover[1]:.3f},{P_hover[2]:.3f}), '
            f'yaw={self.yaw_deg:.1f}°, 用时 {self.move_time:.1f}s'
        )
        self._done = True
        # 收尾退出
        self.create_timer(0.5, self._shutdown_once)

    def _shutdown_once(self):
        self.get_logger().info('GoToHoverOnce 完成，退出节点。')
        rclpy.shutdown()


def main():
    rclpy.init()
    node = GoToHoverOnce()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
