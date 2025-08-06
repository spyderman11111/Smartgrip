#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
import tf_transformations
import time

from ur_ikfast_kinematics import UR5eKinematics  

class SpiralTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('spiral_trajectory_publisher')

        # 初始化 IKFast 解算器（你可替换为 trac_ik 或其他）
        self.ik_solver = UR5eKinematics(base_link="base_link", ee_link="tool0")

        # 发布器
        self.publisher = self.create_publisher(JointTrajectory,
                                               '/scaled_joint_trajectory_controller/joint_trajectory',
                                               10)

        # 定义 UR5e 的关节顺序
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # 执行
        self.publish_spiral_trajectory()

    def generate_spiral_cartesian_poses(self, center, radius, height, revolutions, num_points):
        poses = []
        for i in range(num_points):
            theta = 2 * np.pi * revolutions * i / num_points
            r = radius
            z = center[2] + (height * i / num_points)
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)

            # 朝向目标点（使末端姿态始终朝向中心）
            forward = np.array(center) - np.array([x, y, z])
            forward /= np.linalg.norm(forward)
            up = np.array([0, 0, 1])
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)

            rot = np.eye(4)
            rot[0:3, 0] = right
            rot[0:3, 1] = up
            rot[0:3, 2] = forward
            quat = tf_transformations.quaternion_from_matrix(rot)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            poses.append(pose)
        return poses

    def publish_spiral_trajectory(self):
        center = [0.4, 0.0, 0.2]
        radius = 0.1
        height = 0.2
        revolutions = 2
        num_points = 100

        poses = self.generate_spiral_cartesian_poses(center, radius, height, revolutions, num_points)
        joint_traj = JointTrajectory()
        joint_traj.joint_names = self.joint_names

        t = 0.0
        dt = 0.1  # 每个点的间隔时间

        for pose in poses:
            solution = self.ik_solver.compute_ik(pose)
            if solution is None:
                self.get_logger().warn('IK failed for a point. Skipping.')
                continue

            point = JointTrajectoryPoint()
            point.positions = solution
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)
            joint_traj.points.append(point)
            t += dt

        self.get_logger().info(f"Publishing spiral trajectory with {len(joint_traj.points)} points.")
        time.sleep(1.0)
        self.publisher.publish(joint_traj)

