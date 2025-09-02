#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最简版：UR5e 三位置采集 + VGGT 重建（ROS 2 / Python）
- 给 /scaled_joint_trajectory_controller/joint_trajectory 发送 3 个关节位姿
- 每个位置固定等待 MOVE_TIME + PAUSE_AFTER_MOVE
- 从 IMAGE_TOPIC 抓取 1 帧保存到 SAVE_DIR/session_xxx/images/
- 采集完成后，调用 VGGTReconstructor 进行重建，输出到 sparse/ 和 sparse_txt/
"""

import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2

# ====== 参数 ======
IMAGE_TOPIC = '/pylon_camera_node/image_raw'
SAVE_DIR = '/home/MA_SmartGrip/Smartgrip/captures'
MOVE_TIME = 3.0
PAUSE_AFTER_MOVE = 0.5
IMAGE_TIMEOUT = 2.0

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

POS1 = [0.9298571348, -1.3298700166, 1.9266884963, -2.1331087552, -1.6006286780, -1.0919039885]
POS2 = [0.9298571348, -1.3298700166, 1.9266884963, -1.8831087552, -1.6006286780, -1.0919039885]
POS3 = [0.9298571348, -1.3298700166, 1.9266884963, -2.3831087552, -1.6006286780, -1.0919039885]
# ==================


def publish_trajectory(node: Node, pub, positions, move_time: float):
    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    traj.header.stamp = node.get_clock().now().to_msg()

    pt = JointTrajectoryPoint()
    pt.positions = list(positions)
    sec = int(move_time)
    nsec = int((move_time - sec) * 1e9)
    pt.time_from_start = Duration(sec=sec, nanosec=nsec)
    traj.points.append(pt)

    pub.publish(traj)
    node.get_logger().info(f'已发送关节目标: {positions}')


import torch
from gripanything.core.vggtreconstruction import VGGTReconstructor

def run_demo(scene_dir: str,
             batch_size: int = 4,
             max_points: int = 100000,
             resolution: int = 518,
             conf_thresh: float = 3.5,
             img_limit: int | None = None,
             shared_camera: bool = False,
             camera_type: str = "SIMPLE_PINHOLE"):
    """
    scene_dir 下应包含 images/ 子目录。
    """
    os.makedirs(os.path.join(scene_dir, "images"), exist_ok=True)

    recon = VGGTReconstructor(
        scene_dir=scene_dir,
        batch_size=batch_size,
        max_points=max_points,
        resolution=resolution,
        conf_thresh=conf_thresh,
        img_limit=img_limit,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )
    with torch.no_grad():
        recon.run()

    return {
        "sparse_dir": os.path.join(scene_dir, "sparse"),
        "sparse_txt_dir": os.path.join(scene_dir, "sparse_txt"),
    }


def grab_one_image(node: Node, bridge: CvBridge, topic: str, timeout_s: float):
    msg_holder = {'msg': None}

    def cb(msg: Image):
        msg_holder['msg'] = msg

    sub = node.create_subscription(Image, topic, cb, 1)
    start_ns = node.get_clock().now().nanoseconds
    while rclpy.ok() and msg_holder['msg'] is None:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.get_clock().now().nanoseconds - start_ns > timeout_s * 1e9:
            break
    node.destroy_subscription(sub)
    return msg_holder['msg']


def main():
    rclpy.init()
    node = Node('ur5e_three_shots_min')
    pub = node.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
    bridge = CvBridge()

    # session 目录与 images 子目录
    session = datetime.now().strftime('session_%Y%m%d_%H%M%S')
    scene_dir = os.path.join(SAVE_DIR, session)
    images_dir = os.path.join(scene_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    node.get_logger().info(f'保存目录: {images_dir}')

    # 依次移动并抓图
    for i, pos in enumerate([POS1, POS2, POS3], start=1):
        publish_trajectory(node, pub, pos, MOVE_TIME)
        time.sleep(MOVE_TIME + PAUSE_AFTER_MOVE)

        img_msg = grab_one_image(node, bridge, IMAGE_TOPIC, IMAGE_TIMEOUT)
        if img_msg is None:
            node.get_logger().warn(f'位置 {i}: 未在 {IMAGE_TIMEOUT}s 内收到图像，跳过保存。')
            continue

        cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        out_path = os.path.join(images_dir, f'pos{i}.png')  # 保存到 images/
        cv2.imwrite(out_path, cv_img)
        node.get_logger().info(f'位置 {i}: 已保存 {out_path}')

    # 采集完成后调用 VGGT 重建
    node.get_logger().info('开始 VGGT 重建...')
    outputs = run_demo(scene_dir=scene_dir, batch_size=4, max_points=100000, resolution=518, conf_thresh=3.5)
    node.get_logger().info(f'VGGT 完成，输出: {outputs}')

    # 结束进程
    node.get_logger().info('全部完成。')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
