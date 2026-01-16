#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class ToolToCameraTFPublisher(Node):
    def __init__(self):
        super().__init__("tool_to_camera_tf_publisher")

        # static tf broadcaster
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # ====== 在这里填你的外参 ======
        # tool -> camera 的平移（单位：米）
        t_tool_cam_xyz = (-0.000006852374024,
                          -0.059182661943126947,
                          -0.00391824813032688)

        # tool -> camera 的四元数（xyzw）
        t_tool_cam_quat_xyzw = (-0.0036165657530785695,
                                -0.000780788838366878,
                                0.7078681983794892,
                                0.7063348529868249)
        # =============================

        # 构造 TransformStamped
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()

        # 注意这里的 frame 名字要和 URDF/UR5e 驱动一致
        tf_msg.header.frame_id = "tool0"          # 父坐标系：机械臂末端
        tf_msg.child_frame_id = "camera_optical"  # 子坐标系：相机

        tf_msg.transform.translation.x = t_tool_cam_xyz[0]
        tf_msg.transform.translation.y = t_tool_cam_xyz[1]
        tf_msg.transform.translation.z = t_tool_cam_xyz[2]

        tf_msg.transform.rotation.x = t_tool_cam_quat_xyzw[0]
        tf_msg.transform.rotation.y = t_tool_cam_quat_xyzw[1]
        tf_msg.transform.rotation.z = t_tool_cam_quat_xyzw[2]
        tf_msg.transform.rotation.w = t_tool_cam_quat_xyzw[3]

        # 发送 static TF（一次即可，tf2 会 latched）
        self.static_broadcaster.sendTransform(tf_msg)
        self.get_logger().info(
            f"Published static TF: {tf_msg.header.frame_id} -> {tf_msg.child_frame_id}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ToolToCameraTFPublisher()
    try:
        # 对 static tf 来说，其实不 spin 也可以；这里保持节点存活更安全一点
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
