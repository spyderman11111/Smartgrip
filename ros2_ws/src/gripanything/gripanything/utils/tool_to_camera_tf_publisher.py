#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class ToolToCameraTFPublisher(Node):
    """
    Publish a static transform: tool0 -> camera_optical.

    Quaternion order is (x, y, z, w).
    """

    def __init__(self):
        super().__init__("tool_to_camera_tf_publisher_new_handeye")
        self._br = StaticTransformBroadcaster(self)

        # ---- New hand-eye extrinsics (tool0 -> camera_optical) ----
        # From your T_gripper_cam (Tsai):
        # t = [-0.06635243685969598, -0.0648213404084122, 0.15038971196340464]
        # R converted to quaternion (xyzw):
        # q = [-0.3297946569387602, -0.0070919447650312, -0.0120596898117427, 0.9439490235957437]
        t_tool_cam_xyz = (
            -0.06635243685969598,
            -0.06482134040841220,
             0.15038971196340464,
        )
        q_tool_cam_xyzw = (
            -0.3297946569387602,
            -0.0070919447650312,
            -0.0120596898117427,
             0.9439490235957437,
        )
        # ----------------------------------------------------------

        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "tool0"
        msg.child_frame_id = "camera_optical"

        msg.transform.translation.x = float(t_tool_cam_xyz[0])
        msg.transform.translation.y = float(t_tool_cam_xyz[1])
        msg.transform.translation.z = float(t_tool_cam_xyz[2])

        msg.transform.rotation.x = float(q_tool_cam_xyzw[0])
        msg.transform.rotation.y = float(q_tool_cam_xyzw[1])
        msg.transform.rotation.z = float(q_tool_cam_xyzw[2])
        msg.transform.rotation.w = float(q_tool_cam_xyzw[3])

        self._br.sendTransform(msg)
        self.get_logger().info(
            f"Static TF published: {msg.header.frame_id} -> {msg.child_frame_id}\n"
            f"t = {t_tool_cam_xyz}\n"
            f"q(xyzw) = {q_tool_cam_xyzw}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ToolToCameraTFPublisher()
    try:
        # Keep the node alive so late subscribers can still receive the latched static TF.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
