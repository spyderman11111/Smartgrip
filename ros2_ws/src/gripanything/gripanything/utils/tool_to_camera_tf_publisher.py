#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class ToolToCameraTFPublisher(Node):
    """
    Publish a static transform: tool0 -> camera_optical.

    Notes:
    - Frame names must match your URDF / driver frames.
    - Quaternion order is (x, y, z, w).
    """

    def __init__(self):
        super().__init__("tool_to_camera_tf_publisher")
        self._br = StaticTransformBroadcaster(self)

        # ---- Fill your hand-eye extrinsics here (tool0 -> camera_optical) ----
        t_tool_cam_xyz = (
            -0.000006852374024,
            -0.059182661943126947,
            -0.00391824813032688,
        )
        q_tool_cam_xyzw = (
            -0.0036165657530785695,
            -0.000780788838366878,
            0.7078681983794892,
            0.7063348529868249,
        )
        # --------------------------------------------------------------------

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
            f"Static TF published: {msg.header.frame_id} -> {msg.child_frame_id}"
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
