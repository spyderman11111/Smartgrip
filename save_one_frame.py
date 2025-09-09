import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class OneShotSaver(Node):
    def __init__(self, topic, path):
        super().__init__('one_shot_saver')
        self.bridge = CvBridge()
        self.path = path
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        self.sub = self.create_subscription(Image, topic, self.cb, qos)
        self.get_logger().info(f"等待来自 {topic} 的一帧图像…")

    def cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        import cv2
        cv2.imwrite(self.path, img)
        self.get_logger().info(f"已保存到 {self.path}")
        rclpy.shutdown()

def main():
    import sys
    topic = '/my_camera/pylon_ros2_camera_node/image_raw'
    path = '/home/MA_SmartGrip/Smartgrip'
    if len(sys.argv) >= 2: topic = sys.argv[1]
    if len(sys.argv) >= 3: path = sys.argv[2]
    rclpy.init()
    rclpy.spin(OneShotSaver(topic, path))

if __name__ == '__main__':
    main()
