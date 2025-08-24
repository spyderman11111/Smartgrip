import rclpy, threading, time
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
from message_filters import Subscriber, ApproximateTimeSynchronizer
from .core import detectors, matcher, buffer, vggt_recon, postproc

class SeeAnythingNode(Node):
    def __init__(self):
        super().__init__('seeanything')
        # params
        self.declare_parameter('prompt', 'green cup')
        self.declare_parameter('min_views', 12)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        # tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # subs
        self.sub_img = Subscriber(self, Image, '/camera/color/image_rect')
        self.sub_info = Subscriber(self, CameraInfo, '/camera/color/camera_info')
        self.sync = ApproximateTimeSynchronizer([self.sub_img, self.sub_info], 10, 0.03)
        self.sync.registerCallback(self.on_image)
        # pubs
        self.pub_pcd = self.create_publisher(PointCloud2, '/gripanything/pcd', 1)
        # state
        self.buf = buffer.MultiViewBuffer(max_size=100)
        self.lock = threading.Lock()
        self.recon_busy = False
        # services
        self.srv_reset = self.create_service(Empty, '/gripanything/reset', self.on_reset)
        self.srv_trigger = self.create_service(Trigger, '/gripanything/trigger_recon', self.on_trigger)

    def on_image(self, img_msg, info_msg):
        prompt = self.get_parameter('prompt').value
        cam_frame = self.get_parameter('camera_frame').value
        base_frame = self.get_parameter('base_frame').value
        # TF lookup
        try:
            tf = self.tf_buffer.lookup_transform(base_frame, cam_frame, img_msg.header.stamp, rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'TF missing: {e}')
            return
        K = camera_info_to_K(info_msg)
        rgb = rosimg_to_np(img_msg)
        # fast pipeline
        det = detectors.run_dino(rgb, prompt)
        if det is None: return
        mask = detectors.run_sam2(rgb, det)
        if mask is None: return
        ok = matcher.crossview_gate(mask, K, tf, self.buf)  # LightGlue+RANSAC评分
        if not ok: return
        with self.lock:
            self.buf.add(rgb, mask, K, tf)

        # auto trigger
        if not self.recon_busy and self.buf.good_count() >= self.get_parameter('min_views').value:
            threading.Thread(target=self.reconstruct_and_publish, daemon=True).start()

    def reconstruct_and_publish(self):
        with self.lock:
            self.recon_busy = True
            views = self.buf.sample_for_recon()
        try:
            result = vggt_recon.run(views)   # 返回 {pcd: np.ndarray Nx3(+rgb), T_world='base_link', path: str}
            pcd = postproc.clean(result['pcd'])
            center = postproc.compute_center(pcd)
            msg = pcd_to_ros(pcd, frame_id='base_link')
            self.pub_pcd.publish(msg)
            self.get_logger().info(f"center={center}, saved={result['path']}")
        finally:
            with self.lock:
                self.recon_busy = False

    def on_reset(self, req, res):
        with self.lock: self.buf.clear()
        return res

    def on_trigger(self, req, res):
        if not self.recon_busy:
            threading.Thread(target=self.reconstruct_and_publish, daemon=True).start()
        return res

def main():
    rclpy.init()
    node = SeeAnythingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
