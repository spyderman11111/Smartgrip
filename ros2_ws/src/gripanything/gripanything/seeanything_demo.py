"""
Virtual Top-Down Locator + GroundingDINO（无直线下探版 + 发布物体TF）

流程概述：
  - 从图像检测目标 -> 得到像素 (u, v)
  - 在相机系下构造视线方向 d_cam_cam，并变换到 base 系 d_cam_base
  - 用 base 系下的“虚拟平面” z = z_virt（法向 n = +Z_base）与视线求交 -> 得到交点 C_base
  - 计算上方点 P_top = C_base + h_above * n，并发布 /p_top、/p_pre
  - 发布 TF: base_frame -> object_frame（位于 C_base，姿态与 base 对齐，单位四元数）
  - 末端姿态保持与虚拟平面平行（-Z_tool // +Z_base）

坐标与变换记号（本文件统一采用）：
  B = base_frame（默认 'base_link'）
  T = tool_frame（默认 'tool0'）
  C = camera（手眼外参：tool->cam）
  O = object_frame（目标物体的 TF，姿态与 B 对齐）

  R_parent_child, p_parent_child 的含义：
    - R_parent_child ∈ SO(3)：把 child 向量表达式变到 parent（v_parent = R_parent_child @ v_child）
    - p_parent_child ∈ R^3：child 原点在 parent 系下的坐标

  常用组合关系：
    T^B_T = (R_base_tool, p_base_tool)          # TF: base <- tool
    T^T_C = (R_tool_cam, p_tool_cam)            # 外参: tool -> cam
    T^B_C = T^B_T ∘ T^T_C = (R_base_tool @ R_tool_cam,
                             R_base_tool @ p_tool_cam + p_base_tool)
    逆变换：
    T^C_B = (R_cam_base, p_cam_base) = inverse(T^B_C)

  物体 TF：
    T^B_O = (R_base_obj, p_base_obj) = (I, C_base)      # 姿态与 base 对齐
    T^C_O = T^C_B ∘ T^B_O = (R_cam_base @ I, R_cam_base @ (p_base_obj - p_base_cam))
"""

# ===================== 可随时修改的“初始观察位姿”与发送函数 ======================
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
POS1 = [0.9004912376403809, -1.2545607549003144, 1.2091739813434046, -1.488419310455658, -1.5398953596698206, -0.6954544226275843]
JOINT_TRAJ_TOPIC = '/scaled_joint_trajectory_controller/joint_trajectory'
INIT_MOVE_TIME = 3.0  # 秒
# ===========================================================================

from dataclasses import dataclass
import math
from typing import Tuple, List, Optional
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration as RclDuration

from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as MsgDuration

import tf2_ros
from tf2_ros import TransformException

from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage

# ---------------------------- 你的默认配置（改这里就行） ---------------------------- #
@dataclass(frozen=True)
class Config:
    # 话题
    IMAGE_TOPIC: str = '/my_camera/pylon_ros2_camera_node/image_raw'
    CAMERA_INFO_TOPIC: str = '/my_camera/pylon_ros2_camera_node/camera_info'

    # 是否使用 CameraInfo（有就覆盖 fx/fy/cx/cy 与畸变）
    USE_CAMERA_INFO: bool = True

    # 相机内参（备用/回退；将被 CameraInfo 覆盖）
    FX: float = 2674.3803723910564
    FY: float = 2667.4211254043507
    CX: float = 954.5922081613583
    CY: float = 1074.965947832258

    # 坐标系与 MoveIt
    BASE_FRAME: str = 'base_link'
    TOOL_FRAME: str = 'tool0'
    GROUP_NAME: str = 'ur_manipulator'
    EEF_LINK: str = 'tool0'

    # 虚拟平面与高度
    Z_VIRT: float = 0.0         # 桌面/工作面在 base 的高度（米）
    H_ABOVE: float = 0.30       # 到物体上方的高度
    APPROACH_CLEARANCE: float = 0.10

    # GroundingDINO
    TEXT_PROMPT: str = "yellow object ."  
    DINO_MODEL_ID: str = 'IDEA-Research/grounding-dino-tiny'
    DINO_DEVICE: str = 'cuda'   # 无 GPU 改 'cpu'
    BOX_THRESHOLD: float = 0.2
    TEXT_THRESHOLD: float = 0.2

    # 手眼外参（tool->cam）
    T_TOOL_CAM_XYZ: List[float] = (0.05468636852374024, -0.029182661943126947, 0.05391824813032688)
    T_TOOL_CAM_QUAT: List[float] = (-0.0036165657530785695, -0.000780788838366878,
                                    0.7078681983794892, 0.7063348529868249)  # [qx, qy, qz, qw]

    # 执行选项
    EXECUTE: bool = False
    MAX_VEL_SCALE: float = 0.2
    MAX_ACC_SCALE: float = 0.2

    # 物体 TF 名称
    OBJECT_FRAME: str = 'object_position'

    # 是否在启动后发送初始观察位姿
    SEND_INIT_POSE: bool = True

CFG = Config()

# GroundingDINO 预测器（按你的工程路径导入）
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except ImportError:
    import sys
    sys.path.append('/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything')
    from gripanything.core.detect_with_dino import GroundingDinoPredictor

# 可选：MoveIt
try:
    import moveit_commander  # type: ignore
    _HAS_MOVEIT = True
except Exception:
    _HAS_MOVEIT = False


# ---------------------------- 初始位姿发送工具 ---------------------------- #
def publish_trajectory(node: Node, pub, positions: List[float], move_time: float):
    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    traj.header.stamp = node.get_clock().now().to_msg()

    pt = JointTrajectoryPoint()
    pt.positions = list(positions)
    sec = int(move_time)
    nsec = int((move_time - sec) * 1e9)
    pt.time_from_start = MsgDuration(sec=sec, nanosec=nsec)
    traj.points.append(pt)

    pub.publish(traj)
    node.get_logger().info(f'已发送关节目标（初始观察位姿）: {np.round(positions, 3).tolist()}')


# ---------------------------- 旋转/四元数工具 ---------------------------- #
def rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,             cp*cr]
    ], dtype=float)

def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    t = m00 + m11 + m22
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    n = math.sqrt(x*x + y*y + z*z + w*w)
    return (x/n, y/n, z/n, w/n)


# ---------------------------- 变换(R,p)工具 ---------------------------- #
def tfmsg_to_Rp(transform: TransformStamped) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 geometry_msgs/TransformStamped 转为 (R_parent_child, p_parent_child)
    其中 parent = transform.header.frame_id, child = transform.child_frame_id
    满足：v_parent = R_parent_child @ v_child
         p_parent_child = child原点在parent系下坐标
    """
    q = transform.transform.rotation
    t = transform.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R_parent_child = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)
    p_parent_child = np.array([t.x, t.y, t.z], dtype=float)
    return R_parent_child, p_parent_child

def compose_Rp(R_ab: np.ndarray, p_ab: np.ndarray,
               R_bc: np.ndarray, p_bc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    复合变换：T^A_C = T^A_B ∘ T^B_C
      R_ac = R_ab @ R_bc
      p_ac = R_ab @ p_bc + p_ab
    """
    R_ac = R_ab @ R_bc
    p_ac = R_ab @ p_bc + p_ab
    return R_ac, p_ac

def invert_Rp(R_ab: np.ndarray, p_ab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    逆变换：T^B_A = (R_ab^T, -R_ab^T @ p_ab)
    """
    R_ba = R_ab.T
    p_ba = -R_ba @ p_ab
    return R_ba, p_ba


# ---------------------------- 主节点 ---------------------------- #
class VirtualTopdownNode(Node):
    def __init__(self):
        super().__init__('virtual_topdown_locator')

        # —— 用脚本内置默认值声明参数（运行时无需再传 CLI）——
        self.declare_parameter('use_camera_info', CFG.USE_CAMERA_INFO)
        self.declare_parameter('camera_info_topic', CFG.CAMERA_INFO_TOPIC)
        self.declare_parameter('fx', CFG.FX); self.declare_parameter('fy', CFG.FY)
        self.declare_parameter('cx', CFG.CX); self.declare_parameter('cy', CFG.CY)
        self.declare_parameter('t_tool_cam_xyz', list(CFG.T_TOOL_CAM_XYZ))
        self.declare_parameter('t_tool_cam_rpy', [0.0, 0.0, 0.0])  # 若用 RPY，填这里；否则保持 0
        self.declare_parameter('t_tool_cam_quat', list(CFG.T_TOOL_CAM_QUAT))
        self.declare_parameter('base_frame', CFG.BASE_FRAME)
        self.declare_parameter('tool_frame', CFG.TOOL_FRAME)
        self.declare_parameter('group_name', CFG.GROUP_NAME)
        self.declare_parameter('eef_link', CFG.EEF_LINK)
        self.declare_parameter('z_virt', CFG.Z_VIRT)
        self.declare_parameter('h_above', CFG.H_ABOVE)
        self.declare_parameter('approach_clearance', CFG.APPROACH_CLEARANCE)
        self.declare_parameter('target_pixel_topic', '/target_pixel')
        self.declare_parameter('execute', CFG.EXECUTE)
        self.declare_parameter('max_vel_scale', CFG.MAX_VEL_SCALE)
        self.declare_parameter('max_acc_scale', CFG.MAX_ACC_SCALE)
        self.declare_parameter('image_topic', CFG.IMAGE_TOPIC)
        self.declare_parameter('text_prompt', CFG.TEXT_PROMPT)
        self.declare_parameter('dino_model_id', CFG.DINO_MODEL_ID)
        self.declare_parameter('dino_device', CFG.DINO_DEVICE)
        self.declare_parameter('box_threshold', CFG.BOX_THRESHOLD)
        self.declare_parameter('text_threshold', CFG.TEXT_THRESHOLD)
        self.declare_parameter('object_frame', CFG.OBJECT_FRAME)

        # 读参数
        g = self.get_parameter
        self.use_ci = g('use_camera_info').value
        self.ci_topic = g('camera_info_topic').value
        self.fx = float(g('fx').value); self.fy = float(g('fy').value)
        self.cx = float(g('cx').value); self.cy = float(g('cy').value)
        self.t_tool_cam_xyz = np.array(g('t_tool_cam_xyz').value, dtype=float)
        self.t_tool_cam_rpy = np.array(g('t_tool_cam_rpy').value, dtype=float)
        self.t_tool_cam_quat = np.array(g('t_tool_cam_quat').value, dtype=float)
        self.base_frame = g('base_frame').value
        self.tool_frame = g('tool_frame').value
        self.group_name = g('group_name').value
        self.eef_link = g('eef_link').value
        self.z_virt = float(g('z_virt').value)
        self.h_above = float(g('h_above').value)
        self.approach_clearance = float(g('approach_clearance').value)
        self.target_pixel_topic = g('target_pixel_topic').value
        self.execute = bool(g('execute').value)
        self.max_vel = float(g('max_vel_scale').value)
        self.max_acc = float(g('max_acc_scale').value)
        self.image_topic = g('image_topic').value
        self.text_prompt = g('text_prompt').value
        self.dino_model_id = g('dino_model_id').value
        self.dino_device = g('dino_device').value
        self.box_th = float(g('box_threshold').value)
        self.text_th = float(g('text_threshold').value)
        self.object_frame = g('object_frame').value

        # 内参缓存（回退值：会被 /camera_info 覆盖）
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0,      0,      1]], dtype=float)
        self.D: Optional[np.ndarray] = None
        self.dist_model: Optional[str] = None
        self._have_K = not self.use_ci

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)  # 发布物体 TF

        # MoveIt（仅用于到 P_top 的一次规划）
        if self.execute:
            try:
                import moveit_commander  # 确保已导入
                moveit_commander.roscpp_initialize([])
                self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
                self.move_group.set_pose_reference_frame(self.base_frame)
                self.move_group.set_end_effector_link(self.eef_link)
                self.move_group.set_max_velocity_scaling_factor(self.max_vel)
                self.move_group.set_max_acceleration_scaling_factor(self.max_acc)
            except Exception as e:
                self.get_logger().warn(f'execute=True 但 MoveIt 初始化失败：{e}，将仅发布位姿。')
                self.execute = False

        # 订阅/发布
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        if self.use_ci:
            self.sub_ci = self.create_subscription(CameraInfo, self.ci_topic, self._cb_ci, qos)
        self.sub_uv = self.create_subscription(PointStamped, self.target_pixel_topic, self._cb_uv, qos)

        self.bridge = CvBridge()
        self.predictor = GroundingDinoPredictor(model_id=self.dino_model_id, device=self.dino_device)
        self.sub_img = self.create_subscription(Image, self.image_topic, self._cb_image, qos)

        self.pub_p_top = self.create_publisher(PoseStamped, '/p_top', 1)
        self.pub_p_pre = self.create_publisher(PoseStamped, '/p_pre', 1)

        # 关节轨迹发布器（初始观察位姿）
        self.traj_pub = self.create_publisher(JointTrajectory, JOINT_TRAJ_TOPIC, 10)
        self._init_pose_sent = False
        if CFG.SEND_INIT_POSE:
            self.create_timer(0.8, self._send_initial_once)

        self.get_logger().info(
            f"[VTDown] z_virt={self.z_virt:.3f}, h_above={self.h_above:.3f}; "
            f"use_camera_info={self.use_ci}, image_topic={self.image_topic}, prompt='{self.text_prompt}', execute={self.execute}, "
            f"object_frame='{self.object_frame}'"
        )
        self._busy = False

    # 只发送一次初始观察位姿
    def _send_initial_once(self):
        if self._init_pose_sent:
            return
        publish_trajectory(self, self.traj_pub, POS1, INIT_MOVE_TIME)
        self._init_pose_sent = True

    # ---------------- CameraInfo 回调 ----------------
    def _cb_ci(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.K = K.copy()
        self.fx, self.fy, self.cx, self.cy = K[0,0], K[1,1], K[0,2], K[1,2]
        if msg.d is not None and len(msg.d) > 0:
            self.D = np.array(msg.d, dtype=float).reshape(-1)
        else:
            self.D = None
        self.dist_model = getattr(msg, 'distortion_model', None) or None
        self._have_K = True

    # ---------------- 图像回调（自动检测 -> (u,v)） ----------------
    def _cb_image(self, msg: Image):
        if self._busy:
            return
        if not self._have_K:
            self.get_logger().warn_once('等待 /camera_info 提供内参/畸变。')
            return
        try:
            self._busy = True
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            pil = PILImage.fromarray(rgb)
            boxes, labels = self.predictor.predict(
                pil, self.text_prompt,
                box_threshold=self.box_th, text_threshold=self.text_th
            )
            if boxes is None or len(boxes) == 0:
                self.get_logger().warn('未检测到目标，检查 text_prompt/阈值。')
                return
            x0, y0, x1, y1 = boxes[0].tolist()
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)
            self._process_uv(u, v, msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f"DINO/图像处理失败: {e}")
        finally:
            self._busy = False

    # ---------------- 手动像素回调（可选） ----------------
    def _cb_uv(self, msg: PointStamped):
        if not self._have_K:
            self.get_logger().warn('未获得相机内参/畸变，忽略此帧。')
            return
        self._process_uv(float(msg.point.x), float(msg.point.y), msg.header.stamp)

    # ---------------- 主流程：像素 -> 射线 -> 求交 -> 发布/执行/发TF ----------------
    def _process_uv(self, u: float, v: float, stamp):
        """
        记号：
          B = base_frame，T = tool_frame，C = camera。
          平面为 B 系下 z = z_virt，法向 n_base = [0,0,1]^T。

        组合关系：
          T^B_T = (R_base_tool, p_base_tool)   # TF: base <- tool
          T^T_C = (R_tool_cam, p_tool_cam)     # 外参: tool -> cam
          T^B_C = T^B_T ∘ T^T_C = (R_base_tool @ R_tool_cam,
                                   R_base_tool @ p_tool_cam + p_base_tool)
        """
        # 1) TF: base <- tool at stamp
        try:
            T_base_tool_msg = self.tf_buffer.lookup_transform(
                self.base_frame, self.tool_frame, rclpy.time.Time.from_msg(stamp)
            )
        except TransformException as ex:
            self.get_logger().warn(f'TF 未就绪: {ex}')
            return
        R_base_tool, p_base_tool = tfmsg_to_Rp(T_base_tool_msg)

        # 2) 手眼外参：tool -> cam（优先 quat）
        if self.t_tool_cam_quat is not None and not np.allclose(self.t_tool_cam_quat, [0, 0, 0, 1.0]):
            R_tool_cam = quat_to_rot(*self.t_tool_cam_quat.tolist())
        else:
            R_tool_cam = rpy_to_rot(*self.t_tool_cam_rpy.tolist())
        p_tool_cam = self.t_tool_cam_xyz

        # 3) 相机在 base 下的位姿：T^B_C
        R_base_cam, p_base_cam = compose_Rp(R_base_tool, p_base_tool, R_tool_cam, p_tool_cam)

        # 4) 像素 -> 相机归一化光线（先去畸变，再归一化）
        if self.D is not None and self.dist_model in (None, '', 'plumb_bob', 'rational_polynomial'):
            pts = np.array([[[u, v]]], dtype=np.float32)
            undist = cv2.undistortPoints(pts, self.K, self.D, P=None)  # -> 1x1x2
            x_n, y_n = float(undist[0, 0, 0]), float(undist[0, 0, 1])
            d_cam_cam = np.array([x_n, y_n, 1.0], dtype=float)  # C 系方向
        else:
            x_n = (u - self.cx) / self.fx
            y_n = (v - self.cy) / self.fy
            d_cam_cam = np.array([x_n, y_n, 1.0], dtype=float)

        d_cam_cam /= np.linalg.norm(d_cam_cam)
        d_cam_base = R_base_cam @ d_cam_cam        # 方向变换到 B 系
        d_cam_base /= np.linalg.norm(d_cam_base)
        o_base = p_base_cam                         # 光心 O 的 B 系坐标

        # 5) 与虚拟平面（B 系下 z = z_virt）求交
        n_base = np.array([0.0, 0.0, 1.0], dtype=float)
        rz = float(d_cam_base[2])
        if abs(rz) < 1e-6:
            self.get_logger().warn('视线近水平（|r_z|≈0），无法与虚拟平面求交。')
            return

        t_star = (self.z_virt - float(o_base[2])) / rz
        if t_star < 0:
            self.get_logger().warn('平面交点在相机后方（t<0），忽略。')
            return

        C_base = o_base + t_star * d_cam_base  # 交点（物体中心）in B

        # 6) 计算上方点（用于执行/可视化）
        P_top_base = C_base + self.h_above * n_base
        P_pre_base = P_top_base + self.approach_clearance * n_base  # 仅可视化

        # 7) 末端期望姿态：-Z_tool // +Z_base
        z_tool_in_base = -n_base
        x_tool_in_base = np.array([1.0, 0.0, 0.0])  # 任意与 z 不共线的单位向量
        y_tool_in_base = np.cross(z_tool_in_base, x_tool_in_base)
        if np.linalg.norm(y_tool_in_base) < 1e-6:
            x_tool_in_base = np.array([0.0, 1.0, 0.0])
            y_tool_in_base = np.cross(z_tool_in_base, x_tool_in_base)
        x_tool_in_base /= np.linalg.norm(x_tool_in_base)
        y_tool_in_base /= np.linalg.norm(y_tool_in_base)
        z_tool_in_base /= np.linalg.norm(z_tool_in_base)
        R_base_tool_des = np.column_stack((x_tool_in_base, y_tool_in_base, z_tool_in_base))
        qx, qy, qz, qw = rot_to_quat(R_base_tool_des)

        # 8) 发布 P_top / P_pre（B 系）
        ps_top = PoseStamped()
        ps_top.header = Header(stamp=stamp, frame_id=self.base_frame)
        ps_top.pose.position.x, ps_top.pose.position.y, ps_top.pose.position.z = map(float, P_top_base)
        ps_top.pose.orientation.x, ps_top.pose.orientation.y, ps_top.pose.orientation.z, ps_top.pose.orientation.w = qx, qy, qz, qw
        self.pub_p_top.publish(ps_top)

        ps_pre = PoseStamped()
        ps_pre.header = Header(stamp=stamp, frame_id=self.base_frame)
        ps_pre.pose.position.x, ps_pre.pose.position.y, ps_pre.pose.position.z = map(float, P_pre_base)
        ps_pre.pose.orientation.x, ps_pre.pose.orientation.y, ps_pre.pose.orientation.z, ps_pre.pose.orientation.w = qx, qy, qz, qw
        self.pub_p_pre.publish(ps_pre)

        # 9) 发布物体 TF（T^B_O）：位置在 C_base，姿态与 B 对齐
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()   # 用当前时间，RViz 更容易对齐
        t.header.frame_id = self.base_frame
        t.child_frame_id = self.object_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, C_base)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info(
            f"(u,v)=({u:.1f},{v:.1f})  "
            f"C_base=({C_base[0]:.3f},{C_base[1]:.3f},{C_base[2]:.3f})  "
            f"P_top=({P_top_base[0]:.3f},{P_top_base[1]:.3f},{P_top_base[2]:.3f})  "
            f"-> TF:{self.object_frame}"
        )

    # ---------------- 仅到 P_top 的 MoveIt 执行 ----------------
    def _go_to_pose(self, ps_target: PoseStamped):
        import moveit_commander
        self.move_group.set_pose_target(ps_target)
        plan = self.move_group.plan()
        ok = plan and len(plan.joint_trajectory.points) > 0
        if not ok:
            self.get_logger().warn('到 P_top 的规划失败，尝试直接 go。')
            self.move_group.set_pose_target(ps_target)
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()


# ---------------------------- 入口：无需 CLI 参数 ---------------------------- #
def main():
    rclpy.init()
    node = VirtualTopdownNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'move_group'):
            try:
                node.move_group.stop()
                node.move_group.clear_pose_targets()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
