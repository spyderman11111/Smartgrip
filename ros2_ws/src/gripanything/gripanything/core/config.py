"""
config.py — Centralized configuration dataclasses and ROS-parameter loading.

Purpose:
- Keep all tunable defaults in one place (clean main node).
- Allow selective overrides via ROS 2 parameters (e.g., --ros-args -p ...).
- Provide a simple loader `load_from_ros_params(node)` used by the main node.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class Frames:
    image_topic: str = '/my_camera/pylon_ros2_camera_node/image_raw'
    base_frame:   str = 'base_link'
    tool_frame:   str = 'tool0'
    pose_frame:   str = 'base_link'
    object_frame: str = 'object_position'
    circle_frame: str = 'object_circle'
    z_virt: float = 0.0

@dataclass
class Camera:
    fx: float = 2674.3803723910564
    fy: float = 2667.4211254043507
    cx: float = 954.5922081613583
    cy: float = 1074.965947832258
    hand_eye_frame: str = 'optical'  # 'optical' or 'link'
    t_tool_cam_xyz: List[float] = (-0.000006852374024, -0.059182661943126947, -0.00391824813032688)
    t_tool_cam_quat_xyzw: List[float] = (-0.0036165657530785695, -0.000780788838366878, 0.7078681983794892, 0.7063348529868249)
    flip_x: bool = False
    flip_y: bool = False

@dataclass
class Dino:
    model_id: str = 'IDEA-Research/grounding-dino-tiny'
    device: str = 'cuda'
    text_prompt: str = 'yellow object .'
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    min_exec_score: float = 0.5  

@dataclass
class Bias:
    enable: bool = True
    bx: float = -0.03
    by: float = -0.23
    bz: float = 0.0

@dataclass
class Control:
    group_name: str = 'ur_manipulator'
    ik_link_name: str = 'tool0'
    ik_timeout: float = 2.0
    controller_topic: str = '/scaled_joint_trajectory_controller/joint_trajectory'
    move_time: float = 3.0
    hover_above: float = 0.30
    init_move_time: float = 5.0
    init_extra_wait: float = 0.3
    joint_order: List[float] = (
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    )
    init_pos: List[float] = (
        0.7734344005584717,
       -1.0457398456386109,
        1.0822847525226038,
       -1.581707616845602,
       -1.5601266066180628,
       -0.8573678175555628,
    )
    require_stationary: bool = True
    vel_eps: float = 0.02
    tf_rebroadcast_hz: float = 20.0
    tf_time_mode: str = 'latest'  # 'image'|'latest'

@dataclass
class Circle:
    n_vertices: int = 8
    num_turns: int = 1
    poly_dir: str = 'ccw'
    start_dir_offset_deg: float = -90.0
    radius: float = 0.10
    orient_mode: str = 'radial_in'
    tool_z_sign: str = '-'
    dwell_time: float = 1.5
    edge_move_time: float = 3.0

@dataclass
class JumpGuard:
    max_safe_jump: float = 1.2
    max_warn_jump: float = 2.2
    ignore_joints: List[str] = ('wrist_3_joint',)

@dataclass
class Runtime:
    frame_stride: int = 2
    require_prompt: bool = True

@dataclass
class Config:
    frames: Frames = Frames()
    cam: Camera = Camera()
    dino: Dino = Dino()
    bias: Bias = Bias()
    control: Control = Control()
    circle: Circle = Circle()
    jump: JumpGuard = JumpGuard()
    runtime: Runtime = Runtime()

def load_from_ros_params(node) -> Config:
    """
    Load a Config with selective overrides from ROS parameters.
    """
    cfg = Config()
    cfg.runtime.require_prompt = node.declare_parameter('require_prompt', cfg.runtime.require_prompt).value
    cfg.dino.text_prompt = node.declare_parameter('text_prompt', cfg.dino.text_prompt).value
    cfg.dino.min_exec_score = float(node.declare_parameter('min_exec_score', cfg.dino.min_exec_score).value) 

    cfg.circle.n_vertices = int(node.declare_parameter('N', cfg.circle.n_vertices).value)
    cfg.circle.radius = float(node.declare_parameter('R', cfg.circle.radius).value)
    cfg.circle.orient_mode = node.declare_parameter('orient_mode', cfg.circle.orient_mode).value
    cfg.circle.poly_dir = node.declare_parameter('poly_dir', cfg.circle.poly_dir).value
    cfg.circle.start_dir_offset_deg = float(node.declare_parameter('start_dir_offset_deg', cfg.circle.start_dir_offset_deg).value)
    cfg.control.hover_above = float(node.declare_parameter('hover', cfg.control.hover_above).value)
    cfg.bias.bx = float(node.declare_parameter('bias_x', cfg.bias.bx).value)
    cfg.bias.by = float(node.declare_parameter('bias_y', cfg.bias.by).value)
    cfg.bias.bz = float(node.declare_parameter('bias_z', cfg.bias.bz).value)
    return cfg
