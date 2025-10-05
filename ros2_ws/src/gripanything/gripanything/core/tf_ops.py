"""
tf_ops.py — Geometry helpers: rotation/quaternion math, TF conversions, and pose builders.

Purpose:
- Keep math utilities isolated and reusable across modules.
- Convert TransformStamped <-> (R, p) fast.
- Generate yaw-aligned quaternions with tool-Z-down option.
"""

import math
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped

def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=float)

def tfmsg_to_Rp(t: TransformStamped):
    q = t.transform.rotation
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    p = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z], dtype=float)
    return R, p

# Rotation: camera_link <- camera_optical (REP-105)
R_CL_CO = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]], dtype=float)

def quat_mul(q1_wxyz, q2_wxyz):
    w1, x1, y1, z1 = q1_wxyz
    w2, x2, y2, z2 = q2_wxyz
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def yaw_to_quat_wxyz(yaw: float, tool_z_sign: str = '-') -> np.ndarray:
    """Return [w, x, y, z] for Rz(yaw), optionally post-multiplied by Rx(pi) for tool-Z-down."""
    c, s = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    qz = np.array([c, 0.0, 0.0, s], dtype=float)
    if str(tool_z_sign).strip() == '-':
        qx_pi = np.array([math.cos(math.pi/2), math.sin(math.pi/2), 0.0, 0.0], dtype=float)
        q = quat_mul(qz, qx_pi)
    else:
        q = qz
    return q / (np.linalg.norm(q) + 1e-12)

def pose_from_pq(p_xyz, q_wxyz, frame) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p_xyz[:3])
    w, x, y, z = map(float, q_wxyz)
    ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = w, x, y, z
    return ps

TWO_PI = 2.0 * math.pi
def wrap_to_near(angle: float, ref: float) -> float:
    """Wrap angle to the 2π-neighborhood of `ref` to avoid large joint jumps."""
    return angle + round((ref - angle) / TWO_PI) * TWO_PI
