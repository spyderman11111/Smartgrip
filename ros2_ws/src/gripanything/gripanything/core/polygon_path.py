"""
polygon_path.py — Generate N-gon vertices along a circle with desired EE orientation.

Purpose:
- Create non-closed, evenly spaced vertices along a circle around C.
- Guarantee first vertex yaw matches current EE yaw (with a configurable offset).
- Support orientation modes: radial_in, radial_out, tangent; and 'ccw'/'cw' ordering.
"""

import math
import numpy as np
from typing import List
from geometry_msgs.msg import PoseStamped
from .tf_ops import yaw_to_quat_wxyz, pose_from_pq

def make_polygon_vertices(node_clock_now_to_msg,
                          center_xyz: np.ndarray, ring_z: float, start_yaw: float,
                          pose_frame: str,
                          n_vertices: int, num_turns: int, poly_dir: str,
                          orient_mode: str, start_dir_offset_deg: float,
                          radius: float, tool_z_sign: str) -> List[PoseStamped]:
    n = max(3, int(n_vertices))
    turns = max(1, int(num_turns))
    total_deg = 360.0 * turns
    ccw = (str(poly_dir).lower().strip() == 'ccw')
    step = math.radians(total_deg / n)
    dir_sign = +1.0 if ccw else -1.0
    s_tan = +1.0 if ccw else -1.0
    offset = math.radians(start_dir_offset_deg)

    if orient_mode == 'radial_in':
        theta0 = (start_yaw - math.pi) + offset
    elif orient_mode == 'radial_out':
        theta0 = start_yaw + offset
    else:  # tangent
        theta0 = (start_yaw - s_tan * (math.pi / 2)) + offset

    wps: List[PoseStamped] = []
    for i in range(n):
        th = theta0 + dir_sign * (i * step)
        px = center_xyz[0] + radius * math.cos(th)
        py = center_xyz[1] + radius * math.sin(th)
        p  = np.array([px, py, ring_z], dtype=float)

        if orient_mode == 'radial_in':
            yaw = th + math.pi - offset
        elif orient_mode == 'radial_out':
            yaw = th - offset
        else:
            yaw = th + s_tan * (math.pi / 2) - offset

        q = yaw_to_quat_wxyz(yaw, tool_z_sign=tool_z_sign)
        ps = pose_from_pq(p, q, pose_frame)
        ps.header.stamp = node_clock_now_to_msg()
        wps.append(ps)
    return wps  # non-closed
