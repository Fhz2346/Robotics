import numpy as np
import math
from . import mspace as Ms
import aspace as As

L1 = L2 = 10.0


def forward_kinematics(q):
    q1, q2 = q
    p0 = np.array([0.0, 0.0])
    p1 = np.array([L1 * np.cos(q1), L1 * np.sin(q1)])
    p2 = p1 + np.array([L2 * np.cos(q1 + q2), L2 * np.sin(q1 + q2)])
    return p0, p1, p2


def inverse_kinematics(point):
    """Return the two analytical IK branches (radians), if point is reachable."""
    x, y = map(float, point)
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if c2 < -1.0 - 1e-9 or c2 > 1.0 + 1e-9:
        return []
    c2 = float(np.clip(c2, -1.0, 1.0))
    solutions = []
    for sign in (1.0, -1.0):
        q2 = sign * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q = As.wrap(np.array([q1, q2]))
        if Ms.is_theta2_valid_rad(q) and not any(As.distance(q, old) < 1e-6 for old in solutions):
            solutions.append(q)
    return solutions
