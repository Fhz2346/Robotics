from .mechanical import *
from .obstacles import *


# configuration space : mechanical space + obstacle avoidance
def is_point_valid(q):
    q = As.wrap(q)
    if not Ms.is_theta2_valid_rad(q):
        return False
    # Keep the same Robotics Toolbox collision-checking approach as Task 1.
    return not any(robot.iscollided(q, obstacle) for obstacle in obstacles)


def is_edge_valid(q1, q2, edge_resolution=np.deg2rad(3.0)):
    """Check an edge by sampling along the shortest toroidal interpolation."""
    disp = As.displacement(q1, q2)
    n = max(2, int(np.ceil(np.max(np.abs(disp)) / edge_resolution)) + 1)
    for s in np.linspace(0.0, 1.0, n):
        if not is_point_valid(As.wrap(q1 + s * disp)):
            return False
    return True
