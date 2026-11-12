import numpy as np

WAYPOINTS = np.array(
    [
        [1.2, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.4, 0.0, 0.0],
        [1.2, 0.0, 0.0],
    ]
)  # [x, y, theta]

DURATION = 10.0 # seconds
FPS = 20

s = lambda tau: (10.0 - 15.0 * tau + 6.0 * tau * tau) * tau**3
dot_s = lambda tau: (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4)

def segment_pose(endpoints, tau):
    beg, end = endpoints
    return beg + s(tau) * (end - beg)


def reference(time):
    """xy position and velocity along the same straight-line profile as task 5."""
    segment_duration = DURATION / (len(WAYPOINTS) - 1)
    segment = min(int(time / segment_duration), len(WAYPOINTS) - 2)
    tau = (time - segment * segment_duration) / segment_duration
    start = WAYPOINTS[segment, :2]
    goal = WAYPOINTS[segment + 1, :2]
    fraction = float(s(tau))
    fraction_rate = float(dot_s(tau)) / segment_duration
    position = start + fraction * (goal - start)
    velocity = fraction_rate * (goal - start)
    return position, velocity
