import numpy as np

# angle space
def wrap(q):
    """Map angles to [-pi, pi)."""
    return (q + np.pi) % (2 * np.pi) - np.pi


def displacement(q1, q2):
    return wrap(q2 - q1)


def distance(q1, q2):
    disp = displacement(q1, q2)
    return np.linalg.norm(disp)


def interpolate(q1, q2, fraction):
    return wrap(q1 + fraction * displacement(q1, q2))
