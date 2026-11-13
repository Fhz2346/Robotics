import numpy as np
from numpy.linalg import norm


def exp_pure(v: np.ndarray):
    assert v.shape[0] == 3
    theta = norm(v)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / theta
    return np.array([np.cos(theta), *(np.sin(theta) * axis)])


def inv(q: np.ndarray):
    assert q.shape[0] == 4
    q_inv = -q
    q_inv[0] = q[0]
    return q_inv
