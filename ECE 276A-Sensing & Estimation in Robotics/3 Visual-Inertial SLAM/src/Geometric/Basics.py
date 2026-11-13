import numpy as np
from Geometric import SO3 as so3


def augment(p):
    N = p.shape[0]
    p_aug = np.hstack((p, np.ones((N, 1))))
    return p_aug


def odot(p):
    assert p[3] == 1.0
    s = p[:3]
    O = np.zeros((4, 6))
    O[:3, :3] = np.eye(3)
    O[:3, 3:] = -so3.hat(s)
    return O

def odot_batch(p):
    assert p.shape[1] == 4 and np.all(p[:, 3] == 1.0)
    s = p[:, :3]
    O = np.zeros((p.shape[0], 4, 6))
    O[:, :3, :3] = np.eye(3)[None, :, :].repeat(p.shape[0], axis=0)
    O[:, :3, 3:] = -so3.hat_batch(s)
    return O
