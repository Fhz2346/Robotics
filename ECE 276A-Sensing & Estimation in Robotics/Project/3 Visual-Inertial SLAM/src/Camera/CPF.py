import numpy as np


def CPF(q):  # canonical projection function
    return q / q[:, 2:3]


def CPF_Jacobian(q):
    J = np.eye(4)
    J[:,2] = -q / q[2]
    J[2,2] = 0
    J /= q[2]
    return J

def CPF_Jacobian_batch(q):
    N = q.shape[0]
    J = np.zeros((N, 4, 4))

    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    z2 = z ** 2

    # row 0
    J[:, 0, 0] = 1 / z
    J[:, 0, 2] = -x / z2

    # row 1
    J[:, 1, 1] = 1 / z
    J[:, 1, 2] = -y / z2

    # row 2 (z/z = 1 → derivative = 0)
    # already zero

    # row 3
    J[:, 3, 2] = -w / z2
    J[:, 3, 3] = 1 / z

    return J