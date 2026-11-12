import numpy as np


def hat(v):
    v_hat = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return v_hat

def hat_batch(v):
    N = v.shape[0]
    v_hat = np.zeros((N, 3, 3))
    v_hat[:, 0, 1] = -v[:, 2]
    v_hat[:, 0, 2] = v[:, 1]
    v_hat[:, 1, 0] = v[:, 2]
    v_hat[:, 1, 2] = -v[:, 0]
    v_hat[:, 2, 0] = -v[:, 1]
    v_hat[:, 2, 1] = v[:, 0]
    return v_hat

def vee(v_hat):
    v = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]])
    return v


def exp(v_hat):
    theta = np.linalg.norm(vee(v_hat))
    if theta < 1e-8:
        return np.eye(3) + v_hat + 0.5 * (v_hat @ v_hat)

    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta**2)
    return np.eye(3) + A * v_hat + B * (v_hat @ v_hat)


def AA2SO3(omega):
    return exp(hat(omega))


def left_jacobian(v):
    theta = np.linalg.norm(v)
    v_hat = hat(v)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * v_hat + (1.0 / 6.0) * (v_hat @ v_hat)
    A = (1.0 - np.cos(theta)) / (theta**2)
    B = (theta - np.sin(theta)) / (theta**3)
    return np.eye(3) + A * v_hat + B * (v_hat @ v_hat)
