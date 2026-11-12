import numpy as np
import aspace as As

# mechanical space : angle space + joint limits
THETA2_LIMIT_DEG = 0.97 * 180
THETA2_LIMIT_RAD = 0.97 * np.pi
is_theta2_valid_deg = lambda q: -THETA2_LIMIT_DEG < q[1] < THETA2_LIMIT_DEG
is_theta2_valid_rad = lambda q: -THETA2_LIMIT_RAD < float(As.wrap(q[1])) < THETA2_LIMIT_RAD


def uniform_sample(rng: np.random.Generator):
    q = [
        rng.uniform(-np.pi, np.pi),
        rng.uniform(-THETA2_LIMIT_RAD, THETA2_LIMIT_RAD),
    ]
    return np.array(q)
