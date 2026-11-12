from robot.sympy import *
import numpy as np


_gravity_torque_fn = sp.lambdify(q, G, modules="numpy")
def get_gravity_torque(q_values):
    q_values = np.asarray(q_values, dtype=float)
    if q_values.shape != (3,):
        raise ValueError("q_values must contain exactly [q1, q2, q3]")
    return np.asarray(_gravity_torque_fn(*q_values), dtype=float).reshape(3)
