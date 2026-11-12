import numpy as np
import sympy as sp

from robot.sympy import G, J_tip, q

_tip_jacobian_fn = sp.lambdify(tuple(q), J_tip, modules="numpy")
_gravity_torque_fn = sp.lambdify(tuple(q), G, modules="numpy")


def jacobian_and_gravity(q_values):
    q_values = np.asarray(q_values, dtype=float)
    if q_values.shape != (3,):
        raise ValueError("q_values must contain exactly [q1, q2, q3]")

    args = tuple(q_values)
    jacobian = np.asarray(_tip_jacobian_fn(*args), dtype=float).reshape(3, 3)
    gravity = np.asarray(_gravity_torque_fn(*args), dtype=float).reshape(3)
    return jacobian, gravity
