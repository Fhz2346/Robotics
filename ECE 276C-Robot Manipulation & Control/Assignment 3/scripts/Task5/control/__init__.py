import numpy as np

from robot.pybullet import get_q_dq, get_p_tip, cfgR
from .fd import jacobian_and_gravity as jacobian_and_gravity_fd
from .analytical import jacobian_and_gravity as jacobian_and_gravity_analytical

if not cfgR.payload:
    KP = 35.0  # N/m
    KD = 8.0  # N*s/m
    JOINT_DAMPING = 0.015  # N*m*s/rad
    MAX_TORQUE = 1.0  # N*m per joint
else:
    KP = 35.0  # N/m; retain the Task 5 stiffness
    KD = 15.0  # N*s/m; add damping for the heavier endpoint
    JOINT_DAMPING = 0.05  # N*m*s/rad
    MAX_TORQUE = 20.0  # N*m; allow payload gravity compensation


def get_torque(x_ref, jacobian_and_gravity_fn):
    q, dq = get_q_dq()
    x = get_p_tip()
    J, grav = jacobian_and_gravity_fn(q)
    dx = J @ dq
    error = x_ref - x
    # Cartesian spring-damper force mapped to joint torques.
    F = KP * error - KD * dx
    tau_task = J.T @ F
    tau = tau_task + grav - JOINT_DAMPING * dq
    tau = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)
    return tau
