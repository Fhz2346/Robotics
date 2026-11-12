from init_task import *
from simulation import *


def damped_pinv(matrix, damping=0.02):
    gram = matrix @ matrix.T
    return matrix.T @ np.linalg.inv(gram + (damping ** 2) * np.eye(2))


KP = 12.0
KN = 8.0
CONTROL_DT = 0.005


U = lambda joint_angles: -np.cos(joint_angles[1]) - np.cos(joint_angles[2])
nabla_U = lambda joint_angles: np.array([0.0, np.sin(joint_angles[1]), np.sin(joint_angles[2])])


def control_step(joint_angles, ref_pos, ref_vel):
    obs_pos = get_obs(joint_angles)
    error = ref_pos - obs_pos
    dot_x1 = ref_vel + KP * error
    jac = jacobian(joint_angles)
    jac_pinv = damped_pinv(jac)
    dot_q1 = jac_pinv @ dot_x1
    dot_q2 = KN * nabla_U(joint_angles)
    dot_q_null = (np.eye(3) - jac_pinv @ jac) @ dot_q2
    dot_q = dot_q1 + dot_q_null
    return joint_angles + dot_q * CONTROL_DT
