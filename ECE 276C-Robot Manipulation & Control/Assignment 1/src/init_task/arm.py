import numpy as np

L1, L2, L3 = 0.5, 0.5, 0.2


def forward_kinematics(joint_angles):
    theta1, theta2, theta3 = joint_angles
    heading1 = theta1
    heading2 = theta1 + theta2
    heading3 = heading2 + theta3
    x = L1 * np.cos(heading1) + L2 * np.cos(heading2) + L3 * np.cos(heading3)
    y = L1 * np.sin(heading1) + L2 * np.sin(heading2) + L3 * np.sin(heading3)
    return np.array([x, y, heading3])


def inverse_kinematics(ee_pose):
    x, y, phi = ee_pose
    wrist_x = x - L3 * np.cos(phi)
    wrist_y = y - L3 * np.sin(phi)
    reach_sq = wrist_x**2 + wrist_y**2
    cosine_2 = (reach_sq - L1**2 - L2**2) / (2.0 * L1 * L2)
    cosine_2 = float(np.clip(cosine_2, -1.0, 1.0))
    sine_2 = np.sqrt(max(0.0, 1.0 - cosine_2**2))
    theta2 = np.arctan2(sine_2, cosine_2)
    theta1 = np.arctan2(wrist_y, wrist_x) - np.arctan2(L2 * sine_2, L1 + L2 * cosine_2)
    theta3 = phi - theta1 - theta2
    joint_angles = np.array([theta1, theta2, theta3])
    return joint_angles


def jacobian(joints):
    """2x3 position Jacobian of the planar arm."""
    theta1, theta2, theta3 = joints
    heading1 = theta1
    heading2 = theta1 + theta2
    heading3 = heading2 + theta3
    j11 = -L1 * np.sin(heading1) - L2 * np.sin(heading2) - L3 * np.sin(heading3)
    j12 = -L2 * np.sin(heading2) - L3 * np.sin(heading3)
    j13 = -L3 * np.sin(heading3)
    j21 = L1 * np.cos(heading1) + L2 * np.cos(heading2) + L3 * np.cos(heading3)
    j22 = L2 * np.cos(heading2) + L3 * np.cos(heading3)
    j23 = L3 * np.cos(heading3)
    return np.array(
        [
            [j11, j12, j13],
            [j21, j22, j23],
        ]
    )


def get_obs(joint_angles):
    return forward_kinematics(joint_angles)[:2]
