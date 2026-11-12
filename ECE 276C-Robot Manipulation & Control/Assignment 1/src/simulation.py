import pybullet as p
import numpy as np


def movable_joint_indices(robot):
    indices = []
    for index in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, index)[2] != p.JOINT_FIXED:
            indices.append(index)
            p.setJointMotorControl2(robot, index, p.VELOCITY_CONTROL, force=0.0)
    return indices


def apply_joints(robot, joint_indices, joints):
    for index, angle in zip(joint_indices, joints):
        p.resetJointState(robot, index, float(angle))


def end_effector_xy(robot):
    end_effector = p.getNumJoints(robot) - 1
    state = p.getLinkState(robot, end_effector, computeForwardKinematics=True)
    return np.array(state[4][:2])
