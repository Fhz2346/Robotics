import numpy as np
import pybullet as p
from simulation import apply_joints


def make_camera(robot, joint_indices, joint_path):
    samples = []
    for joints in joint_path[::5]:
        apply_joints(robot, joint_indices, joints)
        base, _ = p.getBasePositionAndOrientation(robot)
        samples.append(base[:2])
        for index in range(p.getNumJoints(robot)):
            state = p.getLinkState(robot, index, computeForwardKinematics=True)
            samples.append(state[4][:2])
    points = np.vstack(samples)
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    span = max(float(np.max(points.max(axis=0) - points.min(axis=0))), 0.8) + 0.35
    distance = span * 1.7
    eye = [center[0] + 0.05 * span, center[1] - 0.18 * span, distance]
    target = [center[0], center[1], 0.05]
    view = p.computeViewMatrix(eye, target, [0.0, 0.0, 1.0])
    projection = p.computeProjectionMatrixFOV(42.0, 1.0, 0.02, 12.0)
    return view, projection


def render(robot, view, projection, width, height):
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        lightDirection=[0.4, 0.6, 1.0],
        renderer=p.ER_TINY_RENDERER,
    )
    return np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
