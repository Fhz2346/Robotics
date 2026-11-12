import os

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

# Joint configurations from the assignment, in radians: [theta1, theta2, theta3]
CONFIGS = [
    ("1", np.array([0.0, 0.0, np.pi / 2])),
    ("2", np.array([0.0, np.pi, 0.0])),
    ("3", np.array([np.pi / 2, np.pi / 2, np.pi / 4])),
    ("4", np.array([np.pi / 3, np.pi / 2, 0.0])),
]

def format_angles(angles):
    labels = {
        0.0: "0",
        0.25: r"\pi/4",
        1.0 / 3.0: r"\pi/3",
        0.5: r"\pi/2",
        1.0: r"\pi",
    }
    pieces = []
    for value in angles:
        key = min(labels, key=lambda candidate: abs(candidate - value / np.pi))
        pieces.append(labels[key])
    return "[" + ", ".join(pieces) + "]"


def arm_points(robot):
    """xy positions of the base and every link frame, read from the simulation."""
    base_position, _ = p.getBasePositionAndOrientation(robot)
    points = [base_position[:2]]
    for index in range(p.getNumJoints(robot)):
        state = p.getLinkState(robot, index, computeForwardKinematics=True)
        points.append(state[4][:2])
    return np.vstack(points)


def capture(robot, movable_joints, name, angles, path):
    for joint, angle in zip(movable_joints, angles):
        p.resetJointState(robot, joint, float(angle))

    points = arm_points(robot)
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    # Pad the arm's footprint so joint posts and the oblique view stay in frame.
    span = max(float(np.max(points.max(axis=0) - points.min(axis=0))), 0.55) + 0.45
    distance = span * 1.85
    eye = [center[0] + 0.06 * span, center[1] - 0.16 * span, distance]
    target = [center[0], center[1], 0.06]
    view = p.computeViewMatrix(eye, target, [0.0, 0.0, 1.0])
    projection = p.computeProjectionMatrixFOV(40.0, 1.0, 0.02, 12.0)

    width = height = 900
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=projection,
        lightDirection=[0.4, 0.6, 1.0],
        renderer=p.ER_TINY_RENDERER,
    )
    image = np.reshape(rgba, (height, width, 4))[:, :, :3]

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.imshow(image)
    ax.set_title(
        rf"Configuration {name}:  $\theta = {format_angles(angles)}$",
        fontsize=13,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(out_dir)

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")
    # Lift the arm just above the ground so the cylinders are not clipped.
    robot = p.loadURDF("three-link-robot.urdf", [0.0, 0.0, 0.08], useFixedBase=True)

    movable_joints = []
    for index in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, index)
        if info[2] != p.JOINT_FIXED:
            movable_joints.append(index)
            p.setJointMotorControl2(robot, index, p.VELOCITY_CONTROL, force=0.0)

    os.makedirs(os.path.join(out_dir, "output", "Q3"), exist_ok=True)
    for name, angles in CONFIGS:
        path = os.path.join(out_dir, "output", "Q3", f"config{name}.png")
        capture(robot, movable_joints, name, angles, path)
        print(f"saved {path}  theta = {angles}")

    p.disconnect()


if __name__ == "__main__":
    main()
