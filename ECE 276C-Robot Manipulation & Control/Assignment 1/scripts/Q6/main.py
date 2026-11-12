import pybullet_data
from PIL import Image
import os

from control import *
from render import *
from plot import *
from Pathes import P


def simulate():
    num_steps = int(round(DURATION / CONTROL_DT))
    times = np.linspace(0.0, DURATION, num_steps + 1)
    joint_angles = np.zeros(3)
    joint_trajectory = [joint_angles.copy()]
    obs_trajectory = [get_obs(joint_angles)]
    for time in times[:-1]:
        ref_pos, ref_vel = reference(time + CONTROL_DT)
        joint_angles = control_step(joint_angles, ref_pos, ref_vel)
        joint_trajectory.append(joint_angles.copy())
        obs_trajectory.append(get_obs(joint_angles))
    return times, np.vstack(joint_trajectory), np.vstack(obs_trajectory)


if __name__ == "__main__":
    out_dir = P.out_dir / "Q6"
    out_dir.mkdir(parents=True, exist_ok=True)

    times, joint_path, measured = simulate()
    tracking_error = plot_results(times, measured, out_dir / "ee_path.png")
    print(f"max tracking error: {tracking_error:.3e} m")

    sample_step = int(round((1.0 / FPS) / CONTROL_DT))
    sampled_joints = joint_path[::sample_step]
    if sampled_joints.shape[0] < int(DURATION * FPS) + 1:
        sampled_joints = np.vstack([sampled_joints, joint_path[-1]])

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF(
        str(P.assets_dir / "three-link-robot.urdf"),
        [0.0, 0.0, 0.08],
        useFixedBase=True,
    )
    joint_indices = movable_joint_indices(robot)
    view, projection = make_camera(robot, joint_indices, sampled_joints)

    width = height = 520
    frames = []
    for joints in sampled_joints:
        apply_joints(robot, joint_indices, joints)
        measured_xy = end_effector_xy(robot)
        frames.append(Image.fromarray(render(robot, view, projection, width, height)))
        if not np.allclose(measured_xy, forward_kinematics(joints)[:2], atol=1e-5):
            print("warning: simulated pose differs from FK", measured_xy)
    p.disconnect()

    gif_path = os.path.join(out_dir, "closed_loop.gif")
    frame_ms = int(round(1000 * sample_step * CONTROL_DT))
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
    print(f"saved {gif_path}  frames={len(frames)}  frame_ms={frame_ms}")
