import pybullet_data
from PIL import Image

from init_task import *
from plot import *
from simulation import *
from render import *
from Pathes import P


def build_trajectory(num_frames):
    times = np.linspace(0.0, DURATION, num_frames)
    num_segments = len(WAYPOINTS) - 1
    segment_duration = DURATION / num_segments
    ee_poses, joint_angles = [], []
    for time in times:
        segment_idx = min(int(time / segment_duration), num_segments - 1)
        tau = (time - segment_idx * segment_duration) / segment_duration # in [0, 1]
        ee_pose = segment_pose((WAYPOINTS[segment_idx], WAYPOINTS[segment_idx + 1]), tau)
        ee_poses.append(ee_pose)
        joint_angles.append(inverse_kinematics(ee_pose))
    return times, np.vstack(ee_poses), np.vstack(joint_angles)


if __name__ == "__main__":
    out_dir = P.out_dir / "Q5"
    out_dir.mkdir(parents=True, exist_ok=True)
    num_frames = int(DURATION * FPS) + 1
    times, poses, joint_path = build_trajectory(num_frames)
    fk_error = np.max(np.linalg.norm(np.vstack([forward_kinematics(q) for q in joint_path]) - poses, axis=1))
    print(f"max IK position error: {fk_error:.3e} m")

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
    view, projection = make_camera(robot, joint_indices, joint_path)

    width = height = 520
    frames = []
    measured = []
    for joints in joint_path:
        apply_joints(robot, joint_indices, joints)
        measured.append(end_effector_xy(robot))
        frames.append(Image.fromarray(render(robot, view, projection, width, height)))
    measured = np.vstack(measured)
    p.disconnect()

    line_error = np.max(np.linalg.norm(measured - poses[:, :2], axis=1))
    print(f"max simulated path error: {line_error:.3e} m")

    gif_path = out_dir / "open_loop.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
    )
    plot_path(times, measured, out_dir / "ee_path.png")
    plot_profile(out_dir / "smooth_profile.png")
    print(f"saved {gif_path}")
