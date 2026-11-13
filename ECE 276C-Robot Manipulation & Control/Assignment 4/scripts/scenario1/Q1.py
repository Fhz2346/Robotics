import cv2
import hydra
import numpy as np
import pybullet as p
from omegaconf import DictConfig
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

from Camera import Camera
from Pathes import P
from controller.camera import CameraController
from init import init_environment


TRANSLATION_STEP = 0.1  # metres
ROTATION_STEP = 0.05  # radians
FPS = 12


def world2pixel_float(camera: Camera, world_point):
    """Project a world point to subpixel image coordinates (u right, v down)."""
    point_h = np.append(np.asarray(world_point, dtype=float), 1.0)
    camera_point = camera.viewMat @ point_h
    ndc = camera.projMat @ camera_point
    ndc = ndc[:3] / ndc[3]
    return np.array(
        [((ndc[0] + 1.0) / 2.0) * camera.W,
         ((1.0 - ndc[1]) / 2.0) * camera.H],
        dtype=float,
    )


def add_overlay(rgb, lines, target_uv):
    """Draw a target marker and readable test details on a captured frame."""
    frame = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    u, v = np.rint(target_uv).astype(int)
    cv2.drawMarker(frame, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    # A solid panel keeps each frame's labels legible and prevents old text
    # from showing through when the GIF encoder stores frame deltas.
    panel_height = 12 + 25 * len(lines)
    panel_top = frame.shape[0] - panel_height
    cv2.rectangle(
        frame, (0, panel_top), (frame.shape[1] - 1, frame.shape[0] - 1),
        (24, 24, 24), -1,
    )
    y = panel_top + 22
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 25
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@hydra.main(version_base=None, config_path=str(P.cfg_dir), config_name="scenario1")
def main(cfg: DictConfig):
    obj = init_environment(cfg.object)
    camera = Camera(cfg.camera)
    initial_pos = np.asarray(cfg.camera.init_pose.pos, dtype=float)
    initial_orn = np.asarray(cfg.camera.init_pose.orn, dtype=float)
    camera.set_pose(initial_pos.copy(), initial_orn.copy())

    controller = CameraController(camera, cfg.controller)
    center_uv = world2pixel_float(camera, obj.center)
    center_cam = camera.viewMat @ np.append(np.asarray(obj.center, dtype=float), 1.0)
    beta_center = float(abs(center_cam[2]))  # positive optical-axis depth
    if not np.isfinite(beta_center) or beta_center <= 0:
        p.disconnect()
        raise RuntimeError(f"Invalid camera-frame depth for target center: {beta_center}")

    # getImageJacobian maps camera-frame camera velocity to pixel velocity.
    # Use the same 3D point tracked by the finite-difference measurement.
    # The depth image at the center pixel describes the visible surface instead.
    jacobian_camera = controller.getImageJacobian(center_uv, beta_center)
    world_to_camera_twist = np.zeros((6, 6), dtype=float)
    world_to_camera_twist[:3, :3] = initial_orn.T
    world_to_camera_twist[3:, 3:] = initial_orn.T
    jacobian_world = jacobian_camera @ world_to_camera_twist

    tests = [(i, TRANSLATION_STEP) for i in range(3)] + [
        (i, ROTATION_STEP) for i in range(3, 6)
    ]
    labels = ["+X translation", "+Y translation", "+Z translation",
              "+X rotation", "+Y rotation", "+Z rotation"]
    gif_frames = []

    try:
        for axis, step in tests:
            # Each axis starts from the same camera pose.
            camera.set_pose(initial_pos.copy(), initial_orn.copy())
            baseline_uv = world2pixel_float(camera, obj.center)
            baseline_rgb = camera.rgb.copy()
            predicted_delta = jacobian_world[:, axis] * step

            if axis < 3:
                final_pos = initial_pos + np.eye(3)[axis] * step
                final_orn = initial_orn.copy()
            else:
                world_axis = np.eye(3)[axis - 3]
                final_pos = initial_pos.copy()
                final_orn = Rot.from_rotvec(world_axis * step).as_matrix() @ initial_orn

            camera.set_pose(final_pos, final_orn)
            observed_uv = world2pixel_float(camera, obj.center)
            observed_delta = observed_uv - baseline_uv
            cosine = float(
                np.dot(predicted_delta, observed_delta)
                / (np.linalg.norm(predicted_delta) * np.linalg.norm(observed_delta) + 1e-12)
            )
            direction_ok = cosine > 0.8
            print(
                f"{labels[axis]:16s} predicted=[{predicted_delta[0]:7.3f}, {predicted_delta[1]:7.3f}] px, "
                f"observed=[{observed_delta[0]:7.3f}, {observed_delta[1]:7.3f}] px, "
                f"direction={'OK' if direction_ok else 'CHECK'} (cos={cosine:.3f})"
            )

            # Show baseline, gradual positive motion, and the measured result.
            title = labels[axis]
            for _ in range(4):
                gif_frames.append(Image.fromarray(add_overlay(
                    baseline_rgb,
                    [f"{title}  |  reset pose", "Yellow cross: projected target center"],
                    baseline_uv,
                )))
            for fraction in np.linspace(0.1, 1.0, 10):
                if axis < 3:
                    intermediate_pos = initial_pos + np.eye(3)[axis] * step * fraction
                    intermediate_orn = initial_orn.copy()
                else:
                    intermediate_pos = initial_pos.copy()
                    intermediate_orn = Rot.from_rotvec(
                        np.eye(3)[axis - 3] * step * fraction
                    ).as_matrix() @ initial_orn
                camera.set_pose(intermediate_pos, intermediate_orn)
                current_uv = world2pixel_float(camera, obj.center)
                gif_frames.append(Image.fromarray(add_overlay(
                    camera.rgb,
                    [f"{title}  |  {fraction:.0%} of test motion",
                     f"predicted d( u,v ) = ({predicted_delta[0]:.2f}, {predicted_delta[1]:.2f}) px"],
                    current_uv,
                )))
            # Hold the final comparison long enough to read it (~2.5 s).
            for _ in range(30):
                gif_frames.append(Image.fromarray(add_overlay(
                    camera.rgb,
                    [f"{title}  |  prediction / observed",
                     f"({predicted_delta[0]:.2f}, {predicted_delta[1]:.2f}) / "
                     f"({observed_delta[0]:.2f}, {observed_delta[1]:.2f}) px",
                     f"Direction check: {'PASS' if direction_ok else 'CHECK'}"],
                    observed_uv,
                )))

        output_path = P.out_dir / "scenario1" / "Q1.gif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=1000 // FPS,
            loop=0,
            optimize=False,
            disposal=2,
        )
        print(f"Saved GIF: {output_path}")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
