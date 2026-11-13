"""Run the three scenario 1 visual-servoing modes and save GIFs."""

import cv2
import hydra
import numpy as np
import pybullet as p
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

from Camera import Camera
from Pathes import P
from controller.camera import CameraController
from init import init_environment


FPS = 12
MAX_ITERATIONS = 200
PIXEL_TOLERANCE = 1.5


def make_frame(camera, uv, error, label):
    """Capture the camera image with a small bottom caption and markers."""
    frame = cv2.cvtColor(camera.rgb.copy(), cv2.COLOR_RGB2BGR)
    target = (camera.W // 2, camera.H // 2)
    object_px = tuple(int(value) for value in uv)
    cv2.drawMarker(frame, target, (0, 220, 0), cv2.MARKER_CROSS, 18, 1)
    if 0 <= object_px[0] < camera.W and 0 <= object_px[1] < camera.H:
        cv2.drawMarker(frame, object_px, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

    lines = [label, f"Target pixel: ({uv[0]}, {uv[1]})  |  error: {error:.1f} px"]
    panel_height = 62
    panel_top = camera.H - panel_height
    cv2.rectangle(frame, (0, panel_top), (camera.W - 1, camera.H - 1), (24, 24, 24), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, panel_top + 24 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def run_mode(camera, obj, cfg, initial_pos, initial_orn, label, enabled_translation,
             enabled_rotation, output_path):
    """Servo from the common initial pose and save one mode's animation."""
    camera.set_pose(initial_pos.copy(), initial_orn.copy())
    cfg.controller.is_enabled.translation = enabled_translation
    cfg.controller.is_enabled.rotation = enabled_rotation
    controller = CameraController(camera, cfg.controller)
    controller.set_objective(obj.center, [camera.W / 2, camera.H / 2])

    uv = camera.world2pixel(obj.center)
    error = float(np.linalg.norm(np.asarray([camera.W / 2, camera.H / 2]) - uv))
    frames = [make_frame(camera, uv, error, label)]
    converged = error <= PIXEL_TOLERANCE

    for iteration in range(MAX_ITERATIONS):
        if converged:
            break
        p.stepSimulation()
        camera.capture()
        dx, domega = controller.control()
        camera.set_pose(
            camera.pos + dx,
            Rot.from_rotvec(domega).as_matrix() @ camera.orn,
        )

        uv = camera.world2pixel(obj.center)
        error = float(np.linalg.norm(np.asarray([camera.W / 2, camera.H / 2]) - uv))
        frames.append(make_frame(camera, uv, error, label))
        converged = error <= PIXEL_TOLERANCE

        if (iteration + 1) % 40 == 0:
            print(f"{label}: iteration={iteration + 1}, target_px={uv}, error={error:.2f}px")

    if converged:
        # Leave the final centered view on screen long enough to inspect.
        frames.extend([frames[-1]] * 18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // FPS,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"{label}: {'centered' if converged else 'max iterations reached'}; saved {output_path}")


@hydra.main(version_base=None, config_path=str(P.cfg_dir), config_name="scenario1")
def main(cfg: DictConfig):
    obj = init_environment(cfg.object)
    camera = Camera(cfg.camera)
    initial_pos = np.asarray(cfg.camera.init_pose.pos, dtype=float)
    initial_orn = np.asarray(cfg.camera.init_pose.orn, dtype=float)
    output_dir = P.out_dir / "scenario1" / "Q2"

    modes = [
        ("Translation only", True, False, "translation_only.gif"),
        ("Rotation only", False, True, "rotation_only.gif"),
        ("Translation + rotation", True, True, "translation_and_rotation.gif"),
    ]
    try:
        for label, use_translation, use_rotation, filename in modes:
            run_mode(
                camera, obj, cfg, initial_pos, initial_orn, label,
                use_translation, use_rotation, output_dir / filename,
            )
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
