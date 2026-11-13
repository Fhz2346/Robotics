"""Scenario 2 visual servoing with nullspace camera-roll control."""
# roll=0 means 地平线水平
import cv2
import hydra
import numpy as np
import pybullet as p
from omegaconf import DictConfig
from PIL import Image

from Camera import Camera
from Pathes import P
from controller.robot import RobotNullController
from eye_in_hand_robot import eye_in_hand_robot
from init import init_environment


FPS = 15
ITERATIONS = 200


def mark_target(camera_bgr, workspace_bgr, target_uv, workspace_uv, error, roll):
    """Mark the target in both views and show tracking and roll values."""
    target_px = tuple(int(value) for value in target_uv)
    workspace_px = tuple(int(value) for value in workspace_uv)
    image_center = (camera_bgr.shape[1] // 2, camera_bgr.shape[0] // 2)

    if 0 <= target_px[0] < camera_bgr.shape[1] and 0 <= target_px[1] < camera_bgr.shape[0]:
        cv2.drawMarker(camera_bgr, target_px, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.drawMarker(camera_bgr, image_center, (0, 220, 0), cv2.MARKER_CROSS, 18, 1)
    if 0 <= workspace_px[0] < workspace_bgr.shape[1] and 0 <= workspace_px[1] < workspace_bgr.shape[0]:
        cv2.drawMarker(workspace_bgr, workspace_px, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

    combined = np.hstack((camera_bgr, workspace_bgr))
    panel_height = 78
    panel_top = combined.shape[0] - panel_height
    cv2.rectangle(
        combined, (0, panel_top), (combined.shape[1] - 1, combined.shape[0] - 1),
        (24, 24, 24), -1,
    )
    lines = [
        "Yellow: object center  |  Green: robot camera image center",
        f"Target pixel: ({target_px[0]}, {target_px[1]})  |  error: {error:.1f} px",
        f"Camera roll relative to ground: {roll:+.4f} rad ({np.degrees(roll):+.2f} deg)",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            combined, line, (12, panel_top + 20 + 24 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


@hydra.main(version_base=None, config_path=str(P.cfg_dir), config_name="scenario2")
def main(cfg: DictConfig):
    obj = init_environment(cfg.object)
    ws_camera = Camera(cfg.camera)
    robot = eye_in_hand_robot(cfg.robot)
    p.stepSimulation()
    controller = RobotNullController(robot, cfg.controller)
    controller.set_objective(obj.center, [robot.camera.W / 2, robot.camera.H / 2])

    ws_camera.set_pose(
        np.asarray(cfg.camera.init_pose.pos, dtype=float),
        np.asarray(cfg.camera.init_pose.orn, dtype=float),
    )
    out_dir = P.out_dir / "scenario2"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "0roll.gif"
    frames = []

    try:
        for iteration in range(ITERATIONS):
            uv = robot.camera.world2pixel(obj.center)
            workspace_uv = ws_camera.world2pixel(obj.center)
            ws_camera.capture()

            error = float(np.linalg.norm(np.array([robot.camera.W / 2, robot.camera.H / 2]) - uv))
            roll = controller._camera_roll(robot.camera.orn)
            if iteration % 2 == 0:
                camera_bgr = cv2.cvtColor(robot.camera.rgb, cv2.COLOR_RGB2BGR)
                workspace_bgr = cv2.cvtColor(ws_camera.rgb, cv2.COLOR_RGB2BGR)
                frame_rgb = mark_target(
                    camera_bgr, workspace_bgr, uv, workspace_uv, error, roll,
                )
                frames.append(Image.fromarray(frame_rgb))

            if iteration % 20 == 0:
                print(
                    f"frame={iteration:3d} target_px={uv} "
                    f"error={error:.1f}px roll={roll:+.4f}rad"
                )

            dq = controller.control()
            robot.set_q(robot.get_q() + dq)
            obj.step()
            robot.step(20)
    finally:
        p.disconnect()

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // FPS,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Saved nullspace-control GIF: {gif_path}")


if __name__ == "__main__":
    main()
