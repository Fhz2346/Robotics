import hydra
import numpy as np
import cv2
import pybullet as p
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from Camera import Camera
from Pathes import P
from eye_in_hand_robot import eye_in_hand_robot
from init import init_environment
from controller.robot import RobotController


def mark_target(camera_bgr, workspace_bgr, target_uv, workspace_uv, error):
    """Mark the tracked object in both camera views and add a bottom caption."""
    target_px = tuple(int(value) for value in target_uv)
    workspace_px = tuple(int(value) for value in workspace_uv)
    center_px = (camera_bgr.shape[1] // 2, camera_bgr.shape[0] // 2)

    if 0 <= target_px[0] < camera_bgr.shape[1] and 0 <= target_px[1] < camera_bgr.shape[0]:
        cv2.drawMarker(camera_bgr, target_px, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.drawMarker(camera_bgr, center_px, (0, 220, 0), cv2.MARKER_CROSS, 18, 1)
    if 0 <= workspace_px[0] < workspace_bgr.shape[1] and 0 <= workspace_px[1] < workspace_bgr.shape[0]:
        cv2.drawMarker(workspace_bgr, workspace_px, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

    combined = np.hstack((camera_bgr, workspace_bgr))
    panel_top = combined.shape[0] - 54
    cv2.rectangle(combined, (0, panel_top), (combined.shape[1] - 1, combined.shape[0] - 1),
                  (24, 24, 24), -1)
    cv2.putText(combined, "Yellow: object center  |  Green: robot camera image center",
                (12, panel_top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(combined, f"Target pixel: ({target_px[0]}, {target_px[1]})  |  error: {error:.1f} px",
                (12, panel_top + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


@hydra.main(version_base=None, config_path=str(P.cfg_dir), config_name="scenario2")
def main(cfg: DictConfig):
    obj = init_environment(cfg.object)
    ws_camera = Camera(cfg.camera)
    robot = eye_in_hand_robot(cfg.robot)
    p.stepSimulation()
    controller = RobotController(robot, cfg.controller)
    ws_camera.set_pose(
        np.asarray(cfg.camera.init_pose.pos, dtype=float),
        np.asarray(cfg.camera.init_pose.orn, dtype=float),
    )
    out_dir = P.out_dir / "scenario2"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "new.gif"
    gif_frames = []
    try:
        for iteration in range(200):
            # Capture both views from the same simulation state.
            uv = robot.camera.world2pixel(obj.center)
            workspace_uv = ws_camera.world2pixel(obj.center)
            ws_camera.capture()
            camera_bgr = cv2.cvtColor(robot.camera.rgb, cv2.COLOR_RGB2BGR)
            workspace_bgr = cv2.cvtColor(ws_camera.rgb, cv2.COLOR_RGB2BGR)
            error = float(np.linalg.norm(np.asarray([robot.camera.W / 2, robot.camera.H / 2]) - uv))
            if iteration % 2 == 0:
                gif_frames.append(Image.fromarray(mark_target(camera_bgr, workspace_bgr, uv, workspace_uv, error)))

            if iteration % 20 == 0:
                print(f"frame={iteration:3d} target_px={uv}")

            controller.set_objective(obj.center, [robot.camera.W / 2, robot.camera.H / 2])
            dq = controller.control()
            robot.set_q(robot.get_q() + dq)
            obj.step()
            robot.step(20)
    finally:
        p.disconnect()

    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=1000 // 15,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Saved side-by-side GIF: {gif_path}")


if __name__ == "__main__":
    main()
