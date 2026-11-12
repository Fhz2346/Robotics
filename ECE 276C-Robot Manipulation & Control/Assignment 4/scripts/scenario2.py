import hydra
import numpy as np
import cv2
import pybullet as p
from omegaconf import DictConfig

from Camera import Camera
from Pathes import P
from eye_in_hand_robot import eye_in_hand_robot
from init import init_environment
from controller.robot import RobotController


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
    if obj.is_moving:
        video_path = out_dir / "moving.mp4"
    else:
        video_path = out_dir / "static.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (robot.camera.W + ws_camera.W, robot.camera.H),
    )
    try:
        for iteration in range(200):
            # Capture both views from the same simulation state.
            uv = robot.camera.world2pixel(obj.center)
            ws_camera.capture()
            camera_bgr = cv2.cvtColor(robot.camera.rgb, cv2.COLOR_RGB2BGR)
            workspace_bgr = cv2.cvtColor(ws_camera.rgb, cv2.COLOR_RGB2BGR)
            writer.write(np.hstack((camera_bgr, workspace_bgr)))

            if iteration % 20 == 0:
                print(f"frame={iteration:3d} target_px={uv}")

            controller.set_objective(obj.center, [robot.camera.W / 2, robot.camera.H / 2])
            dq = controller.control()
            robot.set_q(robot.get_q() + dq)
            obj.step()
            robot.step(20)
    finally:
        writer.release()
        p.disconnect()
    print(f"Saved side-by-side video: {video_path}")


if __name__ == "__main__":
    main()
