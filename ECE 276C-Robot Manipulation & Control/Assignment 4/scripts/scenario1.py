"""Q1/Q2 camera-based visual servoing simulation."""

import cv2
import hydra
import numpy as np
import pybullet as p
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as Rot

from Camera import Camera
from Pathes import P
from controller.camera import CameraController
from init import init_environment


@hydra.main(version_base=None, config_path=str(P.cfg_dir), config_name="scenario1")
def main(cfg: DictConfig):
    obj = init_environment(cfg.object)
    camera = Camera(cfg.camera)
    camera.set_pose(
        np.asarray(cfg.camera.init_pose.pos, dtype=float),
        np.asarray(cfg.camera.init_pose.orn, dtype=float),
    )

    controller = CameraController(camera, cfg.controller)
    controller.set_objective(obj.center, [camera.W / 2, camera.H / 2])

    P.out_dir.mkdir(parents=True, exist_ok=True)
    video_path = P.out_dir / "scenario1.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (camera.W, camera.H)
    )
    if not writer.isOpened():
        p.disconnect()
        raise RuntimeError(f"Could not create video file: {video_path}")

    try:
        for iteration in range(200):
            p.stepSimulation()
            camera.capture()
            uv = camera.world2pixel(obj.center)

            dx, domega = controller.control()
            camera.set_pose(
                camera.pos + dx,
                Rot.from_rotvec(domega).as_matrix() @ camera.orn,
            )

            writer.write(cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2BGR))
            if iteration % 20 == 0:
                print(f"frame={iteration:3d} target_px={uv} camera_pos={camera.pos.round(4)}")
    finally:
        writer.release()
        p.disconnect()

    print(f"Saved video: {video_path}")


if __name__ == "__main__":
    main()
