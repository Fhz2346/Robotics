import numpy as np
from .camera import CameraController, pinv
from eye_in_hand_robot import eye_in_hand_robot


class RobotController:

    def __init__(self, robot: eye_in_hand_robot, cfg):
        self.robot = robot
        self.dq_limit = cfg.dq_limit
        self.camera_controller = CameraController(robot.camera, cfg.camera)

    def set_objective(self, tgt_pos, tgt_uv):
        self.camera_controller.set_objective(tgt_pos, tgt_uv)

    def control(self):
        dx, domega = self.camera_controller.control()
        desired_twist = np.r_[dx, domega]
        J = self.robot.get_jacobian()
        dq = pinv(J, damping=1e-3) @ desired_twist
        # Bound per-frame joint motion to prevent oscillation near singularities.
        dq = np.clip(dq, -self.dq_limit, self.dq_limit)
        return dq
