import numpy as np
from scipy.spatial.transform import Rotation as Rot
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

class RobotNullController:
    """Image-based tracking with a camera-leveling secondary task."""

    def __init__(self, robot: eye_in_hand_robot, cfg):
        self.robot = robot
        self.dq_limit = cfg.dq_limit
        self.roll_gain = float(cfg.get("roll_gain", 0.02))
        self.camera_controller = CameraController(robot.camera, cfg.camera)

    def set_objective(self, tgt_pos, tgt_uv):
        self.camera_controller.set_objective(tgt_pos, tgt_uv)

    def control(self):
        dx, domega = self.camera_controller.control()
        desired_twist = np.r_[dx, domega]
        J_robot = self.robot.get_jacobian()
        dq_primary = pinv(J_robot, damping=1e-3) @ desired_twist

        camera = self.robot.camera
        uv = camera.world2pixel(self.camera_controller.tgt_pos)
        depth = camera.depth[uv[1], uv[0]]
        L_image = self.camera_controller.getImageJacobian(uv, depth)
        if not self.camera_controller.is_translation:
            L_image[:, :3] = 0.0
        if not self.camera_controller.is_rotation:
            L_image[:, 3:] = 0.0

        # J_robot is expressed in world coordinates; the image Jacobian expects
        # camera-frame twist, so rotate both linear and angular rows into camera.
        R_cw = camera.orn.T
        world_to_camera = np.zeros((6, 6), dtype=float)
        world_to_camera[:3, :3] = R_cw
        world_to_camera[3:, 3:] = R_cw
        J_image_joint = L_image @ world_to_camera @ J_robot

        # Minimize half the squared roll angle while preserving image motion.
        image_pinv = np.linalg.pinv(J_image_joint, rcond=1e-5)
        nullspace = np.eye(J_robot.shape[1]) - image_pinv @ J_image_joint
        roll = self._camera_roll(camera.orn)
        roll_gradient = self._roll_gradient(camera.orn, J_robot[3:, :])
        dq_secondary = -self.roll_gain * nullspace @ (roll * roll_gradient)
        dq = dq_primary + dq_secondary

        # Bound per-frame joint motion to prevent oscillation near singularities.
        return np.clip(dq, -self.dq_limit, self.dq_limit)

    @staticmethod
    def _camera_roll(camera_orn):
        """Signed roll relative to the projection of world up onto the image plane."""
        world_up = np.array([0.0, 0.0, 1.0])
        forward = camera_orn[:, 2]
        camera_up = -camera_orn[:, 1]
        level_up = world_up - np.dot(world_up, forward) * forward
        norm = np.linalg.norm(level_up)
        if norm < 1e-8:
            return 0.0
        level_up /= norm
        return float(np.arctan2(
            np.dot(forward, np.cross(level_up, camera_up)),
            np.dot(level_up, camera_up),
        ))

    @classmethod
    def _roll_gradient(cls, camera_orn, angular_jacobian, epsilon=1e-5):
        """Finite-difference roll derivative with respect to each joint angle."""
        gradient = np.zeros(angular_jacobian.shape[1], dtype=float)
        for joint in range(gradient.size):
            delta = angular_jacobian[:, joint] * epsilon
            R_plus = Rot.from_rotvec(delta).as_matrix() @ camera_orn
            R_minus = Rot.from_rotvec(-delta).as_matrix() @ camera_orn
            roll_plus = cls._camera_roll(R_plus)
            roll_minus = cls._camera_roll(R_minus)
            wrapped_difference = np.arctan2(
                np.sin(roll_plus - roll_minus),
                np.cos(roll_plus - roll_minus),
            )
            gradient[joint] = wrapped_difference / (2.0 * epsilon)
        return gradient
