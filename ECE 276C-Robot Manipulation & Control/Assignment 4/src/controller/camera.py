import numpy as np
from Camera import Camera


def pinv(M, damping=1e-6):
    return M.T @ np.linalg.inv(M @ M.T + damping * np.eye(M.shape[0]))


class CameraController:

    def __init__(self, camera: Camera, cfg):
        self.camera = camera
        fov_rad = np.pi / 180 * self.camera.fov
        self.camera_f_px = 0.5 * self.camera.H / np.tan(0.5 * fov_rad)  # focal length in pixels

        self.K_p_x = cfg.K_p_x
        self.K_p_Omega = cfg.K_p_Omega
        self.K_p = np.diag([self.K_p_x] * 3 + [self.K_p_Omega] * 3)

    def set_objective(self, tgt_pos, tgt_uv):
        self.tgt_pos = tgt_pos
        self.tgt_uv = np.asarray(tgt_uv, dtype=float).reshape(2)

    def getImageJacobian(self, uv, Z):
        f = self.camera_f_px
        u, v = uv
        x = (u - self.camera.W / 2.0) / f
        y = (v - self.camera.H / 2.0) / f
        # Normalized image Jacobian, converted to pixel coordinates.
        L = np.array(
            [
                [-1 / Z, 0, x / Z, x * y, -(1 + x * x), y],
                [0, -1 / Z, y / Z, 1 + y * y, -x * y, -x],
            ]
        )
        return f * L

    def control(self):
        uv = self.camera.world2pixel(self.tgt_pos)
        depth = self.camera.depth[uv[1], uv[0]]
        error = self.tgt_uv - np.asarray(uv, dtype=float).reshape(2)
        L = self.getImageJacobian(uv, depth)
        twist_cam = self.K_p @ pinv(L) @ error
        delta_X = self.camera.orn @ twist_cam[:3]
        delta_Omega = self.camera.orn @ twist_cam[3:]
        return delta_X, delta_Omega
