import numpy as np
from Robot import Robot
from Camera.CPF import CPF_Jacobian, CPF_Jacobian_batch
from Geometric.Basics import odot, augment, odot_batch
import Geometric.SE3 as se3


class OdometryEKF:

    def __init__(self, rbt: Robot, sigma_v, sigma_w):
        self.rbt = rbt
        self.W = np.diag([
            sigma_v**2, sigma_v**2, sigma_v**2,
            sigma_w**2, sigma_w**2, sigma_w**2
        ])
        self.mu = np.eye(4)
        self.Sigma = 1e-6 * np.eye(6)
        self.F = np.eye(6)

    def predict(self, u, dt):
        dT = se3.twist2SE3(u * dt)
        self.mu = self.mu @ dT
        F = se3.Ad(dT.T)
        self.Sigma = F @ self.Sigma @ F.T + self.W * (dt**2)

    def predict1(self, u, dt):
        dT = se3.twist2SE3(u * dt)
        self.mu = self.mu @ dT
        self.F = se3.Ad(dT.T)

    def stereo_camera_jacobian(self, m):
        w_m = np.hstack([m, 1.0])
        i_m = self.rbt.iTw @ w_m
        o_m = self.rbt.oTw @ w_m

        J_pi = CPF_Jacobian(o_m)
        if J_pi is None:
            return None

        H = -self.rbt.cam.Ks @ J_pi @ self.rbt.o_T_i @ odot(i_m)
        return H

    def stereo_camera_jacobian_batch(self, m):
        H = np.zeros((m.shape[0], 4, 6))
        w_m = augment(m)
        i_m = (self.rbt.iTw @ w_m.T).T
        o_m = (self.rbt.oTw @ w_m.T).T
        mask = o_m[:, 2] > 1e-8
        if mask.sum() == 0:
            return None
        J = CPF_Jacobian_batch(o_m[mask])
        H_ = -self.rbt.cam.Ks @ J @ self.rbt.o_T_i @ odot_batch(i_m[mask])
        H[mask] = H_
        return H, mask

    def update(self, ):
        # Implement the update step of the EKF
        pass
