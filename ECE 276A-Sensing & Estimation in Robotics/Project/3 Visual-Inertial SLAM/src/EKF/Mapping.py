import numpy as np
from Robot import Robot
from Camera.CPF import CPF_Jacobian, CPF_Jacobian_batch
import Geometric.SE3 as se3
from Geometric.Basics import augment

class MappingEKF:

    def __init__(self, rbt: Robot, sigma_obs, init_var_xy, init_var_z, M):
        self.rbt = rbt
        self.V = (sigma_obs**2) * np.eye(4)
        self.sigma_obs = sigma_obs
        self.R = np.diag(np.array([init_var_xy, init_var_xy, init_var_z]))
        self.M = M
        self.D = 3 * self.M
        self.mu = np.full((self.M, 3), np.nan)
        self.Sigma = np.zeros((self.D, self.D))
        self.initialized_LM = np.zeros(self.M, dtype=bool)
        self.P = np.zeros((3,4))
        self.P[:3, :3] = np.eye(3)

    def get_landmarks(self):
        return self.mu[self.initialized_LM]

    def initialize_LM(self, z, wTi):
        mask_observed = np.all(z >= 0, axis=1)
        if mask_observed.sum() == 0:
            return []
        indices_observed = np.where(mask_observed)[0]
        indices_update = []
        for i in indices_observed:
            if not self.initialized_LM[i]:
                m0 = self.rbt.cam.initialize_landmark(z[i], wTi)
                if m0 is None:
                    continue

                self.mu[i] = m0
                self.initialized_LM[i] = True
            else:
                indices_update.append(i)
        return indices_update

    def initialize_LM1(self, z):
        mask_observed = np.all(z >= 0, axis=1)
        if mask_observed.sum() == 0:
            return []
        indices_observed = np.where(mask_observed)[0]
        indices_update = []
        c_m, mask = self.rbt.cam1.reconstruct1(z[indices_observed])
        w_m = (self.rbt.wTc @ augment(c_m).T).T[:, :3]
        for j, i in enumerate(indices_observed):
            if not mask[j]:
                continue
            self.mu[i] = w_m[j]
            self.initialized_LM[i] = True
            indices_update.append(i)
        return indices_update

    def stereo_camera_jacobian(self, m):
        w_m = np.hstack([m, 1.0])
        o_m = self.rbt.oTw @ w_m
        if o_m[2] <= 1e-8:
            return None
        J = CPF_Jacobian(o_m)
        H = self.rbt.cam.Ks @ J @ self.rbt.oTw @ self.P.T
        return H

    def stereo_camera_jacobian_batch(self, m):
        H = np.zeros((len(m), 4, 3))
        w_m = augment(m)
        o_m = (self.rbt.oTw @ w_m.T).T
        mask = o_m[:, 2] > 1e-8
        if mask.sum() == 0:
            return None
        J = CPF_Jacobian_batch(o_m[mask])
        H_ = self.rbt.cam.Ks @ J @ self.rbt.oTw @ self.P.T
        H[mask] = H_
        return H, mask

    def update(self, z, indices_valid, initialized_LM, wTi):
        H_blocks = []
        z_hat_blocks = []
        z_obs_blocks = []
        indices_updated = []

        for i in indices_valid:
            i_tmp = 3 * i

            # 1) 初始化 landmark
            if not initialized_LM[i]:
                m0 = self.rbt.cam.initialize_landmark(z[i], wTi)
                if m0 is None:
                    continue

                self.mu[i] = m0
                initialized_LM[i] = True
                self.Sigma[i_tmp:i_tmp + 3, i_tmp:i_tmp + 3] = self.R.copy()
                continue

            # 2) 预测观测
            z_hat_i, H_i = self.rbt.cam.observation_model(self.mu[i], wTi)
            if z_hat_i is None or H_i is None:
                continue

            # 3) 单个观测 gating（推荐）
            innovation_i = z[i] - z_hat_i
            if np.linalg.norm(innovation_i) > 20:
                continue

            # 4) 构造全局 H 的一块
            H_row = np.zeros((4, self.D))
            H_row[:, i_tmp:i_tmp + 3] = H_i

            H_blocks.append(H_row)
            z_hat_blocks.append(z_hat_i)
            z_obs_blocks.append(z[i])
            indices_updated.append(i)

        # 没有可更新的观测
        if len(indices_updated) == 0:
            return

        H = np.vstack(H_blocks)                      # (4K, D)
        z_hat = np.hstack(z_hat_blocks)             # (4K,)
        z_obs = np.hstack(z_obs_blocks)             # (4K,)
        innovation = z_obs - z_hat                  # (4K,)
        K_obs = len(indices_updated)

        V_big = np.kron(np.eye(K_obs), self.V)      # (4K, 4K)

        S = H @ self.Sigma @ H.T + V_big
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        delta = K @ innovation

        for i in indices_updated:
            i_tmp = 3 * i
            self.mu[i] += delta[i_tmp:i_tmp + 3]

        I = np.eye(self.D)
        KH = K @ H
        self.Sigma = (I - KH) @ self.Sigma @ (I - KH).T + K @ V_big @ K.T

        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
