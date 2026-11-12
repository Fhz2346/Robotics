import numpy as np

o_R_r = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)

class StereoCamera:
    def __init__(self, K_l, K_r, l_T_i, r_T_i):
        self.K_l = K_l
        self.K_r = K_r
        self.l_T_i = l_T_i
        self.r_T_i = r_T_i

        self.i_T_l = np.linalg.inv(l_T_i)
        self.i_T_r = np.linalg.inv(r_T_i)

        # 与你当前 reconstruct 一致的 baseline 定义
        r_T_l = r_T_i @ np.linalg.inv(l_T_i)
        r_t_l = o_R_r @ r_T_l[:3, 3]
        self.baseline = r_t_l[0]

        self.fx = K_l[0, 0]
        self.fy = K_l[1, 1]
        self.cx = K_l[0, 2]
        self.cy = K_l[1, 2]

        self.Ks = np.zeros((4, 4))
        self.Ks[:2, :3] = K_l[:2, :3]
        self.Ks[2:, :3] = K_l[:2, :3]
        self.Ks[2, 3] = -self.baseline * self.fx

    def reconstruct(self, z):
        z = np.atleast_2d(z)

        u_l = z[:, 0]
        v_l = z[:, 1]
        u_r = z[:, 2]
        disparity = u_l - u_r

        valid = disparity > 1e-6
        if not np.any(valid):
            return None, valid

        u_l_v = u_l[valid]
        v_l_v = v_l[valid]
        disparity_v = disparity[valid]

        Z_o = self.fx * self.baseline / disparity_v
        X_o = (u_l_v - self.cx) * Z_o / self.fx
        Y_o = (v_l_v - self.cy) * Z_o / self.fy

        depth_mask = (Z_o > 10) & (Z_o < 50)
        valid_idx = np.where(valid)[0]
        valid[valid_idx] = depth_mask

        if not np.any(valid):
            return None, valid

        X_o = X_o[depth_mask]
        Y_o = Y_o[depth_mask]
        Z_o = Z_o[depth_mask]

        p_o = np.stack([X_o, Y_o, Z_o], axis=1)
        c_R_o = o_R_r.T
        p_c = (c_R_o @ p_o.T).T   # left camera frame

        return p_c, valid

    def initialize_landmark(self, z, w_T_i):
        p_c, valid = self.reconstruct(z)
        if p_c is None or p_c.shape[0] == 0:
            return None

        p_l = p_c[0]
        p_l_h = np.hstack([p_l, 1.0])

        # world <- imu <- left_cam
        w_T_l = w_T_i @ self.i_T_l
        p_w = w_T_l @ p_l_h
        return p_w[:3]

    def observation_model(self, m_w, w_T_i):
        i_T_w = np.linalg.inv(w_T_i)
        m_w_h = np.hstack([m_w, 1.0])
        p_i = i_T_w @ m_w_h

        p_l = self.l_T_i @ p_i
        p_c = p_l[:3]

        p_o = o_R_r @ p_c
        X, Y, Z = p_o

        if Z <= 1e-6:
            return None, None

        u_l = self.fx * X / Z + self.cx
        v_l = self.fy * Y / Z + self.cy
        u_r = u_l - self.fx * self.baseline / Z
        v_r = v_l

        z_hat = np.array([u_l, v_l, u_r, v_r])

        H_proj = np.array([
            [self.fx / Z,         0.0, -self.fx * X / (Z * Z)],
            [0.0,         self.fy / Z, -self.fy * Y / (Z * Z)],
            [self.fx / Z,         0.0, -self.fx * X / (Z * Z) + self.fx * self.baseline / (Z * Z)],
            [0.0,         self.fy / Z, -self.fy * Y / (Z * Z)],
        ])

        H_po_pc = o_R_r
        H_pc_mw = (self.l_T_i @ i_T_w)[:3, :3]
        H = H_proj @ H_po_pc @ H_pc_mw
        return z_hat, H
