import numpy as np
from Geometric.Basics import augment
from Camera.CPF import CPF

o_R_c = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)


class StereoCamera1:

    def __init__(self, K_l, K_r, l_T_i, r_T_i):
        r_T_l = r_T_i @ np.linalg.inv(l_T_i)
        r_t_l = o_R_c @ r_T_l[:3, 3]
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
        u_l = z[:, 0]
        v_l = z[:, 1]
        u_r = z[:, 2]
        disparity = u_l - u_r

        mask_disparity = disparity > 1e-6
        u_l = u_l[mask_disparity]
        v_l = v_l[mask_disparity]
        disparity = disparity[mask_disparity]

        Z_o = self.fx * self.baseline / disparity
        X_o = (u_l - self.cx) * Z_o / self.fx
        Y_o = (v_l - self.cy) * Z_o / self.fy

        mask_depth = (Z_o > 10) & (Z_o < 50)

        X_o = X_o[mask_depth]
        Y_o = Y_o[mask_depth]
        Z_o = Z_o[mask_depth]

        p_o = np.stack([X_o, Y_o, Z_o], axis=1)

        c_R_o = o_R_c.T
        p_c = (c_R_o @ p_o.T).T  # p_l
        return p_c

    def reconstruct1(self, z):
        u_l = z[:, 0]
        v_l = z[:, 1]
        u_r = z[:, 2]
        disparity = u_l - u_r

        mask_disparity = disparity > 1e-6

        Z_o = self.fx * self.baseline / disparity
        X_o = (u_l - self.cx) * Z_o / self.fx
        Y_o = (v_l - self.cy) * Z_o / self.fy

        mask_depth = (Z_o > 10) & (Z_o < 50)

        p_o = np.stack([X_o, Y_o, Z_o], axis=1)
        mask = mask_disparity & mask_depth
        p_o[mask == False] = 0.0

        c_R_o = o_R_c.T
        p_c = (c_R_o @ p_o.T).T  # p_l
        return p_c, mask

    def project(self, p_c):
        p_o = (o_R_c @ p_c.T).T
        p_o_aug = augment(p_o)
        z = (self.Ks @ CPF(p_o_aug).T).T
        return z
