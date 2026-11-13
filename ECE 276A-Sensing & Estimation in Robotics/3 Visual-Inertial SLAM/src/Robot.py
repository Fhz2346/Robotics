import numpy as np
from Camera.StereoCamera import StereoCamera, o_R_r
from Camera.StereoCamera1 import StereoCamera1

o_T_r = np.eye(4)
o_T_r[:3, :3] = o_R_r
class Robot:

    def __init__(self, K_l, K_r, L_T_i, R_T_i):
        self.cam = StereoCamera(K_l, K_r, L_T_i, R_T_i)
        self.cam1 = StereoCamera1(K_l, K_r, L_T_i, R_T_i)
        self.L_T_i = L_T_i
        self.R_T_i = R_T_i
        self.o_T_i = o_T_r @ self.L_T_i

    def update(self, wTi):
        self.wTi = wTi
        self.iTw = np.linalg.inv(wTi)
        self.oTw = self.o_T_i @ self.iTw
        self.cTw = self.L_T_i @ self.iTw
        self.wTc = np.linalg.inv(self.cTw)
