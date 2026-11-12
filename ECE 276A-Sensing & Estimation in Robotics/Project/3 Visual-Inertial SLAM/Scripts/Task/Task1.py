import numpy as np
from Utils.Load import load_data
from Utils.pr3_utils import visualize_trajectory_2d
import EKF_odom as EKF_odom
from Config import cfgG


V, W, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data()

T_steps = V.shape[0]
wTi = np.zeros((T_steps, 4, 4))
Sigma_all = np.zeros((T_steps, 6, 6))

wTi[0] = np.eye(4)
Sigma_all[0] = 1e-6 * np.eye(6)

sigma_v = 0.05
sigma_w = 0.02
Q = np.diag([sigma_v**2, sigma_v**2, sigma_v**2, sigma_w**2, sigma_w**2, sigma_w**2])

for t in range(T_steps - 1):
    dt = timestamps[t + 1] - timestamps[t]
    wTi[t + 1], Sigma_all[t + 1] = EKF_odom.predict(wTi[t], Sigma_all[t], V[t], W[t], dt, Q)

visualize_trajectory_2d(wTi, show_ori=True)
np.savez(f"{cfgG.out_dir}/task1.npz", wTi=wTi, Sigma_all=Sigma_all)
