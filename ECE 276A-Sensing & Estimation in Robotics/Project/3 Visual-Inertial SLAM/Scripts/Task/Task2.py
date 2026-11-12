from MappingEKF import run_mapping_ekf
import numpy as np
from Utils.Load import load_data
from Utils.Utils import select_landmarks
from Camera.StereoCamera import StereoCamera
from Utils.pr3_utils import visualize_trajectory_2d
import Utils.Visualize as vis
from Config import cfgG


V, W, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data()
cam = StereoCamera(K_l, K_r, extL_T_imu, extR_T_imu)
wTi = np.load(f"{cfgG.out_dir}/task1.npz", allow_pickle=True)["wTi"]
selected_idx = select_landmarks(features, min_obs=15, max_landmarks=-1)

mu_map, Sigma_map, initialized = run_mapping_ekf(
    features=features,
    w_T_i=wTi,
    cam=cam,
    selected_idx=selected_idx,
    obs_noise=2.0,
    init_var_xy=5.0,
    init_var_z=20.0,
)

landmarks = mu_map[initialized]

print("selected landmarks:", len(selected_idx))
print("initialized landmarks:", np.sum(initialized))
print("final landmark array shape:", landmarks.shape)

import matplotlib.pyplot as plt

traj_xy = wTi[:, :2, 3]
lm_xy = landmarks[:, :2]
margin = 10

xmin, ymin = traj_xy.min(axis=0) - margin
xmax, ymax = traj_xy.max(axis=0) + margin

mask = (
    (lm_xy[:,0] > xmin) &
    (lm_xy[:,0] < xmax) &
    (lm_xy[:,1] > ymin) &
    (lm_xy[:,1] < ymax)
)

lm_xy_plot = lm_xy[mask]

vis.save_map(traj_xy, lm_xy_plot, title="Task2")
