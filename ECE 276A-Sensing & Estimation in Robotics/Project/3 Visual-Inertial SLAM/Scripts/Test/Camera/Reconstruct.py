from StereoCamera1 import StereoCamera
from Utils.Load import load_data, load_images
import numpy as np
from Utils.Visualize import plot_pts
from Config import cfgG

V, W, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data()
r_T_l = extR_T_imu @ np.linalg.inv(extL_T_imu)
cam = StereoCamera(K_l, K_r, extL_T_imu, extR_T_imu)

t = 1000
z = features[t, :, :]
valid = np.logical_and(z[:, 0] != -1, z[:, 2] != -1)
z = z[valid]

p_c = cam.reconstruct(z)
plot_pts(p_c, title="Reconstructed points in camera frame")
z_ = cam.observation_model(p_c)
