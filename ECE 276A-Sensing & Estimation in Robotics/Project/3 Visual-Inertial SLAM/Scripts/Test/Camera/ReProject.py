import numpy as np
from Utils.Load import load_data, load_images
from Utils.Visualize import mark_image
from StereoCamera1 import StereoCamera
import numpy as np

V, W, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data()
cam = StereoCamera(K_l, K_r, extL_T_imu, extR_T_imu)
imgs_l, imgs_r = load_images()

t = 1000
z = features[t, :, :]
valid = np.logical_and(z[:, 0] != -1, z[:, 2] != -1)
z = z[valid]

img_l = imgs_l[t]
img_r = imgs_r[t]
p_c = cam.reconstruct(z)
z_ = cam.project(p_c)
mark_image(img_l, z_[:, 0], z_[:, 1], title=f"Reprojected img left, t={t}")
mark_image(img_r, z_[:, 2], z_[:, 3], title=f"Reprojected img right, t={t}")
