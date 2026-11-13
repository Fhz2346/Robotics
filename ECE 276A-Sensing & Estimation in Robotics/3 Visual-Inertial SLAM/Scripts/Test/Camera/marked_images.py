import numpy as np
import matplotlib.pyplot as plt
from Utils.Load import load_data, load_images
from Utils.Visualize import mark_image
import numpy as np
from Config import cfgG

V, W, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data()
imgs_l, imgs_r = load_images()

t = 1000
z = features[t, :, :]
valid = np.logical_and(z[:, 0] != -1, z[:, 2] != -1)
z = z[valid]

img_l = imgs_l[t]
img_r = imgs_r[t]

# 左右图像上的像素坐标
u_l, v_l = z[:, 0], z[:, 1]
u_r, v_r = z[:, 2], z[:, 3]


mark_image(img_l, u_l, v_l, title=f"Marked img left, t={t}")
mark_image(img_r, u_r, v_r, title=f"Marked img right, t={t}")
