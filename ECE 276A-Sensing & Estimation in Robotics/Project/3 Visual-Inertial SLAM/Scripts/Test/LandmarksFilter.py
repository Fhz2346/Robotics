from Utils.Load import load_data
import numpy as np
from Config import cfgLF
import Utils.Visualize as Vis

V, W, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data()

valid = np.logical_and(features[:, :, 0] != -1, features[:, :, 2] != -1)
counts = np.sum(valid, axis=0)
# Vis.plot_landmark_counts(counts)
idices = np.where(counts >= cfgLF.min_obs)[0]
print(len(idices), "landmarks with at least", cfgLF.min_obs, "observations")
idices = idices[np.argsort(counts[idices])[::-1]]
best_idices = idices[: cfgLF.max_landmarks]
