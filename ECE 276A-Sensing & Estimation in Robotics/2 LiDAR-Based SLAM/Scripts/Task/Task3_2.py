import numpy as np

import scipy.spatial.transform as D3
import Vis.Plot as Vis
from Utils.Map import ColoredGridMap
from Config import cfgG, cfg3_2
from Robot import Robot, SensorType

data = np.load(f"{cfgG.out_dir}/scan_matching.npz")
traj = data["traj"]
Theta = data["Theta"]
Cmap = ColoredGridMap()

activation = [SensorType.LIDAR, SensorType.RGBD]
rbt = Robot()
rbt.activate_all(activation)
lidar = rbt.lid
kin = rbt.kin
r_R_c = rbt.r_R_c
t_R_c = rbt.r_t_c
rbt.build_Tmap()
l2d = rbt.Tmap[("disp", "lid")]
l2rgb = rbt.Tmap[("rgb", "lid")]

for k in range(0, lidar.N, cfg3_2.step_lidar):
    c_pts, colors = kin.reconstruct(l2d[k], l2rgb[k])
    w_R_r = D3.Rotation.from_euler("z", Theta[k]).as_matrix()
    r_pts = c_pts @ r_R_c.T + t_R_c
    w_pts = r_pts @ w_R_r.T
    w_pts[:, :2] += traj[k]

    # mask_floor = np.abs(w_pts[:, 2]) < cfg3_2.th_floor
    # w_pts = w_pts[mask_floor]
    # colors = colors[mask_floor]
    Cmap.update(w_pts[:, :2], colors)

Vis.save_Cmap(Cmap.map)
