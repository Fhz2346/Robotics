import numpy as np
from Sensor.Kinect import Kinect
from src.load_data import load_dataset
from Utils.Utils import get_R
from Utils.Map import OccupancyGridMap
from Sensor.Lidar import LiDAR
import scipy.spatial.transform as D3
from Vis.Visualize import show_pc3D


# ===== load everything =====
ds = load_dataset(20, data_dir="../data")
kin = Kinect.from_dataset(ds.kinect)
kin.id = 20

data = np.load("Output/Task2/10_scan_matching.npz")
traj = data["traj"]
Theta = data["Theta"]
Rs = get_R(Theta)

k = 0
c_pts, colors = kin.reconstruct(k, k)

# ===== camera(depth) -> robot center extrinsic (given) =====
R_base = np.array(
    [
        [0, 0, 1],  # x_r =  z_c
        [-1, 0, 0],  # y_r = -x_c
        [0, -1, 0],  # z_r = -y_c
    ],
    dtype=np.float64,
)
r_t_c = np.array([0.18, 0.005, 0.36], dtype=np.float64)  # meters
roll, pitch, yaw = 0.0, 0.36, 0.021
R_rpy = D3.Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
r_R_c = R_rpy @ R_base
w_R_r = D3.Rotation.from_euler("z", Theta[k]).as_matrix()

r_pts = c_pts @ r_R_c.T + r_t_c
w_pts = r_pts @ w_R_r.T
w_pts[:, :2] += traj[k]

show_pc3D(r_pts, colors, "Robot Point Cloud")
show_pc3D(w_pts, colors, "World Point Cloud")
