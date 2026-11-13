import numpy as np
from Utils.ICP2D import ICP2D, make_T
from Sensor.Lidar import LiDAR
from src.load_data import load_dataset
from Vis.Visualize import save_pc


data = np.load("odom.npz")
traj = data["traj"]
Theta = data["Theta"]

ds = load_dataset(20, data_dir="../data")
lidar = LiDAR.from_dataset(ds.lidar)

N = traj.shape[0]
k0 = 1000
k1 = k0 + 100

for stride in [1, 2, 4, 8]:
    P0 = lidar.get_pc(k0, stride)
    P1 = lidar.get_pc(k1, stride)
    save_pc(P0, P1, title=f"scan Before ICP with stride {stride}")
