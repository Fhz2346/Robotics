import numpy as np
import os
from src.load_data import load_dataset
from Vis.Visualize import save_traj, save_Theta, save_pc
from Utils.Utils import get_map_idx
from Sensor.Encoder import Encoder
from Sensor.IMU import IMU
from Sensor.Lidar import LiDAR
from Utils.ICP2D import ICP2D, ICP2D_debug, best_init, filter_pc
import Utils.Transformation2D as D2
from Config.Task4 import CFG


def clear(folder_path="Output/Debug/Loop"):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


ds = load_dataset(21, data_dir=CFG.data_dir)
lidar = LiDAR.from_dataset(ds.lidar)
data = np.load(f"Loop.npz")
clear("Output/Debug/Loop0")
clear("Output/Debug/Loop1")


def show_loop(J, K, Ts, dir="Output/Debug/Loop"):
    for i, (j, k) in enumerate(zip(J, K)):
        P0 = lidar.get_pc(j, stride=1)
        P1 = lidar.get_pc(k, stride=1)
        Tr = Ts[i]
        save_pc(P0, D2.apply_T(P1, Tr), title=f"{j} vs {k}", dir=dir)


# J = data["cand0"][:, 0]
# K = data["cand0"][:, 1]
# Ts = data["Tr0"]
# show_loop(J, K, Ts, dir="Output/Debug/Loop0")

J = data["cand1"][:, 0]
K = data["cand1"][:, 1]
Ts = data["Tr1"]
show_loop(J, K, Ts, dir="Output/Debug/Loop1")
