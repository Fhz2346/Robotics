import numpy as np
import Vis.Plot as Vis
from Robot import Robot, SensorType
from Utils.ICP2D import ICP2D
from Config import cfgG

activation = [SensorType.ENCODER, SensorType.IMU, SensorType.LIDAR]
rbt = Robot()
rbt.activate_all(activation)
lidar = rbt.lid
rbt.build_Tmap()
l2e = rbt.Tmap[("enc", "lid")]

T_world = np.eye(3, dtype=np.float64)
traj = np.zeros((lidar.N, 2), dtype=np.float64)
Theta = np.zeros(lidar.N, dtype=np.float64)
for k in range(lidar.N - 1):
    Tr_odom = rbt.odom_rel(l2e[k], l2e[k + 1])
    Tr = Tr_odom # Tr : relative transformation from scan k to k+1
    if (k + 1) % 10 == 0:
        P0 = lidar.get_pc(k, stride=4)
        P1 = lidar.get_pc(k + 1, stride=4)
        Tr_icp, err = ICP2D(P1, P0, init_T=Tr_odom, max_iters=25, tol_mse=1e-5, trim_ratio=0.8)
        Tr = Tr_icp
    T_world = T_world @ Tr
    traj[k + 1] = T_world[:2, 2]
    Theta[k + 1] = np.arctan2(T_world[1, 0], T_world[0, 0])
    
np.savez(f"{cfgG.out_dir}/scan_matching.npz", traj=traj, Theta=Theta)
Vis.save_traj(traj)
Theta = np.unwrap(Theta)
Vis.save_Theta(Theta, lidar.T)
