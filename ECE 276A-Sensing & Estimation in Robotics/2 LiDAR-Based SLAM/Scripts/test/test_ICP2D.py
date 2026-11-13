import numpy as np
from Utils.ICP2D import ICP2D
import Utils.Transformation2D as D2
from Robot import Robot, SensorType
from load_data import load_dataset
import Vis.Plot as Vis
from Config import cfgG


data = np.load(f"{cfgG.out_dir}/odom.npz")
traj = data["traj"]
Theta = data["Theta"]

rbt = Robot()
rbt.activate_all([SensorType.LIDAR])
lidar = rbt.lid

N = traj.shape[0]
k0 = 1000
k1 = k0 + 100

T0 = D2.make_T(Theta[k0], traj[k0])
T1 = D2.make_T(Theta[k1], traj[k1])
T_odom = np.linalg.inv(T1) @ T0

stride = 1
P0 = lidar.get_pc(k0, stride)
P1 = lidar.get_pc(k1, stride)
Vis.save_pc(P0, P1, title=f"scan Before ICP with stride {stride}", dir=f"{cfgG.out_dir}/Debug")
# P0_ = apply_T(P0, T0)
# P1_ = apply_T(P1, T1)
# save_pc(P0_, P1_, title=f"scan reinit")
Vis.save_pc(P0, P1, title=f"scan Before ICP with stride {stride}", dir=f"{cfgG.out_dir}/Debug")
T_icp, err = ICP2D(P0, P1, init_T=T_odom, max_iters=20, tol_mse=1e-2, trim_ratio=0.8)
