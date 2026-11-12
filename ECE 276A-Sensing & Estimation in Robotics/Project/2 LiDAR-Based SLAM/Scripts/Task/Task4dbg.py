import numpy as np
import gtsam
from gtsam import Pose2
from gtsam.symbol_shorthand import X

from src.load_data import load_dataset
from Vis.Visualize import save_traj, save_Theta, save_pc
from Utils.Utils import get_map_idx
from Sensor.Encoder import Encoder
from Sensor.IMU import IMU
from Sensor.Lidar import LiDAR
from Utils.ICP2D import ICP2D, ICP2D_debug, best_init, filter_pc
import Utils.Transformation2D as D2
from Config.Task4 import CFG

ds = load_dataset(21, data_dir=CFG.data_dir)

enc = Encoder.from_dataset(ds.encoders)
imu = IMU.from_dataset(ds.imu)
lidar = LiDAR.from_dataset(ds.lidar)
N = lidar.N

e2i = get_map_idx(imu.T, enc.T)
l2e = get_map_idx(enc.T, lidar.T)


def odom_rel(i0, i1):
    dx = 0.0
    dy = 0.0
    dtheta = 0.0
    theta_tmp = 0.0
    for i in range(i0, i1):
        dt = enc.T[i + 1] - enc.T[i]
        d = enc.estimate_dist(i)
        w = imu.w[e2i[i]]
        dtheta += w * dt
        theta_tmp += w * dt
        dx += d * np.cos(theta_tmp)
        dy += d * np.sin(theta_tmp)
    return D2.make_T(dtheta, [dx, dy])

def T2pose2(T):
    dx = float(T[0, 2])
    dy = float(T[1, 2])
    dtheta = float(np.arctan2(T[1, 0], T[0, 0]))
    return Pose2(dx, dy, dtheta)

def pose22T(pose):
    return D2.make_T(pose.theta(), [pose.x(), pose.y()])

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()
pose0 = Pose2(0.0, 0.0, 0.0)
graph.add(gtsam.PriorFactorPose2(X(0), pose0, CFG.prior_noise))
initial.insert(X(0), pose0)
init_chain = [pose0]

cand0, Tr0 = [], []
cand1, Tr1 = [], []
Nle = 0  # num of loop edges
for k in range(N - 1):
    Tr_odom = odom_rel(l2e[k], l2e[k + 1])
    Tr = Tr_odom
    if (k + 1) % CFG.icp_every == 0:
        P0 = lidar.get_pc(k, stride=CFG.stride)
        P1 = lidar.get_pc(k + 1, stride=CFG.stride)
        Tr_icp, err = ICP2D(P1, P0, init_T=Tr_odom, max_iters=25, tol_mse=1e-5, trim_ratio=0.8)
        Tr = Tr_icp

    pose_r = T2pose2(Tr)
    graph.add(gtsam.BetweenFactorPose2(X(k), X(k + 1), pose_r, CFG.odom_noise))
    prev = init_chain[-1]
    curr = prev.compose(pose_r)
    init_chain.append(curr)
    initial.insert(X(k + 1), curr)

    if (k + 1) % CFG.loop_every == 0 and (Nle < CFG.max_loop_edges):
        cur_xy = np.array([curr.x(), curr.y()])
        best_j = None
        best_d = np.inf
        for j in range(0, max(0, k - CFG.min_sep), 10):  # step 10 to keep it cheap
            pose = init_chain[j]
            d = np.linalg.norm(cur_xy - np.array([pose.x(), pose.y()]))
            if d < best_d:
                best_d = d
                best_j = j
        if (best_j is not None) and (best_d < CFG.dist_gate):
            pose_r = init_chain[best_j].between(curr)
            Tr_ = pose22T(pose_r)
            cand0.append([best_j, k + 1])
            Tr0.append(Tr_)
            P0 = lidar.get_pc(best_j, stride=CFG.stride)
            P1 = lidar.get_pc(k + 1, stride=CFG.stride)
            Tr, err = ICP2D(P1, P0, init_T=Tr_, max_iters=30, tol_mse=1e-5, trim_ratio=0.8)
            if err > CFG.th_err:
                continue
            pose_r = T2pose2(Tr)
            graph.add(gtsam.BetweenFactorPose2(X(best_j), X(k + 1), pose_r, CFG.loop_noise))
            Nle += 1
            print(f"[Loop] added edge: {best_j} -> {k+1}, init_dist={best_d:.2f}m, total={Nle}")
            cand1.append([best_j, k + 1])
            Tr1.append(Tr)


np.savez("Loop.npz", cand0=cand0, cand1=cand1, Tr0=Tr0, Tr1=Tr1)
# save_pc(D2.apply_T(P0, Tr), P1, title=f"{best_j} vs {k}", dir="Output/Debug/Loop")
