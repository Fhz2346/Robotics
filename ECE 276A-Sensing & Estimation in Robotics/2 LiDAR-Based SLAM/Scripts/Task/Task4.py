import numpy as np
import gtsam
from gtsam import Pose2
from gtsam.symbol_shorthand import X

import Vis.Plot as Vis
from Robot import Robot, SensorType
from Utils.ICP2D import ICP2D, ICP2D_debug, best_init
import Utils.Transformation2D as D2
from Config import cfg4, cfgG
from Utils.gtsam import T2pose2, pose22T


rbt = Robot()
rbt.activate_all([SensorType.ENCODER, SensorType.IMU, SensorType.LIDAR])
enc = rbt.enc
imu = rbt.imu
lidar = rbt.lid
rbt.build_Tmap()
e2i = rbt.Tmap[("imu", "enc")]
l2e = rbt.Tmap[("enc", "lid")]

N = lidar.N

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()
pose0 = Pose2(0.0, 0.0, 0.0)
graph.add(gtsam.PriorFactorPose2(X(0), pose0, cfg4.prior_noise))
initial.insert(X(0), pose0)
init_chain = [pose0]

Nle = 0  # num of loop edges
for k in range(N - 1):
    Tr_odom = rbt.odom_rel(l2e[k], l2e[k + 1])
    Tr = Tr_odom
    if (k + 1) % cfg4.icp_every == 0:
        P0 = lidar.get_pc(k, stride=cfg4.stride)
        P1 = lidar.get_pc(k + 1, stride=cfg4.stride)
        Tr_icp, err = ICP2D(P1, P0, init_T=Tr_odom, max_iters=25, tol_mse=1e-5, trim_ratio=0.8)
        Tr = Tr_icp

    pose_r = T2pose2(Tr)
    graph.add(gtsam.BetweenFactorPose2(X(k), X(k + 1), pose_r, cfg4.odom_noise))
    prev = init_chain[-1]
    curr = prev.compose(pose_r)
    init_chain.append(curr)
    initial.insert(X(k + 1), curr)

    if (k + 1) % cfg4.loop_every == 0 and (Nle < cfg4.max_loop_edges):
        cur_xy = np.array([curr.x(), curr.y()])
        best_j = None
        best_d = np.inf
        for j in range(0, max(0, k - cfg4.min_sep), 10):  # step 10 to keep it cheap
            pose = init_chain[j]
            d = np.linalg.norm(cur_xy - np.array([pose.x(), pose.y()]))
            if d < best_d:
                best_d = d
                best_j = j
        if (best_j is not None) and (best_d < cfg4.dist_gate):
            pose_r = init_chain[best_j].between(curr)
            Tr_ = pose22T(pose_r)
            P0 = lidar.get_pc(best_j, stride=cfg4.stride)
            P1 = lidar.get_pc(k + 1, stride=cfg4.stride)
            Tr, err = ICP2D(P1, P0, init_T=Tr_, max_iters=30, tol_mse=1e-5, trim_ratio=0.8)
            if err > cfg4.th_err:
                continue
            pose_r = T2pose2(Tr)
            graph.add(gtsam.BetweenFactorPose2(X(best_j), X(k + 1), pose_r, cfg4.loop_noise))
            Nle += 1
            print(f"[Loop] added edge: {best_j} -> {k+1}, init_dist={best_d:.2f}m, total={Nle}")

# Optimize
params = gtsam.LevenbergMarquardtParams()
params.setVerbosity("ERROR")
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()

traj_opt = np.zeros((N, 2), dtype=np.float64)
theta_opt = np.zeros(N, dtype=np.float64)
for k in range(N):
    p = result.atPose2(X(k))
    traj_opt[k, 0] = p.x()
    traj_opt[k, 1] = p.y()
    theta_opt[k] = p.theta()

theta_opt = np.unwrap(theta_opt)
np.savez(f"{cfgG.out_dir}/posegraph_simple.npz", traj=traj_opt, Theta=theta_opt)
Vis.save_traj(traj_opt)
Vis.save_Theta(theta_opt, lidar.T)
print("Saved: posegraph_simple.npz")
