import numpy as np
from Utils.Load import load_data
from Robot import Robot
from EKF.Odometry import OdometryEKF
from EKF.Mapping import MappingEKF
from Utils.Utils import select_landmarks
from Geometric.Basics import augment
from Geometric import SE3 as se3
import Utils.Visualize as vis

V, W, Ts, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data()

U = np.hstack((V, W))
rbt = Robot(K_l, K_r, extL_T_imu, extR_T_imu)
oEKF = OdometryEKF(rbt, sigma_v=0.03, sigma_w=0.015)

selected_indices = select_landmarks(features, min_obs=15, max_landmarks=120)
features = features[:, selected_indices]
M = len(selected_indices)
mEKF = MappingEKF(rbt, sigma_obs=2.0, init_var_xy=2.0, init_var_z=10.0, M=M)

NT = V.shape[0]
w_T_i = np.zeros((NT, 4, 4))
w_T_i[0] = np.eye(4)

D = 6 + 3 * M
P = np.zeros((D, D))
P[:6, :6] = 1e-6 * np.eye(6)

F = np.eye(D)
W = np.zeros((D, D))
for t in range(NT - 1):
    dt = Ts[t + 1] - Ts[t]
    # Predict
    oEKF.predict1(U[t], dt)
    rbt.update(oEKF.mu)
    F[:6, :6] = oEKF.F
    W[:6, :6] = oEKF.W * dt
    P = F @ P @ F.T + W
    # Update
    z = features[t + 1]
    indices_update = mEKF.initialize_LM1(z)
    if len(indices_update) == 0:
        continue
    w_m = mEKF.mu[indices_update]
    c_m = rbt.cTw @ augment(w_m).T
    c_m = c_m.T[:, :3]
    z_hat = rbt.cam1.project(c_m)
    innovation = z[indices_update] - z_hat
    mask_innovation = np.all(np.abs(innovation) < 10.0, axis=1)
    if mask_innovation.sum() == 0:
        continue
    Hm, mask_Hm = mEKF.stereo_camera_jacobian_batch(w_m)
    Ho, mask_Ho = oEKF.stereo_camera_jacobian_batch(w_m)
    mask = mask_innovation & mask_Hm & mask_Ho
    if mask.sum() == 0:
        continue
    
    # keep only valid measurements/Jacobians
    idx_all = np.array(indices_update)
    idx_used = idx_all[mask]

    z_used = z[idx_used]                      # (N, 4)
    z_hat_used = z_hat[mask]                 # (N, 4)
    innovation = (z_used - z_hat_used).reshape(-1)   # (4N,)

    Hm = Hm[mask]    # expected shape: (N, 4, 3) or already (4N, 3N_local)
    Ho = Ho[mask]    # expected shape: (N, 4, 6)

    N = len(idx_used)
    if N == 0:
        continue

    # Build global H
    H = np.zeros((4 * N, D))

    # Pose block
    # Ho is assumed to be (N, 4, 6)
    for k in range(N):
        H[4 * k : 4 * (k + 1), :6] = Ho[k]

    # Landmark block
    # Hm is assumed to be (N, 4, 3)
    for k, j in enumerate(idx_used):
        col = 6 + 3 * j
        H[4 * k : 4 * (k + 1), col : col + 3] = Hm[k]

    # Measurement noise
    R = (mEKF.sigma_obs ** 2) * np.eye(4 * N)

    # EKF update
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    delta = K @ innovation

    # State update
    delta_pose = delta[:6]
    delta_map = delta[6:]

    # Update pose on manifold
    oEKF.mu = oEKF.mu @ se3.twist2SE3(delta_pose)
    rbt.update(oEKF.mu)

    # Update landmarks in Euclidean space
    mEKF.mu += delta_map.reshape(M, 3)

    # Covariance update (Joseph form)
    I = np.eye(D)
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

    # Save pose
    w_T_i[t + 1] = oEKF.mu

landmarks = mEKF.get_landmarks()
traj_xy = w_T_i[:, :2, 3]
lm_xy = landmarks[:, :2]
vis.save_map(traj_xy, lm_xy, title="Task 3 SLAM")