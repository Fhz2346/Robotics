import numpy as np
import matplotlib.pyplot as plt
from Utils.Load import load_data
from Utils.pr3_utils import visualize_trajectory_2d
import EKF_odom
from Utils.Utils import select_landmarks
import Utils.Visualize as vis
from Camera.StereoCamera import StereoCamera, o_R_r
from Camera.CPF import CPF_Jacobian
from Geometric.Basics import odot
from Geometric import SE3 as se3
from Config import cfgG

V, W, Ts, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data()
U = np.hstack((V, W))

o_T_r = np.eye(4)
o_T_r[:3, :3] = o_R_r
o_T_i = o_T_r @ extL_T_imu

cam = StereoCamera(K_l, K_r, extL_T_imu, extR_T_imu)

NT = V.shape[0]

selected_indices = select_landmarks(features, min_obs=15, max_landmarks=500)
M = len(selected_indices)

w_T_i = np.zeros((NT, 4, 4))
w_T_i[0] = np.eye(4)

D = 6 + 3 * M
P = np.zeros((D, D))
P[:6, :6] = 1e-6 * np.eye(6)

mu_map = np.full((M, 3), np.nan)
initialized = np.zeros(M, dtype=bool)

sigma_v = 0.03
sigma_w = 0.015
W_odom = np.diag([
    sigma_v**2, sigma_v**2, sigma_v**2,
    sigma_w**2, sigma_w**2, sigma_w**2
])

obs_noise = 2.0
R_obs_single = (obs_noise ** 2) * np.eye(4)

init_var_xy = 2.0
init_var_z = 10.0
V = np.diag(np.array([init_var_xy, init_var_xy, init_var_z]))


def pose_update_left(T, delta_xi):
    return T @ se3.twist2SE3(delta_xi)


def pose_measurement_jacobian_numeric(T, m, eps=1e-6):
    z0, _ = cam.observation_model(m, T)
    if z0 is None:
        return None

    H_pose = np.zeros((4, 6))
    for k in range(6):
        d = np.zeros(6)
        d[k] = eps
        T_perturbed = pose_update_left(T, d)
        z1, _ = cam.observation_model(m, T_perturbed)
        if z1 is None:
            return None
        H_pose[:, k] = (z1 - z0) / eps
    return H_pose


def pose_measurement_jacobian_formula(T, m, Ks):
    m_h = np.hstack([m, 1.0])
    i_T_w = np.linalg.inv(T)
    p_i = i_T_w @ m_h
    q = o_T_i @ p_i
    if q[2] <= 1e-8:
        return None
    J_pi = CPF_Jacobian(q)

    H_pose = -Ks @ J_pi @ o_T_i @ odot(p_i)
    return H_pose


F = np.eye(D)
W = np.zeros((D, D))
for t in range(NT - 1):
    dt = Ts[t + 1] - Ts[t]

    mu_odom_pred, F_pose = EKF_odom.predict1(w_T_i[t], U[t], dt)
    w_T_i[t + 1] = mu_odom_pred
    F[:6, :6] = F_pose
    W[:6, :6] = W_odom * dt

    P = F @ P @ F.T + W

    H_blocks = []
    innovations = []
    R_blocks = []
    valid = np.all(features[t][selected_indices] >= 0, axis=1)
    for i_loc, i_feat in enumerate(selected_indices):
        if not valid[i_loc]:
            continue
        i_tmp = 6 + 3 * i_loc

        z = features[t + 1, i_feat]
        if not initialized[i_loc]:
            m0 = cam.initialize_landmark(z, w_T_i[t + 1])
            if m0 is None:
                continue

            mu_map[i_loc] = m0
            initialized[i_loc] = True

            P[i_tmp : i_tmp + 3, i_tmp : i_tmp + 3] = V.copy()
            continue

        z_hat, H_landmark = cam.observation_model(mu_map[i_loc], w_T_i[t + 1])
        if z_hat is None:
            continue

        # H_pose = pose_measurement_jacobian_numeric(w_T_i[t + 1], mu_map[i_loc])
        H_pose = pose_measurement_jacobian_formula(w_T_i[t + 1], mu_map[i_loc], cam.Ks)
        if H_pose is None:
            continue

        innovation = z - z_hat

        if np.linalg.norm(innovation) > 20:
            continue

        H_row = np.zeros((4, D))
        H_row[:, :6] = H_pose
        H_row[:, i_tmp : i_tmp + 3] = H_landmark

        H_blocks.append(H_row)
        innovations.append(innovation)
        R_blocks.append(R_obs_single)

        if len(H_blocks) >= 25:
            break

    if len(H_blocks) == 0:
        continue

    H = np.vstack(H_blocks)
    y = np.hstack(innovations)
    R = np.zeros((4 * len(R_blocks), 4 * len(R_blocks)))
    for k, Rk in enumerate(R_blocks):
        i0 = 4 * k
        R[i0:i0 + 4, i0:i0 + 4] = Rk

    S = H @ P @ H.T + R
    K_gain = P @ H.T @ np.linalg.inv(S)

    delta = K_gain @ y

    delta_pose = delta[:6]
    w_T_i[t + 1] = pose_update_left(w_T_i[t + 1], delta_pose)

    delta_map = delta[6:]
    if delta_map.size > 0:
        mu_map = mu_map.copy()
        for j in range(M):
            if not initialized[j]:
                continue
            idx0 = 3 * j
            mu_map[j] += delta_map[idx0:idx0 + 3]

    P = (np.eye(D) - K_gain @ H) @ P

landmarks = mu_map[initialized]

visualize_trajectory_2d(w_T_i, show_ori=True)
traj_xy = w_T_i[:, :2, 3]
lm_xy = landmarks[:, :2]
vis.save_map(traj_xy, lm_xy, title="Task 3 SLAM")

np.savez(
    f"{cfgG.out_dir}/task3.npz",
    w_T_i=w_T_i,
    P=P,
    mu_map=mu_map,
    initialized=initialized,
    selected_indices=selected_indices,
)
