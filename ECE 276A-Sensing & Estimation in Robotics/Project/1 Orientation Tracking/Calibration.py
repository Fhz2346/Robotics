import numpy as np
from numpy.linalg import norm

from transforms3d.quaternions import qmult
from transforms3d.euler import quat2euler

from Utils.Load import *
from Vis.Visualize import *
from Vis.Animation import animate_two_rot_paths
from Utils.nQuat import exp_pure


def stationary_mask(thresh=0.08):
    # plot_RPY_v(ts_v, RPY_v)
    dist = np.linalg.norm(RPY_v, axis=1)
    mask_v = dist  < thresh
    # plot_mask_v(ts_v, mask_v, dist=dist)
    idx_fall = np.where(mask_v[:-1] & (~mask_v[1:]))[0][0]
    mask_v[idx_fall:] = False
    plot_mask_v(ts_v, mask_v, dist=dist)
    mask_i = mask_v[idx_v2i]
    return mask_i


def imu_raw2physical():
    mask_stat = stationary_mask()
    bias_W = np.mean(W_raw[mask_stat], axis=0, keepdims=True)

    Vref = 3300.0  # mV
    max_adc = 1023.0
    sens_acc = 330.0  # mV/g
    sens_gyro = 3.33 * 180.0 / np.pi  # mV/rad/s

    scale_A = (Vref / max_adc) / sens_acc
    scale_W = (Vref / max_adc) / sens_gyro

    mean_A = np.mean(A_raw[mask_stat], axis=0, keepdims=True)
    g_ref = np.array([[0.0, 0.0, 1.0]])
    bias_A = mean_A - (g_ref / scale_A)

    A = (A_raw - bias_A) * scale_A  # g
    W = (W_raw - bias_W) * scale_W  # rad/s
    
    # A_norm = np.linalg.norm(A, axis=1)
    # plot_A_norm(ts_i, A_norm)
    
    A_v_i = A_v[idx_v2i]
    plot_compare_A(ts_i, A, A_v_i)
    
    # plot_A(ts_i, A, unit="g")
    # plot_W(ts_i, W, unit="rad/s")
    return A, W


def integrate_gyro_to_quat(W, q0=np.array([1.0, 0.0, 0.0, 0.0])):
    N = ts_i.shape[0]
    q = q0
    Q = np.zeros((N, 4), dtype=np.float64)
    Q[0] = q
    for k in range(N - 1):
        dt = ts_i[k + 1] - ts_i[k]
        dq = exp_pure(dt * W[k] / 2)
        q = qmult(q, dq)
        q = q / norm(q)
        Q[k + 1] = q
    return Q


if __name__ == "__main__":
    A, W = imu_raw2physical()
    Q_est = integrate_gyro_to_quat(W, q0=np.array([1.0, 0.0, 0.0, 0.0]))
    N = Q_est.shape[0]
    RPY_est = [quat2euler(Q_est[k], axes="sxyz") for k in range(N)]
    RPY_est = np.array(RPY_est)
    RPY_gt_aligned = RPY_v[idx_v2i]
    plot_rpy(ts_i, RPY_gt_aligned, RPY_est, title_prefix=f"Trainset{trainset} - ")
    # animate_two_rot_paths(ts_i, RPY_gt_aligned, RPY_est)
