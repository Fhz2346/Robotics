import numpy as np

from Utils.Load import *
from Utils.nQuat import exp_pure
from numpy.linalg import norm


def imu_raw2physical():
    bias_W = np.mean(W_raw[:100], axis=0, keepdims=True)

    Vref = 3300.0  # mV
    max_adc = 1023.0
    sens_acc = 330.0  # mV/g
    sens_gyro = 3.33 * 180.0 / np.pi  # mV/rad/s

    scale_A = (Vref / max_adc) / sens_acc
    scale_W = (Vref / max_adc) / sens_gyro

    mean_A = np.mean(A_raw[:100], axis=0, keepdims=True)
    g_ref = np.array([[0.0, 0.0, 1.0]])
    bias_A = mean_A - (g_ref / scale_A)

    A = (A_raw - bias_A) * scale_A  # g
    W = (W_raw - bias_W) * scale_W  # rad/s
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
