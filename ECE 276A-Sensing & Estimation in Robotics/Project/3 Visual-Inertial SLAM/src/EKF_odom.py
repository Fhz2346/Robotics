import numpy as np
import Geometric.SE3 as se3


def predict(mu_prev, Sigma_prev, v, w, dt, Q):
    zeta = np.hstack((v, w))
    dT = se3.twist2SE3(zeta * dt)
    mu_pred = mu_prev @ dT

    F = se3.Ad(dT.T)
    Sigma_pred = F @ Sigma_prev @ F.T + Q * (dt**2)
    return mu_pred, Sigma_pred


def predict1(mu_prev, u, dt):
    dT = se3.twist2SE3(u * dt)
    mu_pred = mu_prev @ dT
    F = se3.Ad(dT.T)
    return mu_pred, F


def get_F(v, w, dt):
    zeta = np.hstack((v, w))
    dT = se3.twist2SE3(zeta * dt)
    F = se3.Ad(dT.T)
    return F
