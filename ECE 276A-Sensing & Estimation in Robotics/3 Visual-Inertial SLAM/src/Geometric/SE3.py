import numpy as np
import Geometric.SO3 as so3


def hat(xi):
    rho = xi[:3]
    theta = xi[3:]
    xi_hat = np.zeros((4, 4))
    xi_hat[:3, :3] = so3.hat(theta)
    xi_hat[:3, 3] = rho
    return xi_hat


def vee(xi_hat):
    theta_hat = xi_hat[:3, :3]
    rho = xi_hat[:3, 3]
    theta = so3.vee(theta_hat)
    xi = np.hstack((rho, theta))
    return xi


def exp(xi_hat):
    theta_hat = xi_hat[:3, :3]
    theta = so3.vee(theta_hat)
    rho = xi_hat[:3, 3]
    R = so3.exp(theta_hat)
    J = so3.left_jacobian(theta)
    t = J @ rho
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def twist2SE3(xi):
    return exp(hat(xi))


def Ad(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = so3.hat(p) @ R
    Ad[3:, 3:] = R
    return Ad


def ad(xi_hat):
    theta_hat = xi_hat[:3, :3]
    rho = xi_hat[:3, 3]
    ad = np.zeros((6, 6))
    ad[:3, :3] = theta_hat
    ad[:3, 3:] = so3.hat(rho)
    ad[3:, 3:] = theta_hat
    return ad
