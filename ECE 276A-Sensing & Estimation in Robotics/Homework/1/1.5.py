import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def skew(v):
    vx, vy, vz = v
    return np.array([
        [0,   -vz,  vy],
        [vz,   0,  -vx],
        [-vy, vx,   0 ]
    ], dtype=v.dtype)

def rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)

    if theta < 1e-12:
        return np.eye(3)

    k = rotvec / theta
    K = skew(k)

    R = (
        np.eye(3)
        + np.sin(theta) * K
        + (1 - np.cos(theta)) * (K @ K)
    )

    return R

pB_A = np.array([1, 1, 0], dtype=float)
pC_B = np.array([0, 1, 2], dtype=float)


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

RB_A = R.from_euler("xyz", [30.0, 60.0, 45.0], degrees=True).as_matrix()
rads = np.radians([30.0, 60.0, 45.0])
RB_A = Rz(rads[2]) @ Ry(rads[1]) @ Rx(rads[0])

axis = (1 / np.sqrt(2)) * np.array([1, 1, 0], dtype=float)
angle = 30.0  # degrees
rad = angle * np.pi / 180.0
rotvec = axis * rad
RC_B = rodrigues(rotvec)

x_C = np.array([1, 1, 1], dtype=float)
x_B = pC_B + RC_B @ x_C
x_A = pB_A + RB_A @ x_B

print("RB_A =", RB_A)
print("RC_B =", RC_B)
print("x_B =", x_B)
print("x_A =", x_A)
