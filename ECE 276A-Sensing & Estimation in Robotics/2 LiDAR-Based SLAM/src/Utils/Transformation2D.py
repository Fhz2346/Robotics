import numpy as np

def get_R(Theta):
    c = np.cos(Theta)
    s = np.sin(Theta)
    Rs = np.array([[[c[k], -s[k]], [s[k], c[k]]] for k in range(len(Theta))])
    return Rs

def get_R_(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R

def make_T(theta, t):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(3, dtype=np.float64)
    T[0, 0] = c;  T[0, 1] = -s
    T[1, 0] = s;  T[1, 1] = c
    T[0, 2] = float(t[0])
    T[1, 2] = float(t[1])
    return T


def apply_T(P, T):
    P = np.asarray(P, dtype=np.float64)
    Ph = np.hstack([P[:, :2], np.ones((P.shape[0], 1), dtype=np.float64)])
    Q = (T @ Ph.T).T
    return Q[:, :2]


def fit_rigid(A, B):
    A = np.asarray(A, dtype=np.float64)[:, :2]
    B = np.asarray(B, dtype=np.float64)[:, :2]
    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    AA = A - ca
    BB = B - cb
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t
