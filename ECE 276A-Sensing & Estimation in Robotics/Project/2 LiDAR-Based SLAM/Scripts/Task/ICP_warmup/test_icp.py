import numpy as np
from utils import *
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R

def make_T(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def apply_T(P, T):
    P_h = np.hstack([P, np.ones((P.shape[0], 1))])
    P_trans = (T @ P_h.T).T
    return P_trans[:, :3]


def fit_rigid(A, B):
    """fit rigid transformation from A to B using SVD"""
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def NNC(src, dst):
    """Nearest Neighbor Correspondences"""
    tree = KDTree(dst)
    dist, idx = tree.query(src, k=1, workers=-1)
    return idx, dist


def ICP(source, target, initT, max_iters=50, delta=1e-6, trim_ratio=0.8):
    assert trim_ratio > 0 and trim_ratio <= 1.0
    T = initT.copy()
    histMSE = []
    for it in range(max_iters):
        src = apply_T(source, T)
        idx, dist = NNC(src, target)
        dst = target[idx]
        # Trim outliers by distance
        k = int(trim_ratio * dist.shape[0])
        keep = np.argpartition(dist, k - 1)[:k]
        src = src[keep]
        dst = dst[keep]
        dist = dist[keep]
        # Fit rigid transform (incremental)
        mse = float(np.mean(dist**2))
        histMSE.append(mse)
        R_inc, t_inc = fit_rigid(src, dst)
        T = make_T(R_inc, t_inc) @ T
        if it > 0 and mse < delta:
            break
    return T


def best_init(source, target, cand_yaw):
    t0 = target.mean(axis=0) - source.mean(axis=0)
    bestD = np.inf
    bestT = np.eye(4, dtype=np.float64)
    for yaw in cand_yaw:
        Rz = R.from_euler('z', yaw).as_matrix()
        T = make_T(Rz, t0)
        src = apply_T(source, T)
        _, dist = NNC(src, target)
        avgD = np.mean(dist)
        if avgD < bestD:
            bestD = avgD
            bestT = T
    return bestT

if __name__ == "__main__":
    obj_name = "liq_container"
    num_pc = 4
    source_pc = read_canonical_model(obj_name)
    cand_yaw = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    for i in range(num_pc):
        target_pc = load_pc(obj_name, i)
        Tinit = best_init(source_pc, target_pc, cand_yaw)
        Tfinal = ICP(source_pc, target_pc, initT=Tinit, max_iters=100, delta=1e-3, trim_ratio=0.85)
        visualize_icp_result(source_pc, target_pc, Tfinal)
