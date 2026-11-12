import numpy as np
from scipy.spatial import cKDTree as KDTree
from Vis.Plot import save_pc
from Utils.Transformation2D import apply_T, fit_rigid, get_R_

def filter_pc(P0, P1, radius_scale=3.0):
    c0 = P0.mean(axis=0)
    c1 = P1.mean(axis=0)
    r0 = np.sqrt(np.mean(np.linalg.norm(P0 - c0, axis=1) ** 2))
    r1 = np.sqrt(np.mean(np.linalg.norm(P1 - c1, axis=1) ** 2))
    R = radius_scale * min(r0, r1)
    mask0 = np.linalg.norm(P0 - c0, axis=1) < R
    mask1 = np.linalg.norm(P1 - c1, axis=1) < R
    return P0[mask0], P1[mask1]

def filter_close(P0, P1):
    center0 = P0.mean(axis=0)
    center1 = P1.mean(axis=0)
    var0 = np.mean(np.linalg.norm(P0 - center0, axis=1) ** 2)
    var1 = np.mean(np.linalg.norm(P1 - center1, axis=1) ** 2)
    if abs(var0 - var1) > 8.0:
        if var0 > var1:
            d_src = np.linalg.norm(P0 - center1, axis=1)
            mask = d_src < 2.5 * np.sqrt(var1)
            P0 = P0[mask]
        else:
            d_tgt = np.linalg.norm(P1 - center0, axis=1)
            mask = d_tgt < 2.5 * np.sqrt(var0)
            P1 = P1[mask]
    return P0, P1


def NNC(src, dst):
    tree = KDTree(dst)
    dist, idx = tree.query(src, k=1, workers=-1)
    return idx, dist


def ICP2D(source, target, init_T, max_iters=40, tol_mse=1e-6, trim_ratio=0.8):
    assert trim_ratio > 0 and trim_ratio <= 1.0
    T = np.asarray(init_T, dtype=np.float64).copy()
    last = None
    for _ in range(max_iters):
        src = apply_T(source, T)
        idx, dist = NNC(src, target)
        dst = target[idx]

        k = int(trim_ratio * dist.shape[0])
        keep = np.argpartition(dist, k - 1)[:k]
        src = src[keep]
        dst = dst[keep]
        dist = dist[keep]

        mse = float(np.mean(dist ** 2))
        R_inc, t_inc = fit_rigid(src, dst)
        T_inc = np.eye(3, dtype=np.float64)
        T_inc[:2, :2] = R_inc
        T_inc[:2, 2] = t_inc
        T = T_inc @ T
        if last is not None and abs(last - mse) < tol_mse:
            break
        last = mse
    return T, last


def ICP2D_debug(source, target, init_T, max_iters=40, tol_mse=1e-6, trim_ratio=0.8):
    assert trim_ratio > 0 and trim_ratio <= 1.0
    T = np.asarray(init_T, dtype=np.float64).copy()
    last = None
    for _ in range(max_iters):
        src = apply_T(source, T)
        save_pc(src, target, title="ICP Init")
        idx, dist = NNC(src, target)
        dst = target[idx]

        k = int(trim_ratio * dist.shape[0])
        keep = np.argpartition(dist, k - 1)[:k]
        src = src[keep]
        dst = dst[keep]
        dist = dist[keep]

        mse = float(np.mean(dist ** 2))
        R_inc, t_inc = fit_rigid(src, dst)
        T_inc = np.eye(3, dtype=np.float64)
        T_inc[:2, :2] = R_inc
        T_inc[:2, 2] = t_inc
        T = T_inc @ T
        if last is not None and abs(last - mse) < tol_mse:
            save_pc(src, target, title="ICP End")
            break
        last = mse
    return T


def best_init(source, target, cand_yaw, trim_ratio=0.7):
    # t0 = target.mean(axis=0) - source.mean(axis=0)
    bestD = np.inf
    bestT = np.eye(3, dtype=np.float64)
    for yaw in cand_yaw:
        Rz = get_R_(yaw)
        T = np.eye(3)
        T[:2, :2] = Rz
        # T[:2, 2] = t0
        src = apply_T(source, T)
        _, dist = NNC(src, target)
        dist_sorted = np.sort(dist)
        k = int(len(dist_sorted) * trim_ratio)
        trimmed_mean = np.mean(dist_sorted[:k] ** 2)
        if trimmed_mean < bestD:
            bestD = trimmed_mean
            bestT = T
    return bestT, bestD
