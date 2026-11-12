import numpy as np


def select_landmarks(features, min_obs=4, max_landmarks=500):
    valid = np.logical_and(features[:, :, 0] != -1, features[:, :, 2] != -1)
    counts = np.sum(valid, axis=0)
    idx = np.where(counts >= min_obs)[0]
    idx = idx[np.argsort(counts[idx])[::-1]]
    return idx[:max_landmarks]
