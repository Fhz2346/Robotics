import numpy as np
from init_task import *


def shortcut_smooth(path, attempts=100, rng=None):
    """Remove unnecessary bends without sacrificing collision safety."""
    if path is None or len(path) < 3:
        return path
    out = [p.copy() for p in path]
    for _ in range(attempts):
        if len(out) < 3:
            break
        i, j = sorted(rng.choice(len(out), size=2, replace=False))
        if j <= i + 1:
            continue
        if Cs.is_edge_valid(out[i], out[j]):
            out = out[:i+1] + out[j:]
    return np.asarray(out)


def densify_path(path, max_step=np.deg2rad(2.0)):
    """Interpolate each edge for a smooth animation."""
    if path is None or len(path) == 0:
        return np.empty((0, 2))
    dense = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        d = As.displacement(a, b)
        n = max(1, int(np.ceil(np.max(np.abs(d)) / max_step)))
        dense.extend([As.wrap(a + (k / n) * d) for k in range(1, n + 1)])
    return np.asarray(dense)
