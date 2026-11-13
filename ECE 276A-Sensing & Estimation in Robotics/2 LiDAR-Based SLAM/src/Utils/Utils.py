import numpy as np
import time
from numba import njit
from gtsam import Pose2


def tic():
    return time.time()


def toc(tstart, name="Operation"):
    print("%s took: %s sec.\n" % (name, (time.time() - tstart)))


def bresenham2D(sx, sy, ex, ey):
    """
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
            (sx, sy)	start point of ray
            (ex, ey)	end point of ray
    """
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex - sx)
    dy = abs(ey - sy)
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx  # swap

    if dy == 0:
        q = np.zeros((dx + 1, 1))
    else:
        q = np.append(
            0,
            np.greater_equal(
                np.diff(
                    np.mod(
                        np.arange(
                            np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy
                        ),
                        dx,
                    )
                ),
                0,
            ),
        )
    if steep:
        if sy <= ey:
            y = np.arange(sy, ey + 1)
        else:
            y = np.arange(sy, ey - 1, -1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx, ex + 1)
        else:
            x = np.arange(sx, ex - 1, -1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x, y))


def get_map_idx(ts_src, ts_query):
    idx = np.searchsorted(ts_src, ts_query, side="right") - 1
    idx = np.clip(idx, 0, len(ts_src) - 1)
    idx = np.asarray(idx, dtype=np.int32)
    return idx


@njit
def bresenham2D(x0, y0, x1, y1):
    """
    Return integer grid cells along the line from (x0,y0) to (x1,y1).
    Output: xs, ys, n (valid length)
    xs, ys are int32 arrays length <= max(dx,dy)+1
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    nmax = max(dx, dy) + 1
    xs = np.empty(nmax, dtype=np.int32)
    ys = np.empty(nmax, dtype=np.int32)

    err = dx - dy
    i = 0
    while True:
        xs[i] = x0
        ys[i] = y0
        i += 1

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return xs, ys, i
