import numpy as np
from Utils.Utils import bresenham2D


class OccupancyGridMap:

    def __init__(
        self,
        res=0.05,
        xmin=-30,
        xmax=30,
        ymin=-30,
        ymax=30,
        lo_occ=1.0,
        lo_free=-0.5,
        lo_min=-5.0,
        lo_max=5.0,
    ):

        self.res = res
        self.xmin = xmin
        self.ymin = ymin

        self.sizex = int((xmax - xmin) / res)
        self.sizey = int((ymax - ymin) / res)

        self.log_odds = np.zeros((self.sizex, self.sizey), dtype=np.float32)

        self.lo_occ = lo_occ
        self.lo_free = lo_free
        self.lo_min = lo_min
        self.lo_max = lo_max

    def world_to_map(self, x, y):
        mx = np.floor((x - self.xmin) / self.res).astype(np.int32)
        my = np.floor((y - self.ymin) / self.res).astype(np.int32)
        return mx, my

    def update(self, robot_pose, P_world):
        rx, ry = self.world_to_map(robot_pose[0], robot_pose[1])
        if not (0 <= rx < self.sizex and 0 <= ry < self.sizey):
            return
        mxs, mys = self.world_to_map(P_world[:, 0], P_world[:, 1])
        for i in range(P_world.shape[0]):
            mx, my = mxs[i], mys[i]
            if 0 <= mx < self.sizex and 0 <= my < self.sizey:
                xs, ys, n = bresenham2D(rx, ry, mx, my)
                if n > 1:
                    self.log_odds[xs[: n - 1], ys[: n - 1]] += self.lo_free
                self.log_odds[mx, my] += self.lo_occ

    def clip(self):
        np.clip(self.log_odds, self.lo_min, self.lo_max, out=self.log_odds)

    def get_probability(self):
        return 1 - 1 / (1 + np.exp(self.log_odds))

class ColoredGridMap:

    def __init__(
        self,
        res=0.05,
        xmin=-30,
        xmax=30,
        ymin=-30,
        ymax=30,
    ):
        self.res = res
        self.xmin = xmin
        self.ymin = ymin
        self.sizex = int((xmax - xmin) / res)
        self.sizey = int((ymax - ymin) / res)
        self.map = np.zeros((self.sizex, self.sizey, 3), dtype=np.uint8)

    def world_to_map(self, x, y):
        mx = np.floor((x - self.xmin) / self.res).astype(np.int32)
        my = np.floor((y - self.ymin) / self.res).astype(np.int32)
        return mx, my

    def update(self,wP, colors):
        mx, my = self.world_to_map(wP[:, 0], wP[:, 1])
        valid = (0 <= mx) & (mx < self.sizex) & (0 <= my) & (my < self.sizey)
        mx, my = mx[valid], my[valid]
        self.map[mx, my] = colors[valid]
