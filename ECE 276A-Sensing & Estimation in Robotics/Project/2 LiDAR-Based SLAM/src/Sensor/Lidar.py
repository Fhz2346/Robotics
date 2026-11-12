import numpy as np
from load_data import LidarData


def ra2xy(r, a):
    x = r * np.cos(a)
    y = r * np.sin(a)
    xy = np.stack([x, y], axis=1)
    return xy


class LiDAR:

    def __init__(self, ranges, time_stamps, angle_lim, angle_increment, range_lim):
        self.r = np.asarray(ranges, dtype=np.float64)
        self.T = np.asarray(time_stamps, dtype=np.float64)

        angle_min, angle_max = angle_lim
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.angle_increment = float(angle_increment)

        range_min, range_max = range_lim
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.init()

    def init(self):
        self.N = self.r.shape[0] # number of scans
        self.num_rays = self.r.shape[1]
        idx = np.arange(self.num_rays, dtype=np.int64)
        self.angles = self.angle_min + idx * self.angle_increment

    def get_pc(self, k=0, stride=1):
        """stride : for angular downsampling"""
        r = self.r[k]
        a = self.angles

        keep = np.arange(0, self.num_rays, stride, dtype=np.int64)
        r = r[keep]
        a = a[keep]

        valid = np.isfinite(r) & (r > self.range_min) & (r < self.range_max)
        r = r[valid]
        a = a[valid]

        xy = ra2xy(r, a)
        return xy

    @staticmethod
    def from_dataset(lidarD: LidarData):
        return LiDAR(
            ranges=lidarD.ranges,
            time_stamps=lidarD.time_stamps,
            angle_lim=(lidarD.angle_min, lidarD.angle_max),
            angle_increment=lidarD.angle_increment,
            range_lim=(lidarD.range_min, lidarD.range_max),
        )
