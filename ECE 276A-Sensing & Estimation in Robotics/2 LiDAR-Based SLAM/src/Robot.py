import numpy as np
from enum import Enum
from Utils.Utils import get_map_idx
import Utils.Transformation2D as D2
import scipy.spatial.transform as D3
from itertools import combinations
from load_data import load_encoders, load_imu, load_kinect, load_lidar


class SensorType(Enum):
    ENCODER = 0
    IMU = 1
    LIDAR = 2
    RGBD = 3


class RGBDType(Enum):
    DISPARITY = 0
    RGB = 1


class Robot:
    def __init__(self):
        self.traj = None
        self.Theta = None
        self.activated = {
            SensorType.ENCODER: False,
            SensorType.IMU: False,
            SensorType.LIDAR: False,
            SensorType.RGBD: False,
        }
        self.Tmap = {}

    def activate(self, sensor_type: SensorType):
        if self.activated[sensor_type]:
            print(f"{sensor_type.name} already activated.")
            return

        if sensor_type == SensorType.ENCODER:
            from Sensor.Encoder import Encoder
            self.enc = Encoder.from_dataset(load_encoders())
        elif sensor_type == SensorType.IMU:
            from Sensor.IMU import IMU
            self.imu = IMU.from_dataset(load_imu())
        elif sensor_type == SensorType.RGBD:
            from Sensor.Kinect import Kinect
            self.kin = Kinect.from_dataset(load_kinect())
            R_base = np.array(
                [
                    [0, 0, 1],  # x_r =  z_c
                    [-1, 0, 0],  # y_r = -x_c
                    [0, -1, 0],  # z_r = -y_c
                ],
                dtype=np.float64,
            )
            self.r_t_c = np.array([0.18, 0.005, 0.36], dtype=np.float64)  # meters
            roll, pitch, yaw = 0.0, 0.36, 0.021
            R_rpy = D3.Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
            self.r_R_c = R_rpy @ R_base
        elif sensor_type == SensorType.LIDAR:
            from Sensor.Lidar import LiDAR
            self.lid = LiDAR.from_dataset(load_lidar())

        self.activated[sensor_type] = True

    def activate_all(self, cand):
        for st in cand:
            self.activate(st)

    def build_Tmap(self):
        Tstr2data = {}
        if self.activated[SensorType.ENCODER]:
            Tstr2data["enc"] = self.enc.T
        if self.activated[SensorType.IMU]:
            Tstr2data["imu"] = self.imu.T
        if self.activated[SensorType.LIDAR]:
            Tstr2data["lid"] = self.lid.T
        if self.activated[SensorType.RGBD]:
            Tstr2data["rgb"] = self.kin.Trgb
            Tstr2data["disp"] = self.kin.Td
        keys = list(Tstr2data.keys())
        for k1, k2 in combinations(keys, 2):
            T1 = Tstr2data[k1]
            T2 = Tstr2data[k2]
            self.Tmap[(k1, k2)] = get_map_idx(T1, T2)
            self.Tmap[(k2, k1)] = get_map_idx(T2, T1)

    def odom_rel(self, e0, e1):
        assert self.activated[SensorType.ENCODER] and self.activated[SensorType.IMU]
        e2i = self.Tmap[("imu", "enc")]
        dx, dy, dtheta = 0.0, 0.0, 0.0
        for i in range(e0, e1):
            dt = self.enc.T[i + 1] - self.enc.T[i]
            d = self.enc.estimate_dist(i)
            w = self.imu.w[e2i[i]]
            dtheta += w * dt
            dx += d * np.cos(dtheta)
            dy += d * np.sin(dtheta)
        return D2.make_T(dtheta, [dx, dy])
