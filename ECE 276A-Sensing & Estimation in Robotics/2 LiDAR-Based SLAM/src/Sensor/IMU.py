import numpy as np
from load_data import ImuData

class IMU:
    def __init__(self, angular_velocity, linear_acceleration, time_stamps):
        self.w = np.asarray(angular_velocity, dtype=np.float64)
        self.v = np.asarray(linear_acceleration, dtype=np.float64)
        self.T = np.asarray(time_stamps, dtype=np.float64)
        self.init()

    def init(self):
        self.N = self.T.shape[0]
        self.w = self.w[:, 2] # only yaw is used in this project

    @staticmethod
    def from_dataset(ds_imu: ImuData):
        return IMU(
            angular_velocity=ds_imu.angular_velocity,
            linear_acceleration=ds_imu.linear_acceleration,
            time_stamps=ds_imu.time_stamps,
        )
