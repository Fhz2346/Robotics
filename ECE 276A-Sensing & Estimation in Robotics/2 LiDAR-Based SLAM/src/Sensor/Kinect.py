import numpy as np
import cv2
from load_data import KinectData
from Config import cfgG


class Kinect:

    def __init__(self, disparity_time_stamps, rgb_time_stamps):
        self.Td = np.asarray(disparity_time_stamps, dtype=np.float64)
        self.Trgb = np.asarray(rgb_time_stamps, dtype=np.float64)
        self.init()

    def init(self):
        self.Nd = self.Td.shape[0]
        self.Nrgb = self.Trgb.shape[0]
        self.H, self.W = 480, 640
        fx = fy = 585.05
        cx = 242.94
        cy = 315.84
        self.K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.K_inv = np.linalg.inv(self.K)
        V, U = np.indices((self.H, self.W))
        self.VU = (V, U)
        pixels = np.stack([U, V, np.ones_like(V)], axis=-1)
        self.rays = pixels @ self.K_inv.T

        self.rgbi = (526.37 * V + 19276) / 585.051  # precompute for efficiency
        self.rgbj = (526.37 * U + 16662) / 585.051

    @staticmethod
    def from_dataset(ds_kinect: KinectData):
        return Kinect(
            disparity_time_stamps=ds_kinect.disparity_time_stamps,
            rgb_time_stamps=ds_kinect.rgb_time_stamps,
        )

    def get_rgb(self, k):
        rgb = cv2.imread(
            f"{cfgG.data_dir}/dataRGBD/RGB{cfgG.dataset_id}/rgb{cfgG.dataset_id}_{k+1}.png"
        )
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    def get_disparity(self, k):
        disparity = cv2.imread(
            f"{cfgG.data_dir}/dataRGBD/Disparity{cfgG.dataset_id}/disparity{cfgG.dataset_id}_{k+1}.png",
            cv2.IMREAD_UNCHANGED,
        )
        return disparity

    def reconstruct(self, k_disp, k_rgb):
        disparity = self.get_disparity(k_disp)
        dd = -0.00304 * disparity + 3.31
        mask_dd2d = dd > 0
        mask_dd = mask_dd2d.reshape(-1)
        depth = np.zeros_like(dd, dtype=float)
        depth[mask_dd2d] = 1.03 / dd[mask_dd2d]
        c_pts = depth[..., None] * self.rays
        c_pts = c_pts[mask_dd2d]
        rgbi = self.rgbi - 7877.07 * dd / 585.051
        rgbj = self.rgbj
        rgbi = rgbi.astype(np.int32).reshape(-1)[mask_dd]
        rgbj = rgbj.astype(np.int32).reshape(-1)[mask_dd]
        mask_rgb = (rgbi >= 0) & (rgbi < self.H) & (rgbj >= 0) & (rgbj < self.W)
        c_pts = c_pts[mask_rgb]
        rgb = self.get_rgb(k_rgb)
        colors = rgb[rgbi, rgbj]
        return c_pts, colors
