import numpy as np
import cv2
import matplotlib.pyplot as plt
from Sensor.Kinect import Kinect
from src.load_data import load_dataset
from Utils.Utils import get_R
from Utils.Map import OccupancyGridMap
from Sensor.Lidar import LiDAR
from Vis.Visualize import save_map
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as D3


# ===== load everything =====
ds = load_dataset(20, data_dir="../data")
lidar = LiDAR.from_dataset(ds.lidar)

data = np.load("Output/Task2/10_scan_matching.npz")
traj = data["traj"]
Theta = data["Theta"]
Rs = get_R(Theta)

kin = Kinect.from_dataset(ds.kinect)
kin.id = 20

grid_map = OccupancyGridMap()

c_pts_, colors = kin.reconstruct(0, 0)

# ===== use first frame =====
k = 0

rgb = kin.get_rgb(k)
disparity = kin.get_disparity(k)

H, W = disparity.shape
i, j = np.indices((H, W))

# ===== disparity → depth =====
dd = -0.00304 * disparity + 3.31
mask_dd = dd > 0
mask_dd_flat = mask_dd.reshape(-1)

depth = np.zeros_like(dd, dtype=np.float64)
depth[mask_dd] = 1.03 / dd[mask_dd]

# ===== depth → 3D camera =====
fx = fy = 585.05
cx = 242.94
cy = 315.84

K = np.array(
    [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ],
    dtype=np.float64,
)
K_inv = np.linalg.inv(K)

i, j = np.indices((H, W))
pixels = np.stack([j, i, np.ones_like(i)], axis=-1)
c_pts = depth[..., None] * (pixels @ K_inv.T)
c_pts = c_pts[mask_dd]

# ===== camera(depth) -> robot center extrinsic (given) =====
R_base = np.array(
    [
        [0, 0, 1],  # x_r =  z_c
        [-1, 0, 0],  # y_r = -x_c
        [0, -1, 0],  # z_r = -y_c
    ],
    dtype=np.float64,
)
r_t_c = np.array([0.18, 0.005, 0.36], dtype=np.float64)  # meters
roll, pitch, yaw = 0.0, 0.36, 0.021
R_rpy = D3.Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
r_R_c = R_rpy @ R_base

# pts: (N,3) in camera frame -> (N,3) in robot frame
r_pts = c_pts @ r_R_c.T + r_t_c

# ===== robot -> world (2D pose) =====
w_R_r = D3.Rotation.from_euler("z", Theta[k]).as_matrix()
w_pts = r_pts @ w_R_r.T
w_pts[:,:2] += traj[k]

# ===== floor filter =====
# floor_mask = np.abs(w_pts[:, 2]) < 10
# w_pts = w_pts[floor_mask]

# ===== RGB mapping =====
rgbi = (526.37 * i + 19276 - 7877.07 * dd) / 585.051
rgbj = (526.37 * j + 16662) / 585.051

# rgbi = rgbi.astype(np.int32).reshape(-1)[valid_flat][floor_mask]
# rgbj = rgbj.astype(np.int32).reshape(-1)[valid_flat][floor_mask]
rgbi = rgbi.astype(np.int32).reshape(-1)[mask_dd_flat]
rgbj = rgbj.astype(np.int32).reshape(-1)[mask_dd_flat]

H_rgb, W_rgb, _ = rgb.shape
mask_rgb = (
    (rgbi >= 0) & (rgbi < H_rgb) &
    (rgbj >= 0) & (rgbj < W_rgb)
)

w_pts = w_pts[mask_rgb]
rgbi = rgbi[mask_rgb]
rgbj = rgbj[mask_rgb]

colors = rgb[rgbi, rgbj]

# ===== world → grid =====
mx = ((w_pts[:, 0] - grid_map.xmin) / grid_map.res).astype(int)
my = ((w_pts[:, 1] - grid_map.ymin) / grid_map.res).astype(int)

valid_grid = (
    (mx >= 0) & (mx < grid_map.sizex) &
    (my >= 0) & (my < grid_map.sizey)
)

mx = mx[valid_grid]
my = my[valid_grid]
colors = colors[valid_grid]

# ===== create single frame color map =====
color_map = np.zeros((grid_map.sizex, grid_map.sizey, 3), dtype=np.uint8)

color_map[mx, my] = colors

# plt.figure(figsize=(6,6))
# plt.title("First Frame Floor Texture")
# plt.imshow(color_map)
# plt.axis("off")
# plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# 下采样防止卡顿
idx = np.random.choice(len(w_pts), size=min(20000, len(w_pts)), replace=False)


ax.scatter(w_pts[idx, 0], w_pts[idx, 1], w_pts[idx, 2], c=colors[idx] / 255.0, s=1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.title("World Point Cloud (First Frame)")
plt.show()
