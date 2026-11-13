import numpy as np
import cv2
import Utils.Transformation2D as D2

from load_data import load_dataset
from Utils.Map import OccupancyGridMap
from Sensor.Lidar import LiDAR
import Vis.Plot as Vis
from Config import cfgG, cfg3_1


def prob2frame(prob):
    frame = (prob * 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


ds = load_dataset()
lidar = LiDAR.from_dataset(ds.lidar)

data = np.load(f"{cfgG.out_dir}/scan_matching.npz")
traj = data["traj"]
Theta = data["Theta"]
Rs = D2.get_R(Theta)


if __name__ == "__main__":
    grid_map = OccupancyGridMap()
    video = cv2.VideoWriter(f"{cfgG.out_dir}/mapping_process.mp4", cv2.VideoWriter_fourcc(*"mp4v"),cfg3_1.fps, (grid_map.sizey, grid_map.sizex))
    for k in range(0, lidar.N, cfg3_1.step_lidar):
        P = lidar.get_pc(k, stride=1)
        R = Rs[k]
        P_w = (R @ P.T).T + traj[k]
        grid_map.update(traj[k], P_w)
        if k % cfg3_1.clip_every == 0:
            grid_map.clip()
        prob = grid_map.get_probability()
        frame = prob2frame(prob)
        video.write(frame)
    video.release()
    prob = grid_map.get_probability()
    Vis.save_map(prob)
    print("Mapping finished.")
