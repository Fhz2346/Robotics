from dataclasses import dataclass
import gtsam
import numpy as np


@dataclass(frozen=True)
class ConfigGlobal:
    data_dir = "/media/fhz/Learning/Data/ECE276A/PR2"
    # data_dir = r"E:\Data\ECE276A\PR2"
    dataset_id = 20
    out_dir = "Output"


@dataclass(frozen=True)
class Config3_1:
    fps = 30
    clip_every = 10
    step_lidar = 20


@dataclass(frozen=True)
class Config3_2:
    step_lidar = 20
    th_floor = 0.1


@dataclass(frozen=True)
class Config4:
    data_dir: str = "/media/fhz/Learning/Data/ECE276A/PR2"
    stride = 4
    icp_every = 20  # same spirit as your Task2.py
    loop_every = 10  # try loop closure occasionally
    min_sep = 300  # only consider far past nodes
    dist_gate = 3.0  # meters: candidate distance threshold
    th_err = 0.01  # ICP error threshold (try to filter out bad loop closures, tune it!)
    max_loop_edges = 100  # keep it small (simplified)
    cand_yaw = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.25, 0.25, 0.12]))
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))


cfgG = ConfigGlobal()
cfg3_1 = Config3_1()
cfg3_2 = Config3_2()
cfg4 = Config4()