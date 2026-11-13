from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConfigGlobal:
    # data_dir = "/media/fhz/Learning/Data/ECE276A/PR3"
    data_dir = r"E:\Data\ECE276A\PR3"
    dataset_id = 1
    out_dir = "Output"


@dataclass(frozen=True)
class ConfigLandmarksFilter:
    min_obs = 15
    max_landmarks: int = -1  # set to -1 to disable limit


@dataclass(frozen=True)
class ConfigOdometry:
    sigma_v = 0.03
    sigma_w = 0.015


@dataclass(frozen=True)
class ConfigSLAM:
    obs_noise = 2.0
    init_var_xy = 2.0
    init_var_z = 10.0


cfgG = ConfigGlobal()
cfgLF = ConfigLandmarksFilter()
