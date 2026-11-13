import numpy as np
from Config import cfgG


def load_data(file_name=f"{cfgG.data_dir}/dataset{cfgG.dataset_id:02d}/dataset{cfgG.dataset_id:02d}.npy"):
    data = np.load(file_name, allow_pickle = True).item()
    V = data["v_t"] # linear velocities
    W = data["w_t"] # angular velocities
    timestamps = data["timestamps"] # UNIX timestamps
    features = data["features"] # 4 x num_features x t : pixel coordinates of the visual features
    features = features.transpose(2,1,0) # transpose to t x num_features x 4
    K_l = data["K_l"] # intrinsic calibration matrix of left camera
    K_r = data["K_r"] # intrinsic calibration matrix of right camera
    extL_T_imu = data["extL_T_imu"] # transformation from imu frame to left camera frame
    extR_T_imu = data["extR_T_imu"] # transformation from imu frame to right camera frame
    return V, W, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu

def load_images(file_name=f"{cfgG.data_dir}/dataset{cfgG.dataset_id:02d}/dataset{cfgG.dataset_id:02d}_imgs.npy"):
    data = np.load(file_name, allow_pickle = True).item()
    imgs_l = data["cam_imgs_L"] # list of left camera images
    imgs_r = data["cam_imgs_R"] # list of right camera images
    return imgs_l, imgs_r