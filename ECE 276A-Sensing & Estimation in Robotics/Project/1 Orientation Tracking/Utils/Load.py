import pickle
import numpy as np
from transforms3d.euler import mat2euler
from transforms3d.quaternions import mat2quat, qmult
from Vis.Visualize import plot_W, plot_A
from Vis.Video import imgs_to_video
from Utils.nQuat import inv
cam_avail = [1, 2, 8, 9, 10, 11]

def read_data(fname):
    with open(fname, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return d


def Mat2RPY(R):
    N = R.shape[0]
    RPY = np.zeros((N, 3))
    for k in range(N):
        euler = mat2euler(R[k], axes="sxyz")
        RPY[k] = np.array(euler)
    return RPY


def Mat2Quat(R):
    N = R.shape[0]
    Q = np.zeros((N, 4))
    for k in range(N):
        q = mat2quat(R[k])
        Q[k] = np.array(q)
    return Q


def pure_rotation(Q):
    g = np.array([0.0, 0.0, 0.0, 1.0])
    N = Q.shape[0]
    A = np.zeros((N, 4))
    for k in range(N):
        q = Q[k]
        A[k] = qmult(qmult(inv(q), g), q)
    return A


def get_map_idx(ts_src, ts_query):
    idx = np.searchsorted(ts_src, ts_query, side="right") - 1
    idx = np.clip(idx, 0, len(ts_src) - 1)
    idx = np.asarray(idx, dtype=np.int32)
    return idx


trainset = 10
is_test = trainset >= 10
if not is_test:
    cfile = f"../data/trainset/cam/cam{trainset}.p"
    ifile = f"../data/trainset/imu/imuRaw{trainset}.p"
    vfile = f"../data/trainset/vicon/viconRot{trainset}.p"
else:
    cfile = f"../data/testset/cam/cam{trainset}.p"
    ifile = f"../data/testset/imu/imuRaw{trainset}.p"
    vfile = f"../data/testset/vicon/viconRot{trainset}.p"

imud = read_data(ifile)
ts_i = imud[0].reshape(-1)
A_raw = imud[1:4].transpose(1, 0)
W_raw = imud[4:7].transpose(1, 0)
# plot_W(ts_i, W_raw, unit="mV")
# plot_A(ts_i, A_raw, unit="mV")

if not is_test:
    vicd = read_data(vfile)
    ts_v = vicd["ts"].reshape(-1)
    R_v = vicd["rots"].transpose(2, 0, 1)
    valid_mask = ~np.isnan(R_v).any(axis=(1, 2))
    R_v = R_v[valid_mask]
    ts_v = ts_v[valid_mask]
    
    RPY_v = Mat2RPY(R_v)
    Q_v = Mat2Quat(R_v)
    A_v = pure_rotation(Q_v)
    idx_v2i = get_map_idx(ts_v, ts_i)

if trainset in cam_avail:
    camd = read_data(cfile)
    ts_c = camd["ts"].reshape(-1)
    Imgs = camd["cam"].transpose(3, 0, 1, 2)
    imgs_to_video(Imgs, out_path=f"Output/Panorama/{trainset}/cam{trainset}.mp4", fps=30)
    idx_i2c = get_map_idx(ts_i, ts_c)
    if not is_test:
        idx_v2c = get_map_idx(ts_v, ts_c)
        
