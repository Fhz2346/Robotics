import numpy as np
import Vis.Plot as Vis
from Robot import Robot, SensorType
from Config import cfgG

rbt = Robot()
activation = [SensorType.ENCODER, SensorType.IMU]
for s in activation:
    rbt.activate(s)
rbt.build_Tmap()

enc = rbt.enc
imu = rbt.imu
i2e = rbt.Tmap[("enc", "imu")]
e2i= rbt.Tmap[("imu", "enc")]

traj = np.zeros((rbt.enc.N, 2))
Theta = np.zeros(rbt.enc.N)
x, y, theta = 0.0, 0.0, 0.0
for i in range(rbt.enc.N - 1):
    dt = enc.T[i + 1] - enc.T[i]
    d = enc.estimate_dist(i)
    w = imu.w[e2i[i]]
    theta += w * dt
    Theta[i + 1] = theta
    x += d * np.cos(theta)
    y += d * np.sin(theta)
    traj[i + 1, :] = [x, y]

np.savez(f"{cfgG.out_dir}/odom.npz", traj=traj, Theta=Theta)
Vis.save_traj(traj)
Vis.save_Theta(Theta, rbt.enc.T)
