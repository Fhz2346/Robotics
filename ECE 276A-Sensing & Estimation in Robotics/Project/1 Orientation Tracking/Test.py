from Utils.Calibration import imu_raw2physical, integrate_gyro_to_quat
from Panorama import build_panorama_sphere_cylinder_to_video, build_panorama_sphere_cylinder
from Utils.Loss import loss_motion, loss_observation
from Utils.Load import ts_i, trainset, idx_i2c
from Utils.tQuat import qnormalize
from Vis.Visualize import plot_RPY_v
from transforms3d.euler import quat2mat, quat2euler
import torch
import cv2
import numpy as np

torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    A_np, W_np = imu_raw2physical()
    Q = integrate_gyro_to_quat(W_np)  # Initialize Q by gyro integration

    A, W, ts = [torch.tensor(item) for item in (A_np, W_np, ts_i)]
    Q = torch.tensor(Q, requires_grad=True)
    N = ts.shape[0]

    # Vanilla gradient descent
    lr = 1e-2
    iters = 200
    for it in range(iters):
        loss = loss_motion(Q, W, ts) + loss_observation(Q, A)
        if Q.grad is not None:
            Q.grad.zero_()
        loss.backward()
        with torch.no_grad():
            Q -= lr * Q.grad
            Q[0] = torch.tensor([1., 0., 0., 0.])
            Q[:] = qnormalize(Q) # Project
        if it % 20 == 0:
            print(f"iter {it:4d} | loss = {loss.item():.6f}")

    # Evaluate and visualize
    Q_est = Q.detach()
    R_est = [quat2mat(Q_est[k]) for k in range(N)]
    R_est = np.array(R_est)
    RPY_est = np.array([quat2euler(Q_est[k].numpy(), axes="sxyz") for k in range(N)])
    plot_RPY_v(ts_i, RPY_est)
    pano = build_panorama_sphere_cylinder(R_est[idx_i2c])
    cv2.imwrite(f"Output/Panorama/{trainset}/panorama{trainset}.png", cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
    pano, out_mp4, n_frames = build_panorama_sphere_cylinder_to_video(
        R_est[idx_i2c],
        out_mp4=f"Output/Panorama/{trainset}/panorama_video{trainset}.mp4",
        fps=30,
    )
