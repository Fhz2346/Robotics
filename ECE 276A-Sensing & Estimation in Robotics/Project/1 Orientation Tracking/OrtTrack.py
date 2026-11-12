import torch
import numpy as np
from transforms3d.euler import quat2euler, quat2mat

from Calibration import imu_raw2physical, integrate_gyro_to_quat
from Utils.Load import ts_i, idx_v2i, RPY_v, trainset
from Vis.Visualize import plot_rpy
from Utils.tQuat import qnormalize, qmult, qinv, rot_vec, exp_pure

torch.set_default_dtype(torch.float64)

# ---------- Vectorized loss terms ----------
def loss_motion(Q: torch.Tensor, W: torch.Tensor, ts: torch.Tensor):
    dt = ts[1:] - ts[:-1]
    dt = dt.unsqueeze(1)
    dq = exp_pure(0.5 * dt * W[:-1])

    Q_pred = qmult(Q[:-1], dq)
    Q_err = qmult(qinv(Q[1:]), Q_pred)
    r_m = rot_vec(Q_err)
    return 0.5 * torch.sum(r_m**2)


def loss_observation(Q, A):
    g = torch.tensor([0, 0, 0, 1])
    A_pred_q = qmult(qmult(qinv(Q), g), Q)
    A_pred = A_pred_q[:, 1:]
    r_o = A - A_pred
    return 0.5 * torch.sum(r_o**2)


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
    R_est = [quat2mat(Q_est[k], axes="sxyz") for k in range(N)]
    np.save(f"R_est{trainset}.npy", R_est)
    RPY_est = [quat2euler(Q_est[k], axes="sxyz") for k in range(N)]
    RPY_est = np.array(RPY_est)
    RPY_gt_aligned = RPY_v[idx_v2i]
    plot_rpy(ts_i, RPY_gt_aligned, RPY_est, title_prefix=f"Trainset{trainset} - ")
