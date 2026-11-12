import torch
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