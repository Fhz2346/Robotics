import torch
from torch.linalg import norm

## -----Vectorized quaternion operations-----


def qnormalize(q, eps=1e-12):
    return q / (norm(q, dim=-1, keepdim=True) + eps)


def qinv(q: torch.Tensor):
    w = q[:, 0].unsqueeze(-1)
    v = -q[:, 1:]
    return torch.cat([w, v], dim=-1)


def qmult(q1: torch.Tensor, q2: torch.Tensor):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    q = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
    q = torch.stack(q, dim=-1)
    return q


def exp_pure(v: torch.Tensor):
    vNorm = norm(v, dim=-1, keepdim=True)
    w_ = torch.cos(vNorm)
    v_ = torch.where(vNorm > 1e-8, torch.sin(vNorm) * v / vNorm, torch.zeros_like(v))
    return torch.cat([w_, v_], dim=-1)


def rot_vec(q: torch.Tensor, eps=1e-12):
    q = qnormalize(q)
    w = torch.clamp(q[:, 0], -1.0, 1.0)
    w = w.unsqueeze(-1)
    q = torch.where(w < 0, -q, q)
    v = q[:, 1:]
    vNorm = norm(v, dim=-1).unsqueeze(-1)
    theta = 2.0 * torch.atan2(vNorm, w)
    return torch.where(vNorm > eps, theta * v / (vNorm + eps), 2.0 * v)
