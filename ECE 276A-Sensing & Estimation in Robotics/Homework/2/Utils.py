import torch

def skew(v: torch.Tensor) -> torch.Tensor:
    vx, vy, vz = v[0], v[1], v[2]
    O = torch.zeros((), dtype=v.dtype, device=v.device)  # scalar 0
    return torch.stack(
        [
            torch.stack([O, -vz, vy]),
            torch.stack([vz, O, -vx]),
            torch.stack([-vy, vx, O]),
        ]
    )


def rodrigues(w):
    theta = torch.norm(w) + 1e-8
    k = w / theta
    K = skew(k)
    I = torch.eye(3)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R
