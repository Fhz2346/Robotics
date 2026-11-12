import torch
from Utils import rodrigues, skew
from torch.linalg import det

u1 = torch.tensor(
    [
        [0.2953, 0.1658, 0.3040, 0.1492, 0.1247, 0.4638, 0.6996, 0.1004],
        [-0.3088, -0.5017, 0.1686, -0.0960, -0.0581, 0.0349, 0.6246, -0.5315],
    ],
    dtype=torch.float32,
).T  # (N,2)

u2 = torch.tensor(
    [
        [0.2909, 0.1647, 1.8975, 1.2081, 1.2510, 1.1337, 2.5496, 0.1398],
        [0.5712, 1.0140, 0.3364, 1.4632, 1.3761, 0.5984, 0.1150, 1.1851],
    ],
    dtype=torch.float32,
).T  # (N,2)

def triangulate_point(P1, P2, x1, x2):
    # x1,x2: (3,) homogeneous
    A = torch.stack([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1],
    ], dim=0)
    _, _, Vt = torch.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]  # (3,)

def cheirality_count(R, t, bu1, bu2):
    P1 = torch.cat([torch.eye(3, device=R.device), torch.zeros(3,1, device=R.device)], dim=1)
    P2 = torch.cat([R, t.reshape(3,1)], dim=1)
    cnt = 0
    for i in range(bu1.shape[0]):
        X = triangulate_point(P1, P2, bu1[i], bu2[i])
        z1 = X[2]
        z2 = (R @ X + t)[2]
        if (z1 > 0) and (z2 > 0):
            cnt += 1
    return cnt

# ✅ FIX 1: N should be number of points
N = u1.shape[0]

# homogeneous (N,3)
bu1 = torch.hstack([u1, torch.ones(N, 1, dtype=u1.dtype)])
bu2 = torch.hstack([u2, torch.ones(N, 1, dtype=u2.dtype)])

w = torch.zeros(3, dtype=torch.float32, requires_grad=True)
t = torch.randn(3, dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([w, t], lr=0.1)

eps = 1e-9

for step in range(2000):
    optimizer.zero_grad()
    R = rodrigues(w)
    t_unit = t / torch.norm(t)
    E = skew(t_unit) @ R
    err = (bu2 @ E * bu1).sum(dim=1)
    loss = (err**2).mean()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, float(loss))

R = rodrigues(w).detach()
t = t / torch.norm(t).detach()

c_pos = cheirality_count(R, t, bu1, bu2)
c_neg = cheirality_count(R, -t, bu1, bu2)
if c_neg > c_pos:
    t = -t

E_hat = (skew(t) @ R).detach()
errs = torch.tensor([bu2[i] @ E_hat @ bu1[i] for i in range(N)])
print("E=\n", E_hat)
print("\nR=\n", R.T)
print("det(R)=", det(R))
print("t (up to scale)=\n", -R.T @ t)
print("\nRMSE(x2^T E x1)=", torch.sqrt(torch.mean(errs**2)))
print("max|x2^T E x1|=", torch.max(torch.abs(errs)))
