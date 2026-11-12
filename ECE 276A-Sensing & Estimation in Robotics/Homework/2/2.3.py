import torch
from Utils import rodrigues


Z = torch.tensor(
    [
        [0.5531, 0.3944, 0.5509, 0.4952, 0.3412, 0.4405, 0.9896, 0.8160],
        [0.4630, 0.3683, 0.8316, 0.8083, 0.8422, 0.6731, 0.2983, 0.2219],
        [0.6784, 0.0812, 0.9339, 0.6688, 0.2196, 0.4302, 0.3114, 0.6057],
    ],
    dtype=torch.float32,
)
M = torch.tensor(
    [
        [-1.2285, -1.0476, -1.5808, -1.5048, -1.4643, -1.3583, -0.7150, -0.8713],
        [-0.5738, -0.5268, -0.4048, -0.3617, -0.2732, -0.4064, -0.2932, -0.5575],
        [0.8641, 0.2675, 1.0842, 0.8268, 0.3608, 0.5928, 0.7645, 0.9325],
    ],
    dtype=torch.float32,
)

w = torch.zeros(3, requires_grad=True)
p = torch.zeros(3, requires_grad=True)
optimizer = torch.optim.SGD([w, p], lr=0.1)
for step in range(3000):
    optimizer.zero_grad()

    R = rodrigues(w)
    pred = R @ Z + p.reshape(3, 1)

    loss = torch.mean((pred - M) ** 2)
    loss.backward()
    optimizer.step()

R = rodrigues(w).detach()
p = p.detach()

print("\nEstimated R =\n", R)
print("det(R) =", torch.linalg.det(R))
print("\nEstimated p =", p)
M_hat = R @ Z + p.reshape(3, 1)
err = M_hat - M
print("RMSE =", torch.sqrt(torch.mean(err**2)).item())

