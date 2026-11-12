import numpy as np

Z = np.array([
    [0.5531, 0.3944, 0.5509, 0.4952, 0.3412, 0.4405, 0.9896, 0.8160],
    [0.4630, 0.3683, 0.8316, 0.8083, 0.8422, 0.6731, 0.2983, 0.2219],
    [0.6784, 0.0812, 0.9339, 0.6688, 0.2196, 0.4302, 0.3114, 0.6057],
], dtype=float)

M = np.array([
    [-1.2285, -1.0476, -1.5808, -1.5048, -1.4643, -1.3583, -0.7150, -0.8713],
    [-0.5738, -0.5268, -0.4048, -0.3617, -0.2732, -0.4064, -0.2932, -0.5575],
    [ 0.8641,  0.2675,  1.0842,  0.8268,  0.3608,  0.5928,  0.7645,  0.9325],
], dtype=float)

w = np.ones(Z.shape[1])
w = w / np.sum(w)

z_bar = Z @ w
m_bar = M @ w

dZ = Z - z_bar[:, None]
dM = M - m_bar[:, None]

Q = (dM * w[None, :]) @ dZ.T

U, S, Vt = np.linalg.svd(Q)
R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt

p = m_bar - R @ z_bar

print("z_bar =\n", z_bar)
print("m_bar =\n", m_bar)
print("Q =\n", Q)
print("R =\n", R)
print("det(R) =", np.linalg.det(R))
print("p =\n", p)

M_hat = R @ Z + p[:, None]
err = M_hat - M
print("RMSE =", np.sqrt(np.mean(err**2)))
