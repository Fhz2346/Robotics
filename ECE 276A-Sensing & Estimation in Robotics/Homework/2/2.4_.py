import numpy as np
from numpy.linalg import svd, det


def skew(v):
    vx, vy, vz = v[0], v[1], v[2]
    v_hat = np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=v.dtype,
    )
    return v_hat


u1 = np.array(
    [
        [0.2953, 0.1658, 0.3040, 0.1492, 0.1247, 0.4638, 0.6996, 0.1004],
        [-0.3088, -0.5017, 0.1686, -0.0960, -0.0581, 0.0349, 0.6246, -0.5315],
    ],
    dtype=float,
).T  # (N,2)

u2 = np.array(
    [
        [0.2909, 0.1647, 1.8975, 1.2081, 1.2510, 1.1337, 2.5496, 0.1398],
        [0.5712, 1.0140, 0.3364, 1.4632, 1.3761, 0.5984, 0.1150, 1.1851],
    ],
    dtype=float,
).T  # (N,2)

N = u1.shape[0]
bu1 = np.hstack([u1, np.ones((N, 1))])  # (N,3)
bu2 = np.hstack([u2, np.ones((N, 1))])  # (N,3)

Y = np.stack(
    [np.outer(bu1[i], bu2[i]).reshape(-1, order="F") for i in range(N)], axis=0
)
_, _, Vt = svd(Y)
E_hat = Vt[-1].reshape(3, 3)

U, S, Vt = svd(E_hat)
if det(U) < 0:
    U[:, -1] *= -1
if det(Vt) < 0:
    Vt[-1, :] *= -1
s = 0.5 * (S[0] + S[1])
E = U @ np.diag([s, s, 0.0]) @ Vt

U, _, Vt = svd(E)
if det(U) < 0:
    U[:, -1] *= -1
if det(Vt) < 0:
    Vt[-1, :] *= -1

W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

R1 = U @ W @ Vt
R2 = U @ W.T @ Vt
t = U[:, 2]
if det(R1) < 0:
    R1 *= -1
if det(R2) < 0:
    R2 *= -1

P1 = np.hstack([np.eye(3), np.zeros((3, 1))])


def triangulate(R, t, x1i, x2i):
    P2 = np.hstack([R, t.reshape(3, 1)])
    A = np.array(
        [
            x1i[0] * P1[2] - P1[0],
            x1i[1] * P1[2] - P1[1],
            x2i[0] * P2[2] - P2[0],
            x2i[1] * P2[2] - P2[1],
        ]
    )
    _, _, VtA = svd(A)
    X = VtA[-1]
    X /= X[3]
    return X[:3]


def cheirality_count(R, t):
    c = 0
    for i in range(N):
        X = triangulate(R, t, bu1[i], bu2[i])
        if X[2] > 0 and (R @ X + t)[2] > 0:
            c += 1
    return c


cands = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
c = [cheirality_count(R, t) for R, t in cands]
R, t = cands[np.argmax(c)]

errs = np.array([bu2[i] @ E @ bu1[i] for i in range(N)])

print("E=\n", E_hat)
print("\nR=\n", R.T)
print("det(R)=", det(R))
print("t (up to scale)=\n", -R.T @ t)
print("\nRMSE(x1^T E x2)=", np.sqrt(np.mean(errs**2)))
print("max|x1^T E x2|=", np.max(np.abs(errs)))
