# Euler-Lagrange Dynamics Equations
from .frames import *
from itertools import product
from omegaconf import OmegaConf
from Pathes import P as Pathes

J_tip = p_tip.jacobian(q)
# Jacobians and inertia
D = sp.zeros(N, N)
K_links = []
P_links = []
for k in range(N):
    Jv = sp.zeros(3, N)  # Jacobian of p_CoM w.r.t. velocity
    Jw = sp.zeros(3, N)  # Jacobian of p_CoM w.r.t. angular velocity
    # all joints are revolute, there is a clear formula for the Jacobians
    for j in range(k + 1):  # only the joints up to the current one will affect p_CoM
        Jv[:, j] = axes[j].cross(p_CoM[k] - origins[j])
        Jw[:, j] = axes[j]
    I_link = R_inertial[k] * I_body * R_inertial[k].T
    I_world = rotations[k] * I_link * rotations[k].T
    D_link = m * Jv.T * Jv + Jw.T * I_world * Jw
    D += D_link
    K_links.append(sp.Rational(1, 2) * (dq.T * D_link * dq)[0])
    P_links.append(m * g * p_CoM[k][2])

K = sum(K_links)
P = sum(P_links)
if is_symbolic:
    print("K=", K)
    print("P=", P)
cfgR = OmegaConf.load(Pathes.cfg_dir / "robot.yaml")
if cfgR.payload == True:
    m_payload = sp.Integer(3)
    P += m_payload * g * p_tip[2]

G = sp.Matrix([sp.diff(P, qi) for qi in q])
C = sp.zeros(N, 1)
for k in range(N):
    for i, j in product(range(N), range(N)):
        christoffel = sp.Rational(1, 2) * (sp.diff(D[k, j], q[i]) + sp.diff(D[k, i], q[j]) - sp.diff(D[i, j], q[k]))
        C[k] += christoffel * dq[i] * dq[j]

if is_symbolic:
    print("D=", D)
    print("G=", G)
    print("C=", C)