import torch
from torch.linalg import norm
from torch.autograd.functional import jacobian

torch.set_default_dtype(torch.float64)


def f(q: torch.Tensor):
    return torch.log(norm(q))


def grad_f(q: torch.Tensor):
    n2 = (q.detach() @ q.detach())
    g_fq = q.detach() / n2
    return g_fq


def g(q: torch.Tensor):
    qv = q[1:]
    return qv / norm(qv)


def grad_g(q: torch.Tensor):
    qv = q.detach()[1:]
    nv = norm(qv)
    I3 = torch.eye(3, dtype=torch.float64)
    Jg_qv = (1.0/nv) * (I3 - torch.ger(qv, qv) / (qv @ qv))
    Jg_q = torch.cat([torch.zeros(3,1, dtype=torch.float64), Jg_qv], dim=1)
    return Jg_q


def h(q: torch.Tensor):
    qs = q[0]
    return torch.arccos(qs / norm(q))


def grad_h(q: torch.Tensor):
    qv = q.detach()[1:]
    nv = norm(qv)
    e1 = torch.tensor([1.,0.,0.,0.], dtype=torch.float64)
    n2 = (q.detach() @ q.detach())
    g_hq = (1.0/nv) * ((q.detach()[0] / n2) * q.detach() - e1)
    return g_hq


q = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
# numerical
Jf = jacobian(f, q)
Jg = jacobian(g, q)
Jh = jacobian(h, q)
# analytical
g_fq = grad_f(q)
Jg_q = grad_g(q)
g_hq = grad_h(q)


# compare
print("numerical f", Jf)
print("analytical f", g_fq)
print("allclose f:", torch.allclose(Jf, g_fq, atol=1e-9, rtol=1e-9))

print("numerical g", Jg)
print("analytical g:\n", Jg_q)
print("allclose g:", torch.allclose(Jg, Jg_q, atol=1e-9, rtol=1e-9))

print("numerical h", Jh)
print("analytical h:", g_hq)
print("allclose h:", torch.allclose(Jh, g_hq, atol=1e-9, rtol=1e-9))

