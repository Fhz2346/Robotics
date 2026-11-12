import numpy as np
from robot.sympy import *

states = [
    (sp.pi / 3, sp.pi / 2, sp.pi / 3, 0, 0, 0),
    (sp.pi / 3, sp.pi / 2, sp.pi / 3, sp.pi / 4, sp.pi / 4, sp.pi / 2),
    (0, sp.pi / 2, 0, sp.pi / 4, sp.pi / 4, sp.pi / 2),
]
if __name__ == "__main__":
    for idx, state in enumerate(states, 1):
        subs = dict(zip(list(q) + list(dq), state))
        D_num = D.subs(subs).evalf(8)
        C_num = C.subs(subs).evalf(8)
        G_num = G.subs(subs).evalf(8)
        print(f"\n========== STATE {idx} ==========")
        print("D(q) =")
        sp.pprint(D_num)
        print("C(q, qdot) =")
        sp.pprint(C_num)
        print("G(q) =")
        sp.pprint(G_num)
        assert np.allclose(np.array(D_num, dtype=float), np.array(D_num.T, dtype=float), atol=1e-9) # Symmetry check
        eig = np.linalg.eigvalsh(np.array(D_num, dtype=float))
        assert np.all(eig > 0) # Positive definiteness
        print("D symmetric: PASS")
        print("D positive definite: PASS")
    print("\n========== ALL CHECKS PASSED ==========")
