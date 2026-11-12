import sympy as sp


def Rx(a):
    return sp.Matrix(
        [
            [1, 0, 0],
            [0, sp.cos(a), -sp.sin(a)],
            [0, sp.sin(a), sp.cos(a)],
        ]
    )


def Ry(a):
    return sp.Matrix(
        [
            [sp.cos(a), 0, sp.sin(a)],
            [0, 1, 0],
            [-sp.sin(a), 0, sp.cos(a)],
        ]
    )


def Rz(a):
    return sp.Matrix(
        [
            [sp.cos(a), -sp.sin(a), 0],
            [sp.sin(a), sp.cos(a), 0],
            [0, 0, 1],
        ]
    )
