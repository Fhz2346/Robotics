import numpy as np


def rz(a):
    return np.array(
        [
            [np.cos(a), -np.sin(a), 0.0],
            [np.sin(a), np.cos(a), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rx(a):
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(a), -np.sin(a)],
            [0.0, np.sin(a), np.cos(a)],
        ]
    )
