from .euler_angle import *
from .basics import *

# Set up Link Coordinates
## Link orientations in world frame, match with URDF settings, read in Intrinsic Axes
R1 = Rz(q1)
R2 = R1 * Rx(sp.pi/2) * Rz(q2)
R3 = R2 * Rz(q3)
rotations = [R1, R2, R3]
## Joint origins in world frame
o1 = sp.zeros(3, 1)
o2 = o1 + R1 * sp.Matrix([0, 0, l])
o3 = o2 + R2 * sp.Matrix([l, 0, 0])
origins = [o1, o2, o3]
p_tip = o3 + R3 * sp.Matrix([l, 0, 0])
## rotation axes in world frame
ez = sp.Matrix([0, 0, 1]) # z-axis in world frame
z1 = R1 * ez
z2 = R2 * ez
z3 = R3 * ez
axes = [z1, z2, z3]

# Inertial frame
p_CoM = [
    o1 + R1 * sp.Matrix([0, 0, l / 2]),
    o2 + R2 * sp.Matrix([l / 2, 0, 0]),
    o3 + R3 * sp.Matrix([l / 2, 0, 0]),
]

R_inertial = [sp.eye(3), Ry(-sp.pi / 2), Ry(sp.pi / 2)] # Inertial frame orientation relative to link
