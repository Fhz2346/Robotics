from robot.pybullet import *


def jacobian_and_gravity(q):
    J = fd(get_p_tip, q)
    grav = fd(get_potential_energy, q)
    return J, grav
