from robot.pybullet import *


def get_gravity_torque(q):
    return fd(get_potential_energy, q)
