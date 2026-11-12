from robot.pybullet import *


payload_mass = 3.0
def get_payload_potential_energy():
    return payload_mass * G * get_p_tip()[2]


def get_potential_energy_with_payload():
    return get_potential_energy() + get_payload_potential_energy()


def get_gravity_torque(q):
    return fd(get_potential_energy_with_payload, q, eps=1.0e-3)
