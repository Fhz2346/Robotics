import numpy as np
import pybullet as p
import pybullet_data
from Object import Object


def init_environment(cfg):
    """Create the PyBullet scene without opening its GUI; return target center."""
    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Could not connect to the PyBullet physics server")

    p.resetSimulation()
    p.setTimeStep(0.001)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    obj = Object(cfg)
    return obj
