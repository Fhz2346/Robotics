import pybullet as p
import numpy as np


class Object:

    def __init__(self, cfg):
        self.length = cfg.length
        self.width = cfg.width
        self.center = np.array(cfg.center)
        self.is_moving = cfg.is_moving
        if self.is_moving:
            self.iteration = 0
            self.center = self.get_center()
        self.create_pybullet_object()

    def create_pybullet_object(self):
        half_extents = [self.length / 2, self.width / 2, self.width / 2]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.8, 0, 0, 1])
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=self.center,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )

    def get_center(self):
        return np.array(
            [
                0.5 + 0.2 * np.sin(np.pi / 2 + self.iteration / 10),
                0.5 + 0.2 * np.sin(self.iteration / 20),
                0.01,
            ]
        )

    def step(self):
        if not self.is_moving:
            return
        self.iteration += 1
        self.center = self.get_center()
        p.resetBasePositionAndOrientation(1, self.center.tolist(), p.getQuaternionFromEuler([0, 0, 0]))
