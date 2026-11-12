import numpy as np
import mujoco

from .base import BaseRobotInterface
from .state import StateMixin
from .actuation import ActuationMixin
from .kinematics import KinematicsMixin
from .contacts import ContactMixin
from .control import ControlMixin


class RobotInterface(
    BaseRobotInterface,
    StateMixin,
    ActuationMixin,
    KinematicsMixin,
    ContactMixin,
    ControlMixin,
):
    def get_rfoot_body_pos(self):
        if isinstance(self.rfoot_body_name, list):
            return [self.data.body(i).xpos.copy() for i in self.rfoot_body_name]
        return self.data.body(self.rfoot_body_name).xpos.copy()

    def get_lfoot_body_pos(self):
        if isinstance(self.lfoot_body_name, list):
            return [self.data.body(i).xpos.copy() for i in self.lfoot_body_name]
        return self.data.body(self.lfoot_body_name).xpos.copy()

    def get_rfoot_floor_contacts(self):
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        rcontacts = []

        rfeet = [self.rfoot_body_name] if isinstance(self.rfoot_body_name, str) else self.rfoot_body_name
        rfeet_ids = [self.model.body(bn).id for bn in rfeet]

        for i, c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name != self.robot_root_name)
            geom2_is_rfoot = (self.model.geom_bodyid[c.geom2] in rfeet_ids)

            if geom1_is_floor and geom2_is_rfoot:
                rcontacts.append((i, c))
        return rcontacts

    def get_lfoot_floor_contacts(self):
        # contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        lcontacts = []
        lfoot_id = self.model.body(self.lfoot_body_name).id
        for i, c in enumerate(self.data.contact):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name != self.robot_root_name)
            geom2_is_lfoot = (self.model.geom_bodyid[c.geom2] == lfoot_id)

            if geom1_is_floor and geom2_is_lfoot:
                lcontacts.append((i, c))
        return lcontacts

    def get_rfoot_grf(self):
        right_contacts = self.get_rfoot_floor_contacts()
        rfoot_grf = 0.0

        for i, _ in right_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            rfoot_grf += np.linalg.norm(c_array)

        return rfoot_grf

    def get_lfoot_grf(self):
        left_contacts = self.get_lfoot_floor_contacts()
        lfoot_grf = 0.0
        for i, _ in left_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            lfoot_grf += np.linalg.norm(c_array)
        return lfoot_grf

    def get_rfoot_body_vel(self, frame=0):
        if isinstance(self.rfoot_body_name, list):
            return [self.get_body_vel(i, frame=frame) for i in self.rfoot_body_name]
        return self.get_body_vel(self.rfoot_body_name, frame=frame)

    def get_lfoot_body_vel(self, frame=0):
        if isinstance(self.lfoot_body_name, list):
            return [self.get_body_vel(i, frame=frame) for i in self.lfoot_body_name]
        return self.get_body_vel(self.lfoot_body_name, frame=frame)

    def check_rfoot_floor_collision(self):
        return len(self.get_rfoot_floor_contacts()) > 0

    def check_lfoot_floor_collision(self):
        return len(self.get_lfoot_floor_contacts()) > 0
