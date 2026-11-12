import numpy as np
import mujoco


class ContactMixin:
    def get_body_floor_contacts(self, body_name):
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        body_contacts = []

        body_names = [body_name] if isinstance(body_name, str) else body_name
        body_ids = [self.model.body(bn).id for bn in body_names]

        for i, c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name != self.robot_root_name)
            geom2_is_body = (self.model.geom_bodyid[c.geom2] in body_ids)

            if geom1_is_floor and geom2_is_body:
                body_contacts.append((i, c))
        return body_contacts

    def get_body_contact_force(self, body):
        if isinstance(body, str):
            body = [body]

        frc = 0.0
        for i, con in enumerate(self.data.contact):
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)

            b1 = self.model.body(self.model.geom(con.geom1).bodyid)
            b2 = self.model.body(self.model.geom(con.geom2).bodyid)

            if b1.name in body or b2.name in body:
                frc += np.linalg.norm(c_array)

        return frc

    def get_interaction_force(self, body1, body2):
        frc = 0.0
        for i, con in enumerate(self.data.contact):
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)

            b1 = self.model.body(self.model.geom(con.geom1).bodyid)
            b2 = self.model.body(self.model.geom(con.geom2).bodyid)

            if (b1.name == body1 and b2.name == body2) or (b1.name == body2 and b2.name == body1):
                frc += np.linalg.norm(c_array)

        return frc

    def check_bad_collisions(self, body_names=None):
        if body_names is None:
            body_names = []

        num_cons = 0
        if not isinstance(body_names, list):
            raise TypeError(f"expected list of body names, got '{type(body_names).__name__}'")

        if not len(body_names):
            num_rcons = len(self.get_rfoot_floor_contacts())
            num_lcons = len(self.get_lfoot_floor_contacts())
            num_cons = num_rcons + num_lcons

        for bn in body_names:
            num_cons += len(self.get_body_floor_contacts(bn))

        return num_cons != self.data.ncon

    def check_self_collisions(self):
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]

        for c in contacts:
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])

            geom1_is_robot = self.model.body(geom1_body.rootid).name == self.robot_root_name
            geom2_is_robot = self.model.body(geom2_body.rootid).name == self.robot_root_name

            if geom1_is_robot and geom2_is_robot:
                return True
        return False