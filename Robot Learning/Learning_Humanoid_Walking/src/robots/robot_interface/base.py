import mujoco


class BaseRobotInterface:

    def __init__(self, model, data, foots_name):
        self.model = model
        self.data = data
        self.rfoot_body_name, self.lfoot_body_name = foots_name
        self.floor_body_name = model.body(0).name
        self.robot_root_name = model.body(1).name
        self.stepCounter = 0

    @property
    def nq(self):
        return self.model.nq

    @property
    def nu(self):
        return self.model.nu

    @property
    def nv(self):
        return self.model.nv

    @property
    def sim_dt(self):
        return self.model.opt.timestep

    def get_robot_mass(self):
        return mujoco.mj_getTotalmass(self.model)

    def get_jnt_id_by_name(self, name):
        return self.model.joint(name)

    def get_jnt_qposadr_by_name(self, name):
        return self.model.joint(name).qposadr

    def get_jnt_qveladr_by_name(self, name):
        return self.model.joint(name).dofadr
