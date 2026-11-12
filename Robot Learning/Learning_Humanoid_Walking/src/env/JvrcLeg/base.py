import numpy as np
from numpy.linalg import norm
import mujoco
from omegaconf import OmegaConf
import transforms3d as tf3
from envs.mujoco_env import MujocoEnv
from Pathes import P
from dataclasses import dataclass


@dataclass
class Foot:
    # Static
    name: str
    id: int = None
    # Dynamic
    floor_contacts = []
    vel: float = 0.0  # foot velocity
    grf: float = 0.0  # Ground Reaction Force

def get_pose(cfg):
    return np.concatenate([cfg.base.pos, cfg.base.quat, np.deg2rad(cfg.angles)])

class JvrcLegBase(MujocoEnv):

    def __init__(self, cfg):
        path_to_xml = P.tmp_dir / "mjcf-export/jvrc_walk/jvrc.xml"
        super().__init__(path_to_xml, cfg.dt.sim)
        _cfg = OmegaConf.load(P.config_dir / "jvrc" / "leg.yaml")
        cfg.robot = _cfg
        # parameters
        cfg.info.dA = len(_cfg.leg_joints)
        self.nominal_pose = get_pose(_cfg.nominal_pose)
        self.root_name = self.model.body(1).name
        self.gear_ratios = self.model.actuator_gear[:, 0]
        self.init_feet(cfg.robot.foot_names)

    def init_feet(self, foot_names):
        self.Lfoot = Foot(name=foot_names[0])
        self.Rfoot = Foot(name=foot_names[1])
        self.Lfoot.id = self.model.body(self.Lfoot.name).id
        self.Rfoot.id = self.model.body(self.Rfoot.name).id

    def get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        root_rp = tf3.euler.quat2euler(qpos[3:7])[0:2] # roll, pitch
        root_ang_vel = qvel[3:6]
        motor_pos = self.get_act_joint_positions()
        motor_vel = self.get_act_joint_velocities()
        state = np.concatenate([root_rp, root_ang_vel, motor_pos, motor_vel])
        return state

    def reset(self):
        self.set_state(self.nominal_pose, np.zeros(self.model.nv))
    
    # Utility functions
    def get_act_joint_velocities(self):
        motor_velocities = self.data.actuator_velocity.copy()
        return motor_velocities / self.gear_ratios

    def get_act_joint_positions(self):
        motor_positions = self.data.actuator_length.copy()
        return motor_positions / self.gear_ratios
    
    def get_act_joint_torques(self):
        motor_torques = self.data.actuator_force
        return motor_torques * self.gear_ratios
    
    ## Feet functions
    def _update_feet_velocities(self):
        self.Lfoot.vel = norm(self.get_body_vel(self.Lfoot.name)[0])
        self.Rfoot.vel = norm(self.get_body_vel(self.Rfoot.name)[0])
    
    def _get_floor_contacts(self):
        floor_contacts = []
        for i, c in enumerate(self.data.contact):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name != self.root_name)
            if geom1_is_floor:
                floor_contacts.append((i, c))
        return floor_contacts

    def _update_feet_floor_contacts(self):
        floor_contacts = self._get_floor_contacts()
        self.Lfoot.floor_contacts = []
        self.Rfoot.floor_contacts = []
        for i, c in floor_contacts:
            if self.model.geom_bodyid[c.geom2] == self.Lfoot.id:
                self.Lfoot.floor_contacts.append((i, c))
            elif self.model.geom_bodyid[c.geom2] == self.Rfoot.id:
                self.Rfoot.floor_contacts.append((i, c))

    def _get_sum_contact_force(self, contacts):
        force = 0.0
        for i, c in contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            force += np.linalg.norm(c_array)
        return force

    def _update_feet_grf(self):
        self._update_feet_floor_contacts()
        self.Lfoot.grf = self._get_sum_contact_force(self.Lfoot.floor_contacts)
        self.Rfoot.grf = self._get_sum_contact_force(self.Rfoot.floor_contacts)

    def _update_feet(self):
        self._update_feet_velocities()
        self._update_feet_grf()
    
    def check_self_collisions(self):
        for c in self.data.contact:
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_robot = self.model.body(geom1_body.rootid).name == self.root_name
            geom2_is_robot = self.model.body(geom2_body.rootid).name == self.root_name
            if geom1_is_robot and geom2_is_robot:
                return True
        return False

    @property
    def mass(self):
        return mujoco.mj_getTotalmass(self.model)
