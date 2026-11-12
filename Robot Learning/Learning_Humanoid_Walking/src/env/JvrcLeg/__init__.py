import numpy as np
import mujoco
from .base import JvrcLegBase

class JvrcLeg(JvrcLegBase): # with PD control

    def __init__(self, cfg):
        super().__init__(cfg)
        self.kp = np.array(cfg.robot.ctrl.kp)
        self.kd = np.array(cfg.robot.ctrl.kd)
        self.tau_d = np.zeros(self.model.nu)
        self.frame_skip = int(cfg.dt.ctrl / cfg.dt.sim)

    def torque_via_PD(self, p, v):
        p_err = p - self.get_act_joint_positions()
        v_err = v - self.get_act_joint_velocities()
        torque = self.kp * p_err + self.kd * v_err
        return torque

    def step(self, joint_angles):
        for _ in range(self.frame_skip):
            w = self.get_act_joint_velocities()
            tau = self.torque_via_PD(joint_angles, np.zeros(self.model.nu))
            tau = tau - self.tau_d * w
            tau /= self.gear_ratios
            np.copyto(self.data.ctrl, tau)
            mujoco.mj_step(self.model, self.data)
        self._update_feet()
