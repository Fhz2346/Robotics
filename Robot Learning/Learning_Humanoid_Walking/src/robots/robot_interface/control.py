import numpy as np
import mujoco


class ControlMixin:

    def set_pd_gains(self, kp, kv):
        assert kp.size == self.model.nu
        assert kv.size == self.model.nu
        self.kp = kp.copy()
        self.kv = kv.copy()

    def step_pd(self, p, v):
        p_hat = self.get_act_joint_positions()
        v_hat = self.get_act_joint_velocities()
        p_err = p - p_hat
        v_err = v - v_hat
        torque = self.kp * p_err + self.kv * v_err
        return torque

    def set_motor_torque(self, torque):
        np.copyto(self.data.ctrl, torque)

    def step(self, n_step=1):
        mujoco.mj_step(self.model, self.data, n_step)
        self.stepCounter += n_step
