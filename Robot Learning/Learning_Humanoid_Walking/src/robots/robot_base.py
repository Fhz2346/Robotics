import numpy as np
from .robot_interface.RobotInterface import RobotInterface


class RobotBase(object): # 输入目标角度, 应用PD控制器, 输出扭矩

    def __init__(self, client: RobotInterface, task, pdgains, control_dt, cfg):
        self.client = client
        self.task = task
        self.control_dt = control_dt
        self.pdrand_k = cfg.pdrand_k

        # set PD gains
        self.kp = pdgains[0]
        self.kd = pdgains[1]
        assert self.kp.shape==self.kd.shape==(self.client.nu,)

        # torque damping param
        self.tau_d = np.zeros(self.client.nu)

        self.client.set_pd_gains(self.kp, self.kd)
        tau = self.client.step_pd(np.zeros(self.client.nu), np.zeros(self.client.nu))
        w = self.client.get_act_joint_velocities()
        assert len(w)==len(tau)

        self.prev_action = None
        self.prev_torque = None
        self.iteration_count = np.inf

        # frame skip parameter
        if np.around(self.control_dt % self.client.sim_dt, 6):
            raise Exception("Control dt should be an integer multiple of Simulation dt.")
        self.frame_skip = int(self.control_dt / self.client.sim_dt)
        
    def _do_simulation(self, target, n_frames):
        ratio = self.client.get_gear_ratios()
        for _ in range(n_frames):
            w = self.client.get_act_joint_velocities()
            tau = self.client.step_pd(target, np.zeros(self.client.nu))
            tau = tau - self.tau_d * w
            tau /= ratio
            self.client.set_motor_torque(tau)
            self.client.step()

    def step(self, action, offset=None):
        # If offset is provided, add to action vector
        if offset is not None:
            action += offset

        if self.prev_action is None:
            self.prev_action = action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.client.get_act_joint_torques())

        # Perform the simulation
        self._do_simulation(action, self.frame_skip)

        # Task-related operations
        self.task.step()
        rewards = self.task.calc_reward(self.prev_torque, self.prev_action, action)
        done = self.task.done()

        self.prev_action = action
        self.prev_torque = np.asarray(self.client.get_act_joint_torques())

        return rewards, done
