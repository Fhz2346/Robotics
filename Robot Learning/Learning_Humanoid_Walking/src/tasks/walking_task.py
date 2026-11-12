import numpy as np

from robots.robot_interface import RobotInterface
import tasks.rewards as R
from .base import BaseTask
from .phase import create_phase


class WalkingTask(BaseTask):
    """Dynamically stable walking on biped."""

    def __init__(self, client: RobotInterface, control_dt, cfg):
        super().__init__(client)
        self._control_dt = control_dt
        self.Rclock, self.Lclock = create_phase(
            cfg.duration.swing,
            cfg.duration.stance,
            0.1,
            "grounded",
            1 / self._control_dt,
        )
        # These depend on the robot, hardcoded for now
        # Ideally, they should be arguments to __init__
        self._goal_height_ref = cfg.goal_height_ref
        self._total_duration = cfg.duration.swing + cfg.duration.stance

        self._root_body_name = client.robot_root_name
        self._lfoot_body_name = client.rfoot_body_name
        self._rfoot_body_name = client.lfoot_body_name

    def calc_reward(self, prev_torque, prev_action, action):
        self.l_foot_vel = self._client.get_lfoot_body_vel()[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel()[0]
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()
        r_frc, r_vel = self.Rclock
        l_frc, l_vel = self.Lclock
        reward = dict(
            foot_frc_score=0.150 * R.foot_frc_clock(self, l_frc, r_frc),
            foot_vel_score=0.150 * R.foot_vel_clock(self, l_vel, r_vel),
            orient_cost=0.050
            * (
                self.body_orient(self._lfoot_body_name)
                + self.body_orient(self._rfoot_body_name)
                + self.body_orient(self._root_body_name)
            )
            / 3,
            root_accel=0.050 * self.root_accel(),
            height_error=0.050 * R.height(self),
            com_vel_error=0.200 * R.fwd_vel(self),
            torque_penalty=0.050 * self.torque(prev_torque),
            action_penalty=0.050 * self.action(action, prev_action),
        )
        return reward

    def step(self):
        if self._phase > self._period:
            self._phase = 0
        self._phase += 1
        return

    def done(self):
        contact_flag = self._client.check_self_collisions()
        qpos = self._client.get_qpos()
        terminate_conditions = {
            "qpos[2]_ll": (qpos[2] < 0.6),
            "qpos[2]_ul": (qpos[2] > 1.4),
            "contact_flag": contact_flag,
        }
        done = True in terminate_conditions.values()
        return done

    def reset(self):
        self._goal_speed_ref = np.random.choice([0, np.random.uniform(0.3, 0.4)])
        # number of control steps in one full cycle
        # (one full cycle includes left swing + right swing)
        self._period = np.floor(2 * self._total_duration * (1 / self._control_dt))
        # randomize phase during initialization
        self._phase = np.random.randint(0, self._period)
