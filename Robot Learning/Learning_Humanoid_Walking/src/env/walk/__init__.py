import numpy as np
import collections

from envs.Utils import ActionSmoother
from ..JvrcLeg import JvrcLeg
from .phase import PhaseClock
from .base import WalkingBase


class Walking(WalkingBase): # with PhaseClock

    def __init__(self, cfg, robot: JvrcLeg):
        super().__init__(cfg, robot)
        self.clock = PhaseClock(cfg.phase, 1 / cfg.dt.ctrl)
        self.cfg.desired_max_foot_frc = self.robot.mass * 9.8 * 0.5
        self.observation_history = collections.deque(maxlen=self.cfg.len_obs_history)
        self.cfg.info.dS = self.get_obs().shape[0]
        self.set_obs_statistics()

    def merge_state_in_observation_history(self, state):
        if len(self.observation_history) == 0:
            for _ in range(self.cfg.len_obs_history):
                self.observation_history.appendleft(np.zeros_like(state))
        self.observation_history.appendleft(state)
        return np.array(self.observation_history).flatten()

    def get_obs(self):
        clock = [np.sin(self.clock.theta), np.cos(self.clock.theta)]
        ext_state = np.concatenate((clock, [self.cfg.goal_speed_ref])) 
        robot_state = self.robot.get_obs()
        state = np.concatenate([robot_state, ext_state])
        return self.merge_state_in_observation_history(state)

    def step(self, action):
        if self.prev_action is None:
            self.prev_action = action
        if self.prev_torque is None:
            self.prev_torque = self.robot.get_act_joint_torques()
        
        self.robot.step(action)
        rewards = self.calc_reward(self.prev_torque, self.prev_action, action)
        done = self.done()
        obs = self.get_obs()
        
        self.prev_action = action
        self.prev_torque = self.robot.get_act_joint_torques()
        return obs, sum(rewards.values()), done, rewards

    def reset(self):
        self.prev_torque = None
        self.prev_action = None
        self.cfg.goal_speed_ref = np.random.choice([0, np.random.uniform(0.3, 0.4)]).item()
        self.clock.reset()
        self.robot.reset()
        self.observation_history = collections.deque(maxlen=self.cfg.len_obs_history)
        return self.get_obs()

    def calc_reward(self, prev_torque, prev_action, action):
        frcLclock, velLclock, frcRclock, velRclock = self.clock()
        reward = dict(
            foot_frc_score=0.150 * self.get_feet_frc_score(frcLclock, frcRclock),
            foot_vel_score=0.150 * self.get_feet_vel_score(velLclock, velRclock),
        )
        reward.update(super().calc_reward(prev_torque, prev_action, action))
        return reward

    def set_obs_statistics(self):
        self.obs_mean = np.concatenate(
            (
                np.zeros(5),
                np.deg2rad(self.robot.nominal_pose[7:]),
                np.zeros(12),
                [0.5, 0.5, 0.5],
            )
        )
        self.obs_std = np.concatenate(
            (
                [0.2, 0.2, 1, 1, 1],
                0.5 * np.ones(12),
                4 * np.ones(12),
                [1, 1, 1],
            )
        )
        self.obs_mean = np.tile(self.obs_mean, self.cfg.len_obs_history)
        self.obs_std = np.tile(self.obs_std, self.cfg.len_obs_history)

    # Reward Functions
    def get_feet_frc_score(self, frcLclock, frcRclock):
        desired_max_foot_frc = self.cfg.desired_max_foot_frc
        normedLfrc = min(self.robot.Lfoot.grf, desired_max_foot_frc) / desired_max_foot_frc
        normedRfrc = min(self.robot.Rfoot.grf, desired_max_foot_frc) / desired_max_foot_frc
        normedLfrc = 2 * normedLfrc - 1
        normedRfrc = 2 * normedRfrc - 1
        frcLscore = np.tan(np.pi / 4 * frcLclock * normedLfrc)
        frcRscore = np.tan(np.pi / 4 * frcRclock * normedRfrc)
        frc_score = (frcLscore + frcRscore) / 2
        return frc_score

    def get_feet_vel_score(self, velLclock, velRclock):
        desired_max_foot_vel = self.cfg.desired_max_foot_vel
        normedLvel = min(self.robot.Lfoot.vel, desired_max_foot_vel) / desired_max_foot_vel
        normedRvel = min(self.robot.Rfoot.vel, desired_max_foot_vel) / desired_max_foot_vel
        normedLvel = 2 * normedLvel - 1
        normedRvel = 2 * normedRvel - 1
        velLscore = np.tan(np.pi / 4 * velLclock * normedLvel)
        velRscore = np.tan(np.pi / 4 * velRclock * normedRvel)
        vel_score = (velLscore + velRscore) / 2
        return vel_score
