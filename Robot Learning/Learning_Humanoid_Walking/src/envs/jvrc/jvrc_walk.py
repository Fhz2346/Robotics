import os
import numpy as np
import transforms3d as tf3
import collections
from omegaconf import OmegaConf

from tasks.walking_task import WalkingTask
from robots.robot_base import RobotBase
from envs.mujoco_env import MujocoEnv
from robots.robot_interface import RobotInterface
from Pathes import P
from envs.Utils import ActionSmoother

def get_pose(cfg):
    return np.concatenate([cfg.base.pos, cfg.base.quat, np.deg2rad(cfg.angles)])

class JvrcWalkEnv(MujocoEnv):

    def __init__(self, cfg):
        cfg.robot = OmegaConf.load(P.config_dir / "jvrc.yaml")
        path_to_xml = P.tmp_dir / "mjcf-export/jvrc_walk/jvrc.xml"
        super().__init__(path_to_xml, cfg.dt.sim)
        self.actuators = cfg.robot.joints.leg
        self.dA = len(self.actuators)
        self.action_smoother = ActionSmoother(cfg.action_smoothing, self.dA)

        self.nominal_pose = get_pose(cfg.robot.nominal_pose)

        self.interface = RobotInterface(self.model, self.data, cfg.robot.bodies.foot.name)
        self.task = WalkingTask(self.interface, cfg.dt.ctrl, cfg.task)
        pd_gains = np.array([cfg.kp, cfg.kd])
        self.robot = RobotBase(self.interface, self.task, pd_gains, cfg.dt.ctrl, cfg.sim)

        # set observation space
        self.history_len = cfg.len_obs_history
        self.dS = 32
        self.observation_history = collections.deque(maxlen=self.history_len)
        self.set_obs_statistics()

    def merge_state_in_observation_history(self, state):
        if len(self.observation_history) == 0:
            for _ in range(self.history_len):
                self.observation_history.appendleft(np.zeros_like(state))
        self.observation_history.appendleft(state)
        return np.array(self.observation_history).flatten()

    def get_obs(self):
        # external state
        theta = 2 * np.pi * self.task._phase / self.task._period
        clock = [np.sin(theta), np.cos(theta)]
        ext_state = np.concatenate((clock, [self.task._goal_speed_ref]))
        # internal state
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())
        root_rp = tf3.euler.quat2euler(qpos[3:7])[0:2] # roll, pitch
        root_ang_vel = qvel[3:6]
        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()
        robot_state = np.concatenate([root_rp, root_ang_vel, motor_pos, motor_vel])
        # overall state
        state = np.concatenate([robot_state, ext_state])
        assert state.shape == (self.dS,)
        return self.merge_state_in_observation_history(state)

    def step(self, action):
        targets = self.action_smoother(action)
        rewards, done = self.robot.step(targets, self.nominal_pose[7:])
        obs = self.get_obs()
        return obs, sum(rewards.values()), done, rewards

    def reset_model(self):
        # set up init state
        self.set_state(self.nominal_pose, np.zeros(self.interface.nv))
        self.task.reset()
        self.action_smoother.reset()
        self.observation_history = collections.deque(maxlen=self.history_len)
        obs = self.get_obs()
        return obs

    def set_obs_statistics(self):
        # manually define observation mean and std
        self.obs_mean = np.concatenate(
            (
                np.zeros(5),
                np.deg2rad(self.nominal_pose[7:]),
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
        self.obs_mean = np.tile(self.obs_mean, self.history_len)
        self.obs_std = np.tile(self.obs_std, self.history_len)
