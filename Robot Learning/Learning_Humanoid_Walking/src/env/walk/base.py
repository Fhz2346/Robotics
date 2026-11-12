import numpy as np
from ..JvrcLeg import JvrcLeg

class WalkingBase:

    def __init__(self, cfg, robot: JvrcLeg):
        self.robot = robot
        self.floor_name = self.robot.model.body(0).name
        self.cfg = cfg

    def done(self):
        contact_flag = self.robot.check_self_collisions()
        qpos = self.robot.data.qpos.copy()
        terminate_conditions = [
            (qpos[2] < 0.6),  # lower limit of height
            (qpos[2] > 1.4),  # upper limit of height
            contact_flag,
        ]
        done = True in terminate_conditions
        return done

    def calc_reward(self, prev_torque, prev_action, action):
        reward = dict(
            orient_cost=0.050
            * (
                self.body_orient(self.robot.Lfoot.name)
                + self.body_orient(self.robot.Rfoot.name)
                + self.body_orient(self.robot.root_name)
            )
            / 3,
            root_accel=0.050 * self.root_accel(),
            height_error=0.050 * self.height(),
            com_vel_error=0.200 * self.fwd_vel(),
            torque_penalty=0.050 * self.torque(prev_torque),
            action_penalty=0.050 * self.action(action, prev_action),
        )
        return reward

    # Reward Functions
    def body_orient(self, body_name, quat_ref=[1, 0, 0, 0]):
        body_quat = self.robot.data.body(body_name).xquat.copy()
        target_quat = np.array(quat_ref)
        error = 10 * (1 - np.inner(target_quat, body_quat) ** 2)
        return np.exp(-error)

    def root_accel(self):
        qvel = self.robot.data.qvel.copy()
        qacc = self.robot.data.qacc.copy()
        error = 0.25 * (np.abs(qvel[3:6]).sum() + np.abs(qacc[0:3]).sum())
        return np.exp(-error)

    def height(self):
        is_contact = (
            len(self.robot.Rfoot.floor_contacts) > 0 or 
            len(self.robot.Lfoot.floor_contacts) > 0
        )
        if is_contact:
            contact_point = min(
                [
                    c.pos[2]
                    for _, c in self.robot.Rfoot.floor_contacts + self.robot.Lfoot.floor_contacts
                ]
            )
        else:
            contact_point = 0
        current_height = self.robot.data.body(self.robot.root_name).xpos[2]
        relative_height = current_height - contact_point
        error = np.abs(relative_height - self.cfg.goal_height_ref)
        deadzone_size = 0.01 + 0.05 * self.cfg.goal_speed_ref
        if error < deadzone_size:
            error = 0
        return np.exp(-40 * np.square(error))

    def fwd_vel(self):
        root_vel = self.robot.data.qvel[0]
        error = root_vel - self.cfg.goal_speed_ref
        return np.exp(-error)

    def action(self, action, prev_action):
        penalty = 5 * sum(np.abs(prev_action - action)) / len(action)
        return np.exp(-penalty)

    def torque(self, prev_torque):
        torque = np.asarray(self.robot.get_act_joint_torques())
        penalty = 0.25 * (sum(np.abs(prev_torque - torque)) / len(torque))
        return np.exp(-penalty)
