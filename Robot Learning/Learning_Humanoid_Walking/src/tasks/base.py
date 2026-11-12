import numpy as np
from numpy.linalg import norm
from robots.robot_interface import RobotInterface

class BaseTask:

    def __init__(self, client: RobotInterface):
        self._client = client

    def body_orient(self, body_name, quat_ref=[1, 0, 0, 0]):
        # body orientation reward
        body_quat = self._client.get_object_xquat_by_name(body_name, "OBJ_BODY")
        target_quat = np.array(quat_ref)
        error = 10 * (1 - np.inner(target_quat, body_quat) ** 2)
        return np.exp(-error)
    
    def root_accel(self):
        qvel = self._client.get_qvel()
        qacc = self._client.get_qacc()
        error = 0.25 * (np.abs(qvel[3:6]).sum() + np.abs(qacc[0:3]).sum())
        return np.exp(-error)
    
    def torque(self, prev_torque):
        # actuator torque reward
        torque = np.asarray(self._client.get_act_joint_torques())
        penalty = 0.25 * (sum(np.abs(prev_torque - torque)) / len(torque))
        return np.exp(-penalty)
    
    def action(self, action, prev_action):
        # action reward
        penalty = 5 * sum(np.abs(prev_action - action)) / len(action)
        return np.exp(-penalty)
