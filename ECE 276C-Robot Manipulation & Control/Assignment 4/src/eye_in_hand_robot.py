import numpy as np
import pybullet as p
import pybullet_data
from Camera import Camera


# Robot with Camera Class
class eye_in_hand_robot:

    def __init__(self, cfg):
        self.robot_id = p.loadURDF(pybullet_data.getDataPath() + "/franka_panda/panda.urdf", useFixedBase=True)
        self.eeFrameId = []
        self.camera_offset = 0.1  # offset camera in z direction to avoid grippers
        self.camera = Camera(cfg.camera)
        # Get the joint info
        self._numLinkJoints = p.getNumJoints(self.robot_id)  # includes passive joint
        jointInfo = [p.getJointInfo(self.robot_id, i) for i in range(self._numLinkJoints)]

        # Get joint locations (some joints are passive)
        self._active_joint_indices = []
        for i in range(self._numLinkJoints):
            if jointInfo[i][2] == p.JOINT_REVOLUTE:
                self._active_joint_indices.append(jointInfo[i][0])
        self.numActiveJoints = len(self._active_joint_indices)  # exact number of active joints
        # reset joints
        for i in range(self._numLinkJoints):
            p.resetJointState(self.robot_id, i, cfg.init_q[i])
        self.camera.set_pose(*self.get_camera_pose())

    def get_ee_pose(self):
        idx = self.numActiveJoints
        state = p.getLinkState(self.robot_id, idx)
        pos = np.array(state[0])
        orn = np.array(p.getMatrixFromQuaternion(state[1])).reshape(3, 3)
        return pos, orn

    def get_camera_pose(self):
        ee_pos, ee_orn = self.get_ee_pose()
        pos = ee_pos + self.camera_offset * ee_orn[:, 2]
        orn = ee_orn
        return pos, orn

    def step(self, num_steps):
        for _ in range(num_steps):
            p.stepSimulation()
        self.camera.set_pose(*self.get_camera_pose())
        self.camera.capture()

    def get_active_joint_states(self):
        """Get the states (position, velocity, and torques) of the active joints of the robot"""
        joint_states = p.getJointStates(self.robot_id, range(self._numLinkJoints))
        joint_infos = [p.getJointInfo(self.robot_id, i) for i in range(self._numLinkJoints)]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def get_q(self):
        q = np.zeros(self.numActiveJoints)
        for i in range(self.numActiveJoints):
            joint_state = p.getJointState(self.robot_id, self._active_joint_indices[i])
            q[i] = joint_state[0]
        return q

    def get_jacobian(self):
        # Returns the Robot Jacobian of the last active link
        mpos, mvel, mtorq = self.get_active_joint_states()
        zero_vec = [0.0] * len(mpos)
        linearJacobian, angularJacobian = p.calculateJacobian(
            self.robot_id,
            self.numActiveJoints,
            [0, 0, self.camera_offset],
            mpos,
            zero_vec,
            zero_vec,
        )
        # only return the active joint's jacobians
        Jacobian = np.vstack((linearJacobian, angularJacobian))
        return Jacobian[:, : self.numActiveJoints]

    def set_q(self, desired_q, kp=1.0, kv=0.3):
        allJointPositionObjectives = [0.0] * self._numLinkJoints
        for i in range(desired_q.shape[0]):
            idx = self._active_joint_indices[i]
            allJointPositionObjectives[idx] = desired_q[i]

        p.setJointMotorControlArray(
            self.robot_id,
            range(self._numLinkJoints),
            p.POSITION_CONTROL,
            targetPositions=allJointPositionObjectives,
            targetVelocities=[0.0] * self._numLinkJoints,
            positionGains=[kp] * self._numLinkJoints,
            velocityGains=[kv] * self._numLinkJoints,
        )
