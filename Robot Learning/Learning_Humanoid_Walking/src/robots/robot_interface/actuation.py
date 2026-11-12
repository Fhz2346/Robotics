import numpy as np
import mujoco


class ActuationMixin:

    def get_motor_speed_limits(self):
        """
        Returns speed limits of the actuator in radians/sec.
        Assumes actuator_user[:, 0] stores RPM.
        """
        rpm_limits = self.model.actuator_user[:, 0]
        return rpm_limits * (2 * np.pi / 60)

    def get_act_joint_speed_limits(self):
        """
        Returns speed limits at joint level in radians/sec.
        """
        gear_ratios = self.model.actuator_gear[:, 0]
        mot_lims = self.get_motor_speed_limits()
        return np.array([float(i / j) for i, j in zip(mot_lims, gear_ratios)])

    def get_gear_ratios(self):
        return self.model.actuator_gear[:, 0]

    def get_motor_names(self):
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

    def get_actuated_joint_inds(self):
        """
        Returns joint indices to which actuators are attached.
        Assumes actuator naming style: <joint_name>_motor
        """
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        actuator_names = self.get_motor_names()
        return [idx for idx, jnt in enumerate(joint_names) if f"{jnt}_motor" in actuator_names]

    def get_actuated_joint_names(self):
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        actuator_names = self.get_motor_names()
        return [jnt for jnt in joint_names if f"{jnt}_motor" in actuator_names]

    def get_motor_qposadr(self):
        indices = self.get_actuated_joint_inds()
        return [self.model.jnt_qposadr[i] for i in indices]

    def get_motor_positions(self):
        return self.data.actuator_length.copy()

    def get_motor_velocities(self):
        return self.data.actuator_velocity.copy()

    def get_act_joint_torques(self):
        gear_ratios = self.model.actuator_gear[:, 0]
        motor_torques = self.data.actuator_force
        return motor_torques * gear_ratios

    def get_act_joint_positions(self):
        gear_ratios = self.model.actuator_gear[:, 0]
        motor_positions = self.get_motor_positions()
        return motor_positions / gear_ratios

    def get_act_joint_velocities(self):
        gear_ratios = self.model.actuator_gear[:, 0]
        motor_velocities = self.get_motor_velocities()
        return motor_velocities / gear_ratios

    def get_act_joint_position(self, act_name):
        assert len(self.data.actuator(act_name).length) == 1
        return self.data.actuator(act_name).length[0] / self.model.actuator(act_name).gear[0]

    def get_act_joint_velocity(self, act_name):
        assert len(self.data.actuator(act_name).velocity) == 1
        return self.data.actuator(act_name).velocity[0] / self.model.actuator(act_name).gear[0]

    def get_act_joint_ranges(self):
        indices = self.get_actuated_joint_inds()
        low, high = self.model.jnt_range[indices, :].T
        return low, high

    def get_act_joint_range(self, joint_name):
        low, high = self.model.joint(joint_name).range
        return low, high

    def get_actuator_ctrl_range(self):
        low, high = self.model.actuator_ctrlrange.copy().T
        return low, high

    def get_actuator_user_data(self):
        return self.model.actuator_user.copy()
