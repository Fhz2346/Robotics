import numpy as np
import mujoco
import transforms3d as tf3


class KinematicsMixin:
    def get_root_body_pos(self):
        return self.data.xpos[1].copy()

    def get_root_body_vel(self):
        qveladr = self.get_jnt_qveladr_by_name("root")
        return self.data.qvel[qveladr:qveladr + 6].copy()

    def get_body_vel(self, body_name, frame=0):
        """
        Returns [linear_velocity, angular_velocity]
        for the given body.
        """
        body_vel = np.zeros(6)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_XBODY,
            body_id,
            body_vel,
            frame,
        )
        return [body_vel[3:6], body_vel[0:3]]

    def get_object_xpos_by_name(self, object_name, object_type):
        if object_type == "OBJ_BODY":
            return self.data.body(object_name).xpos.copy()
        elif object_type == "OBJ_GEOM":
            return self.data.geom(object_name).xpos.copy()
        elif object_type == "OBJ_SITE":
            return self.data.site(object_name).xpos.copy()
        else:
            raise ValueError("object_type should be OBJ_BODY / OBJ_GEOM / OBJ_SITE.")

    def get_object_xquat_by_name(self, object_name, object_type):
        if object_type == "OBJ_BODY":
            return self.data.body(object_name).xquat.copy()
        elif object_type == "OBJ_GEOM":
            xmat = self.data.geom(object_name).xmat.reshape(3, 3)
            return tf3.quaternions.mat2quat(xmat)
        elif object_type == "OBJ_SITE":
            xmat = self.data.site(object_name).xmat.reshape(3, 3)
            return tf3.quaternions.mat2quat(xmat)
        else:
            raise ValueError("object_type should be OBJ_BODY / OBJ_GEOM / OBJ_SITE.")

    def get_object_affine_by_name(self, object_name, object_type):
        pos = self.get_object_xpos_by_name(object_name, object_type)
        quat = self.get_object_xquat_by_name(object_name, object_type)
        return tf3.affines.compose(pos, tf3.quaternions.quat2mat(quat), np.ones(3))

    def get_robot_com(self):
        sensor_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(self.model.nsensor)
        ]
        if "subtreecom" not in sensor_names:
            raise RuntimeError("subtreecom sensor not attached.")
        return self.data.subtree_com[1].copy()

    def get_robot_linmom(self):
        sensor_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(self.model.nsensor)
        ]
        if "subtreelinvel" not in sensor_names:
            raise RuntimeError("subtreelinvel sensor not attached.")
        linvel = self.data.subtree_linvel[1].copy()
        total_mass = self.get_robot_mass()
        return linvel * total_mass

    def get_robot_angmom(self):
        sensor_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(self.model.nsensor)
        ]
        if "subtreeangmom" not in sensor_names:
            raise RuntimeError("subtreeangmom sensor not attached.")
        com_pos = self.get_robot_com()
        lin_mom = self.get_robot_linmom()
        return self.data.subtree_angmom[1].copy() + np.cross(com_pos, lin_mom)