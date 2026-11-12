import contextlib
import os
import numpy as np
import mujoco
import mujoco_viewer

DEFAULT_SIZE = 500


class MujocoEnv:

    def __init__(self, model_path, sim_dt):
        self.spec = mujoco.MjSpec()
        self.spec.from_file(str(model_path))
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.model.opt.timestep = sim_dt

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 2.0
        self.viewer.cam.elevation = -20
        self.viewer.vopt.geomgroup[2] = 0
        self.viewer._render_every_frame = True

    def viewer_is_paused(self):
        return self.viewer._paused

    # -----------------------------
    # (some methods are taken directly from dm_control)

    @contextlib.contextmanager
    def disable(self, *flags):
        old_bitmask = self.model.opt.disableflags
        new_bitmask = old_bitmask
        for flag in flags:
            if isinstance(flag, str):
                try:
                    field_name = "mjDSBL_" + flag.upper()
                    flag = getattr(mujoco.mjtDisableBit, field_name)
                except AttributeError:
                    valid_names = [
                        field_name.split("_")[1].lower()
                        for field_name in list(mujoco.mjtDisableBit.__members__)[:-1]
                    ]
                    raise ValueError(
                        "'{}' is not a valid flag name. Valid names: {}".format(
                            flag, ", ".join(valid_names)
                        )
                    ) from None
            elif isinstance(flag, int):
                flag = mujoco.mjtDisableBit(flag)
            new_bitmask |= flag.value
        self.model.opt.disableflags = new_bitmask
        try:
            yield
        finally:
            self.model.opt.disableflags = old_bitmask

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.act = []
        self.data.plugin_state = []
        # Disable actuation since we don't yet have meaningful control inputs.
        with self.disable("actuation"):
            mujoco.mj_forward(self.model, self.data)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer_setup()
        self.viewer.render()

    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        if hfieldid is not None:  # hfield
            mujoco.mjr_uploadHField(self.model, self.viewer.ctx, hfieldid)
        if meshid is not None:  # mesh
            mujoco.mjr_uploadMesh(self.model, self.viewer.ctx, meshid)
        if texid is not None:  # texture
            mujoco.mjr_uploadTexture(self.model, self.viewer.ctx, texid)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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
    
    