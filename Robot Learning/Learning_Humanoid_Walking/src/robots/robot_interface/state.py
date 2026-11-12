class StateMixin:

    def get_qpos(self):
        return self.data.qpos.copy()

    def get_qvel(self):
        return self.data.qvel.copy()

    def get_qacc(self):
        return self.data.qacc.copy()

    def get_cvel(self):
        return self.data.cvel.copy()

    def get_body_ext_force(self):
        return self.data.cfrc_ext.copy()

    def get_sensordata(self, sensor_name):
        sensor = self.model.sensor(sensor_name)
        sensor_adr = sensor.adr[0]
        data_dim = sensor.dim[0]
        return self.data.sensordata[sensor_adr : sensor_adr + data_dim]
