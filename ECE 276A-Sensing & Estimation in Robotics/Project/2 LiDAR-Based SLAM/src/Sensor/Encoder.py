import numpy as np
from load_data import EncoderData


class Encoder:
    def __init__(self, counts, time_stamps):
        self.counts = np.asarray(counts, dtype=np.float64)
        self.T = np.asarray(time_stamps, dtype=np.float64)
        self.tick2m = 0.0022
        self.init()

    def init(self):
        self.N = self.counts.shape[0]

    @staticmethod
    def from_dataset(ds_encoders: EncoderData):
        return Encoder(ds_encoders.counts, ds_encoders.time_stamps)

    def estimate_dist(self, k):
        assert 0 <= k < self.N - 1
        FR, FL, RR, RL = self.counts[k]
        dR = ((FR + RR) / 2.0) * self.tick2m
        dL = ((FL + RL) / 2.0) * self.tick2m
        d = (dR + dL) / 2.0
        return d
