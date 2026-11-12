import torch
import torch.nn as nn
from .utils import normc_fn
from .MLP import MLP


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        self.welford_reward_mean = 0.0
        self.welford_reward_mean_diff = 1.0
        self.welford_reward_n = 1

    def forward(self):
        raise NotImplementedError

    def normalize_reward(self, r, update=True):
        if update:
            if len(r.size()) == 1:
                r_old = self.welford_reward_mean
                self.welford_reward_mean += (r - r_old) / self.welford_reward_n
                self.welford_reward_mean_diff += (r - r_old) * (r - r_old)
                self.welford_reward_n += 1
            elif len(r.size()) == 2:
                for r_n in r:
                    r_old = self.welford_reward_mean
                    self.welford_reward_mean += (r_n - r_old) / self.welford_reward_n
                    self.welford_reward_mean_diff += (r_n - r_old) * (r_n - r_old)
                    self.welford_reward_n += 1
            else:
                raise NotImplementedError

        return (r - self.welford_reward_mean) / torch.sqrt(
            self.welford_reward_mean_diff / self.welford_reward_n
        )


class FF_V(Critic):

    def __init__(self, env, cfg, info):
        super(FF_V, self).__init__()
        self.value = MLP(info.dS, cfg.layers, 1)

        self.obs_std = torch.tensor(env.obs_std).float()
        self.obs_mean = torch.tensor(env.obs_mean).float()

        # weight initialization scheme used in PPO paper experiments
        self.normc_init = cfg.normc_init

        self.init_parameters()

    def init_parameters(self):
        if self.normc_init:
            self.apply(normc_fn)

    def forward(self, inputs):
        x = (inputs - self.obs_mean) / self.obs_std
        value = self.value(x)
        return value
