import torch
import torch.nn as nn
from torch.distributions import Normal
from .MLP import MLP
from .utils import normc_fn


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()

    def forward(self):
        raise NotImplementedError


class Gaussian_FF_Actor(Actor):  # more consistent with other actor naming conventions

    def __init__(self, env, cfg, info):
        super(Gaussian_FF_Actor, self).__init__()
        self.means = MLP(info.dS, cfg.layers, info.dA)
        self.learn_std = cfg.learn_std
        stds = cfg.std_dev * torch.ones(info.dA)
        if self.learn_std:
            self.stds = nn.Parameter(stds)
        else:
            self.stds = stds
        # Initialized to no input normalization, can be modified later
        self.obs_std = torch.Tensor(env.obs_std)
        self.obs_mean = torch.Tensor(env.obs_mean)
        # weight initialization scheme used in PPO paper experiments
        self.normc_init = cfg.normc_init
        self.bounded = cfg.bounded
        self.init_parameters()

    def init_parameters(self):
        if self.normc_init:
            self.apply(normc_fn)
            last_layer = self.means.net[-1]
            last_layer.weight.data.mul_(0.01)

    def _get_dist_params(self, state):
        state = (state - self.obs_mean) / self.obs_std
        mean = self.means(state)
        if self.bounded:
            mean = torch.tanh(mean)
        sd = torch.zeros_like(mean)
        if hasattr(self, "stds"):
            sd = self.stds
        return mean, sd

    def forward(self, state, deterministic=True):
        mu, sd = self._get_dist_params(state)
        if not deterministic:
            action = Normal(mu, sd).sample()
        else:
            action = mu
        return action

    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        return Normal(mu, sd)
