import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, dI, dHs, dO):
        super().__init__()
        layers = []
        dims = [dI] + dHs
        for i in range(len(dHs)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dHs[-1], dO))
        self.net = nn.Sequential(*layers)
        self.to(dtype=torch.float)

    def forward(self, x):
        return self.net(x)
