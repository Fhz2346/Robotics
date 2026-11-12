import numpy as np


class ActionSmoother:

    def __init__(self, alpha, dA):
        self.alpha = alpha
        self.alpha_comp = 1 - alpha
        self.dA = dA
        self.reset()

    def __call__(self, action: np.ndarray):
        action_smoothed = self.alpha * action + self.alpha_comp * self.prev_action
        self.prev_action = action
        return action_smoothed

    def reset(self):
        self.prev_action = np.zeros(self.dA)
