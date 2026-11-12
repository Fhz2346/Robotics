import torch


class EpisodeBuffer:

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


class BatchBuffer:

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.returns = []
        self.lens = []
        self.last_values = []

    def __len__(self):
        return len(self.states)

    def append_episode(self, episode_buffer: EpisodeBuffer, last_value):
        self.actions += episode_buffer.actions
        self.states += episode_buffer.states
        self.rewards += episode_buffer.rewards
        ep_ret = sum(episode_buffer.rewards)
        ep_len = len(episode_buffer.rewards)
        self.returns.append(ep_ret)
        self.lens.append(ep_len)
        self.last_values.append(last_value.view(-1)[0])

    def toTensor(self):
        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions)
        self.rewards = torch.stack(self.rewards).view(-1, 1)
        self.returns = torch.tensor(self.returns, dtype=torch.float32)
        self.lens = torch.tensor(self.lens, dtype=torch.float32)
        self.last_values = torch.stack(self.last_values)


class PPOBuffer:

    def __init__(self):
        self.observations = []
        self.actions = []
        self.advantages = []
        self.RTGs = []

    def from_batch_buffer(self, batch_buffer: BatchBuffer):
        self.observations = batch_buffer.states
        self.actions = batch_buffer.actions

    def __getitem__(self, indices):
        buffer = PPOBuffer()
        buffer.observations = self.observations[indices]
        buffer.actions = self.actions[indices]
        buffer.RTGs = self.RTGs[indices]
        buffer.advantages = self.advantages[indices]
        return buffer