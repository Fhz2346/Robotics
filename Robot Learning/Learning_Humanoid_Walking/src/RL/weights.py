import torch
from RL.Buffer import BatchBuffer


def reward_to_go(rewards, gamma):
    T = len(rewards)
    return [
        torch.sum(rewards[i:] * (gamma ** torch.arange(T - i, device=rewards.device)))
        for i in range(T)
    ]


def getRTGs(batch: BatchBuffer, gamma):
    rtgs = []
    idx_ep_beg = 0
    for ep_len in batch.lens.tolist():
        ep_len = int(ep_len)
        ep_rewards = batch.rewards[idx_ep_beg : idx_ep_beg + ep_len].float().view(-1)
        ep_rtgs = reward_to_go(ep_rewards, gamma)
        rtgs += ep_rtgs
        idx_ep_beg += ep_len
    return torch.stack(rtgs) if rtgs else torch.zeros(0, dtype=torch.float32)
