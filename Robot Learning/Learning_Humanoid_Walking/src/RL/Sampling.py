import torch
from RL.Buffer import BatchBuffer, EpisodeBuffer

@torch.no_grad()
def _sample_one_traj(actor, critic, env, cfg):
    buffer = EpisodeBuffer()
    state = torch.tensor(env.reset(), dtype=torch.float)
    done = False
    traj_len = 0
    while not done and traj_len < cfg.max_traj_len:
        action = actor(state, cfg.deterministic)
        next_state, reward, done, _ = env.step(action.numpy().copy())
        reward = torch.tensor(reward, dtype=torch.float)
        buffer.append(state, action, reward)
        state = torch.tensor(next_state, dtype=torch.float)
        traj_len += 1
    value = critic(state)
    return buffer, done, value


def sample(actor, critic, env, cfg):
    buffer = BatchBuffer()
    while len(buffer) < cfg.batch_size:
        episode_buffer, done, value = _sample_one_traj(actor, critic, env, cfg)
        buffer.append_episode(episode_buffer, last_value=(not done) * value)
    return buffer
