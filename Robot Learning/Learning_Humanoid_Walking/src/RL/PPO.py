from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import sys
import time
import numpy as np
import datetime

from Pathes import P
from RL.Sampling import sample
from RL.Buffer import BatchBuffer, PPOBuffer
from RL.TrainStats import TrainStats
from RL.weights import getRTGs
from network import Gaussian_FF_Actor, FF_V


def normalize(data, eps):
    return (data - data.mean()) / (data.std() + eps)


class PPO:

    def __init__(self, env, cfg, info):
        self.cfg = cfg
        cfg.sample.max_traj_len = cfg.sample.batch_size // cfg.sample.n_proc
        self.total_steps = 0
        self.highest_reward = -np.inf
        self.save_path = P.log_dir
        Path.mkdir(self.save_path, parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_path, flush_secs=10)

        policy = Gaussian_FF_Actor(env, cfg.network.actor, info)
        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = FF_V(env, cfg.network.critic, info)

    def update_actor_critic(self, batch: PPOBuffer, cfg):
        pdf = self.policy.distribution(batch.observations)
        log_probs = pdf.log_prob(batch.actions).sum(-1, keepdim=True)
        old_pdf = self.old_policy.distribution(batch.observations)
        old_log_probs = old_pdf.log_prob(batch.actions).sum(-1, keepdim=True)
        # ratio between old and new policy, should be one at the first iteration
        ratio = (log_probs - old_log_probs).exp()
        # clipped surrogate loss
        cpi_loss = ratio * batch.advantages
        clip_loss = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * batch.advantages
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()
        # only used for logging
        clip_fraction = torch.mean((torch.abs(ratio - 1) > cfg.clip).float()).item()
        # Value loss using the TD(gae_lambda) target
        values = self.critic(batch.observations)
        critic_loss = F.mse_loss(batch.RTGs, values)
        # Entropy loss favor exploration
        entropy_penalty = -pdf.entropy().mean()
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            approx_kl_div = torch.mean((ratio - 1) - log_ratio)
        self.actor_optimizer.zero_grad()
        (actor_loss + cfg.coeff.entropy * entropy_penalty).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()
        return (
            actor_loss,
            entropy_penalty,
            critic_loss,
            approx_kl_div,
            clip_fraction,
        )

    def train_epoch(self, cfg, batch: BatchBuffer):
        num_samples = len(batch)
        batch.toTensor()
        buffer = PPOBuffer()
        buffer.from_batch_buffer(batch)
        buffer.RTGs = getRTGs(batch, self.cfg.gamma)
        with torch.no_grad():
            values = self.critic(buffer.observations)
        buffer.advantages = normalize(buffer.RTGs - values, cfg.eps)
        self.total_steps += num_samples
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.train_stats = TrainStats()
        for epoch in range(cfg.epochs):
            random_indices = SubsetRandomSampler(range(num_samples))
            sampler = BatchSampler(random_indices, cfg.sampler.minibatch_size, drop_last=True)
            for indices in sampler:
                mini_batch = buffer[indices]
                scalars = self.update_actor_critic(mini_batch, cfg.update_ac)
                actor_loss, entropy_penalty, critic_loss, approx_kl_div, clip_fraction = scalars
                self.train_stats.append(actor_loss, entropy_penalty, critic_loss, approx_kl_div, clip_fraction)
        action_noise = self.policy.stds.data.tolist()
        return batch, action_noise

    def train(self, env, cfg):
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=cfg.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.lr, eps=cfg.eps)
        train_start_time = time.time()
        for itr in range(cfg.n_itr):
            print("********** Iteration {} ************".format(itr))
            self.policy.train()
            self.critic.train()
            sample_start_time = time.time()
            batch = sample(self.policy, self.critic, env, self.cfg.sample)
            sampling_elapsed = time.time() - sample_start_time
            print(f"Sampling took {sampling_elapsed:.2f}s for {len(batch.states)} steps.")
            optimizer_start_time = time.time()
            batch, action_noise = self.train_epoch(cfg, batch)
            optimizer_elapsed = time.time() - optimizer_start_time
            print("Optimizer took: {:.2f}s".format(optimizer_elapsed))
            self._print_train_stats_table(batch, action_noise)
            elapsed = time.time() - train_start_time
            iter_avg = elapsed / (itr + 1)
            ETA = round((cfg.n_itr - itr) * iter_avg)
            print("Total time elapsed: {:.2f}s. Total steps: {} (fps={:.2f}. iter-avg={:.2f}s. ETA={})".format(
                elapsed, self.total_steps, self.total_steps/elapsed, iter_avg, datetime.timedelta(seconds=ETA)))
            # To save time, perform evaluation only after 100 iters
            if itr == 0 or (itr + 1) % cfg.eval_freq == 0:
                nets = {"actor": self.policy, "critic": self.critic}
                evaluate_start = time.time()
                eval_batches = self.evaluate(env, nets, itr)
                eval_time = time.time() - evaluate_start
                eval_ep_lens = [float(i) for b in eval_batches for i in b.lens]
                eval_ep_rewards = [float(i) for b in eval_batches for i in b.returns]
                avg_eval_ep_lens = np.mean(eval_ep_lens)
                avg_eval_ep_rewards = np.mean(eval_ep_rewards)
                print("====EVALUATE EPISODE====")
                print("(Episode length:{:.3f}. Reward:{:.3f}. Time taken:{:.2f}s)".format(
                    avg_eval_ep_lens, avg_eval_ep_rewards, eval_time))
                self._log_eval_to_tensorboard(avg_eval_ep_rewards, avg_eval_ep_lens, itr)
            self._log_train_to_tensorboard(batch, action_noise, itr)

    def evaluate(self, env, nets, itr, num_batches=5):
        # set all nets to .eval() mode
        for net in nets.values():
            net.eval()
        # collect some batches of data
        eval_batches = []
        for _ in range(num_batches):
            batch = sample(self.policy, self.critic, env, self.cfg.sample)
            eval_batches.append(batch)
        # save all the networks
        self.save(nets, self.save_path, "_" + repr(itr))
        # save as actor.pt, if it is best
        eval_ep_rewards = [float(i) for i in batch.returns for batch in eval_batches]
        avg_eval_ep_rewards = np.mean(eval_ep_rewards)
        if self.highest_reward < avg_eval_ep_rewards:
            self.highest_reward = avg_eval_ep_rewards
            self.save(nets, self.save_path)
        return eval_batches

    @staticmethod
    def save(nets, save_path, suffix=""):
        filetype = ".pt"
        for name, net in nets.items():
            path = Path(save_path, name + suffix + filetype)
            torch.save(net, path)
            print(f"Saved {name} at {path}")
        return

    def _print_train_stats_table(self, batch, action_noise):
        sys.stdout.write("-" * 37 + "\n")
        sys.stdout.write("| %15s | %15s |" % ('Mean Eprew', "%8.5g" % torch.mean(batch.returns)) + "\n")
        sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', "%8.5g" % torch.mean(batch.lens)) + "\n")
        sys.stdout.write("| %15s | %15s |" % ('Actor loss', "%8.3g" % np.mean(self.train_stats.actor_losses)) + "\n")
        sys.stdout.write("| %15s | %15s |" % ('Critic loss', "%8.3g" % np.mean(self.train_stats.critic_losses)) + "\n")
        sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % np.mean(self.train_stats.kls)) + "\n")
        sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % np.mean(self.train_stats.entropies)) + "\n")
        sys.stdout.write("| %15s | %15s |" % ('Clip Fraction', "%8.3g" % np.mean(self.train_stats.clip_fractions)) + "\n")
        sys.stdout.write("| %15s | %15s |" % ('Mean noise std', "%8.3g" % np.mean(action_noise)) + "\n")
        sys.stdout.write("-" * 37 + "\n")
        sys.stdout.flush()

    def _log_eval_to_tensorboard(self, avg_eval_ep_rewards, avg_eval_ep_lens, itr):
        self.writer.add_scalar("Eval/mean_reward", avg_eval_ep_rewards, itr)
        self.writer.add_scalar("Eval/mean_episode_length", avg_eval_ep_lens, itr)

    def _log_train_to_tensorboard(self, batch, action_noise, itr):
        self.writer.add_scalar("Loss/actor", np.mean(self.train_stats.actor_losses), itr)
        self.writer.add_scalar("Loss/critic", np.mean(self.train_stats.critic_losses), itr)
        self.writer.add_scalar("Train/mean_reward", torch.mean(batch.returns), itr)
        self.writer.add_scalar("Train/mean_episode_length", torch.mean(batch.lens), itr)
        self.writer.add_scalar("Train/mean_noise_std", np.mean(action_noise), itr)
