class TrainStats:

    def __init__(self):
        self.actor_losses = []
        self.entropies = []
        self.critic_losses = []
        self.kls = []
        self.clip_fractions = []

    def append(
        self,
        actor_loss,
        entropy_penalty,
        critic_loss,
        approx_kl_div,
        clip_fraction,
    ):
        self.actor_losses.append(actor_loss.item())
        self.entropies.append(entropy_penalty.item())
        self.critic_losses.append(critic_loss.item())
        self.kls.append(approx_kl_div.item())
        self.clip_fractions.append(clip_fraction)