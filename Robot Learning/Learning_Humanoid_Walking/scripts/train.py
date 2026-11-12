from omegaconf import OmegaConf

from RL.PPO import PPO
from envs.env_dict import env_dict
from Pathes import P


if __name__ == "__main__":
    cfg = OmegaConf.load(P.config_dir / "walk.yaml")
    Env = env_dict[f"{cfg.env.robot.name[:4]}_{cfg.env.task.name}"](cfg.env)
    ppo = PPO(Env, cfg.ppo, cfg.info)
    ppo.train(Env, cfg.ppo.train)
