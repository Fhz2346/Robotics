from omegaconf import OmegaConf

from RL.PPO import PPO
from envs.env_dict import env_dict
from Pathes import P
from env.walk import Walking
from env.JvrcLeg import JvrcLeg


if __name__ == "__main__":
    cfg = OmegaConf.load(P.config_dir / "Task" / "walk.yaml")
    robot = JvrcLeg(cfg)
    Env = Walking(cfg, robot)
    ppo = PPO(Env, cfg.rl, cfg.info)
    ppo.train(Env, cfg.rl.train)
