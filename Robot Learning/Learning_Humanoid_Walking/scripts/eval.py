import torch
import time
from pathlib import Path
from pathlib import Path
import sys
import argparse
import ray
from functools import partial

import numpy as np
import torch
import pickle
import shutil
import mujoco
import mujoco.viewer

import imageio
from datetime import datetime


class EvaluateEnv:
    def __init__(self, env, policy, args):
        self.env = env
        self.policy = policy
        self.ep_len = args.ep_len

        if args.out_dir is None:
            args.out_dir = Path(args.path.parent, "videos")

        video_outdir = Path(args.out_dir)
        try:
            Path.mkdir(video_outdir, exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            video_fn = Path(video_outdir, args.path.stem + "-" + now + ".mp4")
            self.writer = imageio.get_writer(video_fn, fps=60)
        except Exception as e:
            print("Could not create video writer:", e)
            exit(-1)

    @torch.no_grad()
    def run(self):

        height = 480
        width = 640
        renderer = mujoco.Renderer(self.env.model, height, width)
        viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        frames = []

        # Make a camera.
        cam = viewer.cam
        mujoco.mjv_defaultCamera(cam)
        cam.elevation = -20
        cam.distance = 4

        reset_counter = 0
        observation = self.env.reset()
        while self.env.data.time < self.ep_len:

            step_start = time.time()

            # forward pass and step
            raw = self.policy.forward(torch.tensor(observation, dtype=torch.float32), deterministic=True).detach().numpy()
            observation, reward, done, _ = self.env.step(raw.copy())

            # render scene
            cam.lookat = self.env.data.body(1).xpos.copy()
            renderer.update_scene(self.env.data, cam)
            pixels = renderer.render()
            frames.append(pixels)

            viewer.sync()

            if done and reset_counter < 3:
                observation = self.env.reset()
                reset_counter += 1

            time_until_next_step = max(
                0, self.env.frame_skip*self.env.model.opt.timestep - (time.time() - step_start))
            time.sleep(time_until_next_step)

        for frame in frames:
            self.writer.append_data(frame)
        self.writer.close()
        self.env.close()
        viewer.close()

def eval(args):

    path_to_actor = ""
    if args.path.is_file() and args.path.suffix == ".pt":
        path_to_actor = args.path
    elif args.path.is_dir():
        path_to_actor = Path(args.path, "actor.pt")
    else:
        raise Exception("Invalid path to actor module: ", args.path)

    path_to_critic = Path(
        path_to_actor.parent, "critic" + str(path_to_actor).split("actor")[1]
    )
    path_to_pkl = Path(path_to_actor.parent, "experiment.pkl")

    # load experiment args
    run_args = pickle.load(open(path_to_pkl, "rb"))
    # load trained policy
    policy = torch.load(path_to_actor, weights_only=False)
    critic = torch.load(path_to_critic, weights_only=False)
    policy.eval()
    critic.eval()

    # load experiment args
    run_args = pickle.load(open(path_to_pkl, "rb"))

    # import the correct environment
    Env = import_env(run_args.env)
    if "yaml" in run_args and run_args.yaml is not None:
        yaml_path = Path(run_args.yaml)
    else:
        yaml_path = None
    env = partial(Env, yaml_path)()

    # run
    e = EvaluateEnv(env, policy, args)
    e.run()
