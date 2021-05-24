import os

import gym
from gym.envs.box2d import LunarLanderContinuous

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import numpy as np

from utils import linear_schedule
from bipedal_walker_env import BipedalWalkerEnvMedium

WORKSTATION = not os.getlogin() == "thomas"

name = "sac_BipedalWalkerMedium_stacked5_nolidar2"

env = BipedalWalkerEnvMedium({"render": not WORKSTATION,
                              "hardcore": True,
                              "cheat": False})
# env = LunarLanderContinuous()
env = Monitor(env)

env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=5)


model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="./logs/all/",
            policy_kwargs=dict(net_arch=[400, 300]),
            learning_rate=7.3e-4,
            train_freq=1,
            gradient_steps=1,
            learning_starts=10000,
            use_sde=True,
            )

# model = SAC.load("out/sac_BipedalWalkerMedium_baseline1_fresh.zip")
# model.set_env(env)
# model.gradient_steps = 1
# model.train_freq = 1
# model.learning_starts = 0
# model.learning_rate = linear_schedule(0.0001)


model.learn(total_timesteps=5_000_000, tb_log_name=name)
model.save(f"out/{name}.zip")
