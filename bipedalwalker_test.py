import os
import time

import gym
from gym.envs.box2d import LunarLanderContinuous

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import numpy as np

from bipedal_walker_env import BipedalWalkerEnvMedium
WORKSTATION = not os.getlogin() == "thomas"


env = BipedalWalkerEnvMedium({"render":not WORKSTATION,
                              "hardcore": True,
                              "cheat": True})
# env = LunarLanderContinuous()

env = DummyVecEnv([lambda:env])
# env = VecFrameStack(env, n_stack=3)


# model = PPO("MlpPolicy", env)
model = SAC.load("out/sac_BipedalWalkerMedium_baseline1_fresh.zip")
model.set_env(env)


obs = env.reset()
while True:
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        # print(action)

        obs, reward, done, info = env.step(action)
        env.render()
        # time.sleep(.03)
        if done:
          obs = env.reset()