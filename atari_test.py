import time

import gym
from gym.envs.box2d import LunarLanderContinuous

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

name = "BattleZone-v0"


env = gym.make(name)
# env = LunarLanderContinuous()

# env = VecFrameStack(env, n_stack=3)


model = PPO.load("Atari_BattleZone-v0.zip")


obs = env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    # action = env.action_space.sample()
    # print(action)

    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(.03)
    if done:
      obs = env.reset()