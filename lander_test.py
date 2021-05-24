import gym
from gym.envs.box2d import LunarLanderContinuous

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


class LunarLanderContinuousDifficult(LunarLanderContinuous):
    def step(self, action):
        ob, reward, done, info = super().step(action)
        ob[2:3] = 0
        return ob, reward, done, info

env = LunarLanderContinuousDifficult()

env = DummyVecEnv([lambda:env])
env = VecFrameStack(env, n_stack=3)


model = SAC.load("Lunar_DummyVecEnv.zip")

obs = env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()