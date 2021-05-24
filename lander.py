import gym
from gym.envs.box2d import LunarLanderContinuous

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class LunarLanderContinuousDifficult(LunarLanderContinuous):
    def step(self, action):
        ob, reward, done, info = super().step(action)
        ob[2:3] = 0
        return ob, reward, done, info

env = LunarLanderContinuousDifficult()
# env = LunarLanderContinuous()
env = Monitor(env)

env = DummyVecEnv([lambda:env])
# env = VecFrameStack(env, n_stack=3)

name = "LunarLanderDifficult"

model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="./logs/all/")
model.learn(total_timesteps=200_000, tb_log_name=name)
model.save(f"Lunar_{name}.zip")