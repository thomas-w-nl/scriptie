

import gym
from gym.envs.box2d import LunarLanderContinuous

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

name = "Pong-v0"


env = gym.make(name)
# env = LunarLanderContinuous()
env = Monitor(env)

env = DummyVecEnv([lambda:env])
# env = VecFrameStack(env, n_stack=3)


model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./logs/all/")
model.learn(total_timesteps=2_000_000, tb_log_name=name)
model.save(f"Atari_{name}.zip")