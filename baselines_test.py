###
### V2
###
import time

from stable_baselines3 import SAC, PPO
# from envs import *
import gym
import numpy as np
from gym.envs import box2d

import torch

from envs import HumanoidEnv, ANYMalStandupEnv


class BipedalWalkerEnv(gym.envs.box2d.BipedalWalker):

    def __init__(self, config):
        self.velocities = []
        self.do_render = config.get("render", False)
        self.cheat = config.get("cheat", False)
        super().__init__()
        self.hardcore = config.get("hardcore", False)

        self.time = 0

        # clip lidar for non cheating agent
        if not self.cheat:
            high = np.array([np.inf] * 14)
            self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def step(self, a):
        ob, reward, done, info = super().step(a)
        vel = self.hull.linearVelocity.x

        self.velocities.append(vel)

        info.update(dict(avg_speed=np.mean(self.velocities),
                         checkpoints=-1))

        if self.do_render:
            self.render()

        self.time += 1
        if self.time > 2500:
            done = True

        # clip lidar for non cheating agent
        if not self.cheat:
            ob = ob[:14]
        return ob, reward, done, info

    def reset(self):
        self.time = 0
        self.velocities = []
        return super().reset()

model_path = "pretrained/BipedalWalkerHardcore_pretrained.zip"


experiment_conf = {"render": True,
                   "terrain": "flat",
                   # "cheat": "cheat" in model_path,
                   "cheat": True,
                   "perturb_magnitude": 1,
                   "hardcore": True,
                   "desc": ""}

# env = ANYMalStandupEnv(experiment_conf)
env = BipedalWalkerEnv(experiment_conf)
# env = HumanoidEnv(experiment_conf)
# env = HumanoidEnvGym(experiment_conf)
env.render()
# env = InvertedPendulumEnvR(experiment_conf)



model = SAC.load(model_path)

# torch.save(model_teacher.policy.state_dict(), "models/baselines/BipedalWalkerHardcore-statedict.th")

# model_teacher = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[400, 300]))
# model_teacher.policy.load_state_dict(torch.load("BipedalWalkerHardcore-statedict.th"))






scores = []
for i in range(100):
    obs = env.reset()
    score = 0
    for i in range(2000):
        action, _states = model.predict(obs)
        # print(action)
        # action = np.array([np.sin(i/50)] * 12)
        # action = env.action_space.sample() * 0
        obs, reward, done, info = env.step(action)
        # print()
        # print("direction_reward", info["direction_reward"])
        # print("speed_reward", info["speed_reward"])
        score += reward
        env.render()
        if done:
            break

    scores.append(score)
    print("SCORE", score)


print("avg", np.mean(scores))

# pendulum cheat: avg 233.75
# pendulum: avg 233.75

# pendulum cheat 2x perturb: avg 102.83
# pendulum 2x perturb: avg 141.27 +