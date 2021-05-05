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

from envs import OLD_ANYmalEnv

experiment_conf = {"render": True,
                   "terrain": "flat",
                   # "cheat": "cheat" in model_path,
                   "cheat": True,
                   "perturb_magnitude": 1,
                   "hardcore": True,
                   "desc": ""}




env = OLD_ANYmalEnv(experiment_conf)
env.render()



model_path = "pretrained/anymal_base.zip"
model = SAC.load(model_path)

# torch.save(model.policy.state_dict(), "models/baselines/BipedalWalkerHardcore-statedict.th")

# model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[400, 300]))
# model.policy.load_state_dict(torch.load("BipedalWalkerHardcore-statedict.th"))

model.set_env(env)
model.learn(total_timesteps=10000)



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