###
### V2
###
import time

from stable_baselines3 import SAC, PPO
from envs import *
import gym

experiment_conf = {"render": True,
                   "terrain": "flat",
                   # "cheat": True,
                   "perturb_magnitude": 1,
                   "desc": ""}

# env = ANYMalStandupEnv(experiment_conf)
env = BipedalWalkerEnv(expezriment_conf)
# env = HumanoidEnvGym(experiment_conf)
env.render()
# env = InvertedPendulumEnvR(experiment_conf)

model = SAC.load("models/sac_TimeLimit_flat_baseline_hman_orig.zip")

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