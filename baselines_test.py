###
### V2
###
import time

from stable_baselines3 import SAC, PPO
from envs import *
import gym

model_path = "models/old/sac_BipedalWalkerEnv_cheat_flat_baseline1/best_model.zip"
model = SAC.load(model_path)


experiment_conf = {"render": True,
                   "terrain": "flat",
                   "cheat": "cheat" in model_path,
                   "perturb_magnitude": 1,
                   "hardcore": False,
                   "desc": ""}

# env = ANYMalStandupEnv(experiment_conf)
env = BipedalWalkerEnv(experiment_conf)
# env = HumanoidEnv(experiment_conf)
# env = HumanoidEnvGym(experiment_conf)
env.render()
# env = InvertedPendulumEnvR(experiment_conf)


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