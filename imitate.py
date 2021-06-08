import os
import time

import gym
from gym.envs.box2d import LunarLanderContinuous

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from collections import deque

import numpy as np
from torch.utils.data import DataLoader

from bipedal_walker_env import BipedalWalkerEnvMedium

WORKSTATION = not os.getlogin() == "thomas"

env = BipedalWalkerEnvMedium({"render": not WORKSTATION,
                              "hardcore": True,
                              "cheat": True})

env = DummyVecEnv([lambda: env])


# env = VecFrameStack(env, n_stack=3)


def predict_grad(model, observation):
    """
     Get the policy action and state from an observation (and optional state).
     Includes sugar-coating to handle different observations (e.g. normalizing images).

     :param observation: the input observation
     :param state: The last states (can be None, used in recurrent policies)
     :param mask: The last masks (can be None, used in recurrent policies)
     :param deterministic: Whether or not to return deterministic actions.
     :return: the model's action and the next state
         (used in recurrent policies)
     """
    # observation = observation.reshape((-1,) + observation.shape)

    observation = torch.as_tensor(observation).to(model.policy.device)

    actions = model.policy._predict(observation, deterministic=True)
    # print("OUTPUT GRD", actions)
    return actions


def get_latent(model, observation):
    observation = torch.as_tensor(observation).to(model.policy.device)

    with torch.no_grad():
        actor = model.policy.actor
        features = actor.extract_features(observation)
        latent_pi = actor.latent_pi(features)

    return latent_pi


HISTORY_LEN = 10

model_student = SAC('MlpPolicy', BipedalWalkerEnvMedium({"render": False, "hardcore": True,
                                                         "cheat": False, "history_len": HISTORY_LEN}),
                    policy_kwargs=dict(net_arch=[400, 300]), )

net_student = model_student.policy

criterion = nn.MSELoss()

optimizer = optim.Adam(net_student.parameters(), lr=0.0001)

model = SAC.load("out/sac_BipedalWalkerMedium_baseline1_fresh.zip")
model.set_env(env)


N_STEPS = 1000
running_loss = 0
minibatch_counter = 0


for ep in range(5):
    obs = env.reset()
    obs_nocheat = obs[:, :14][0]
    student_obs_queue = deque([obs_nocheat] * HISTORY_LEN, maxlen=HISTORY_LEN)

    for step in range(N_STEPS):
        optimizer.zero_grad()

        student_obs = np.array(list(student_obs_queue))[np.newaxis, ...]

        latent_teacher = get_latent(model, obs).numpy()

        action_teacher, _ = model.predict(obs, deterministic=True)

        action_s = predict_grad(model_student, student_obs)
        latent_student_s = get_latent(model_student, student_obs)

        obs, reward, done, info = env.step(action_s.detach().numpy())


        loss_action = criterion(action_s, torch.as_tensor(action_teacher))
        loss_latent = criterion(latent_student_s, torch.as_tensor(latent_teacher))
        loss = loss_action + loss_latent
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        minibatch_counter += 1
        if minibatch_counter % 10 == 9:  # print every 2000 mini-batches
            print(f'pretrain loss: {running_loss / 2000:.7f}')
            running_loss = 0.0

        env.render()
        # time.sleep(.03)
        if done:
            break

exit()
