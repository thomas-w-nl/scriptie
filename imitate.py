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

MAX_LEN = 10

model_student = SAC('MlpPolicy', BipedalWalkerEnvMedium({"render": False, "hardcore": True, "cheat": False}),
                    policy_kwargs=dict(net_arch=[400, 300]), )

net_student = model_student.policy

criterion = nn.MSELoss()

optimizer = optim.Adam(net_student.parameters(), lr=0.001)

model = SAC.load("out/sac_BipedalWalkerMedium_baseline1_fresh.zip")
model.set_env(env)


dataset = deque(maxlen=10_000)
labels_action = deque(maxlen=10_000)
labels_latent = deque(maxlen=10_000)

running_loss = 0
minibatch_counter = 0

N_STEPS = 1000

for epoch in range(10):
    # episodes
    for episode in range(5):
        obs = env.reset()
        obs_nocheat = obs[:, :14]
        student_obs_queue = deque([obs_nocheat * MAX_LEN], maxlen=MAX_LEN)

        # steps
        for step in range(N_STEPS):



            student_obs = np.array(list(student_obs_queue))[0]

            action, _ = model_student.predict(student_obs, deterministic=True)
            # assert np.allclose(action.cpu().detach().numpy(), action_NOGRAD), f"NOT CLOSE {action} and {action_NOGRAD}"


            latent_student = get_latent(model_student, student_obs)
            latent_teacher = get_latent(model, obs)

            action_teacher, _ = model.predict(obs, deterministic=True)
            # print(action_teacher)


            obs, reward, done, info = env.step(action)

            obs_nocheat = obs[:, :14]
            student_obs_queue.appendleft(obs_nocheat)

            env.render()
            # time.sleep(.03)
            if done:
                break

            dataset.appendleft(student_obs_queue)
            labels_action.appendleft(action_teacher)
            labels_latent.appendleft(latent_teacher)

    th_dataset = torch.as_tensor(dataset)
    th_labels_action = torch.as_tensor(labels_action)
    th_labels_latent = torch.as_tensor(labels_latent)
    # training
    for batch in range(100):
        idx = np.random.random_integers(0, len(dataset), 64)
        # zero the parameter gradients
        optimizer.zero_grad()

        data = th_dataset[idx]
        action_teacher = th_labels_action[idx]
        latent_teacher = th_labels_latent[idx]

        action = predict_grad(model_student, student_obs)
        latent_student = get_latent(model_student, student_obs)

        loss_action = criterion(action, torch.as_tensor(action_teacher))
        loss_latent = criterion(latent_student, latent_teacher)
        loss = loss_action + loss_latent
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        minibatch_counter += 1
        if minibatch_counter % 100 == 99:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, episode + 1, running_loss / 2000))
            running_loss = 0.0
