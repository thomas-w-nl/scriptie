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

dataset = deque(maxlen=10_000)
labels_action = deque(maxlen=10_000)
labels_latent = deque(maxlen=10_000)

N_STEPS = 1000
running_loss = 0
minibatch_counter = 0


for ep in range(5):
    obs = env.reset()
    obs_nocheat = obs[:, :14][0]
    student_obs_queue = deque([obs_nocheat] * HISTORY_LEN, maxlen=HISTORY_LEN)

    for step in range(N_STEPS):

        student_obs = np.array(list(student_obs_queue))[np.newaxis, ...]

        latent_teacher = get_latent(model, obs).numpy()

        action_teacher, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action_teacher)

        env.render()
        # time.sleep(.03)
        if done:
            break

        dataset.appendleft(student_obs_queue)
        labels_action.appendleft(action_teacher)
        labels_latent.appendleft(latent_teacher)

        obs_nocheat = obs[:, :14][0]
        student_obs_queue.appendleft(obs_nocheat)

th_dataset = torch.as_tensor(dataset)
th_labels_action = torch.as_tensor(labels_action)
th_labels_latent = torch.as_tensor(labels_latent)

# print("th_dataset", th_dataset.shape)
# print("th_labels_action", th_labels_action.shape)
# training
print("Training for", len(th_dataset) // 64, "batches")
print("dataset size", len(th_dataset))
for epoch in range(10):
    for idx in DataLoader(range(len(th_dataset)), shuffle=True, batch_size=64):
        # print(idx)
        # zero the parameter gradients
        optimizer.zero_grad()

        data = th_dataset[idx]
        action_teacher = th_labels_action[idx]
        latent_teacher = th_labels_latent[idx]

        action = []
        latent_student = []
        for sample in data:
            # print("data", data.shape)
            # print("sample", sample.shape)
            action_s = predict_grad(model_student, sample.unsqueeze(0))
            latent_student_s = get_latent(model_student, sample.unsqueeze(0))
            action.append(action_s)
            latent_student.append(latent_student_s)

        # print(action)
        action = torch.stack(action)
        latent_student = torch.stack(latent_student)

        loss_action = criterion(action, torch.as_tensor(action_teacher))
        loss_latent = criterion(latent_student, latent_teacher)
        loss = loss_action + loss_latent
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        minibatch_counter += 1
        if minibatch_counter % 10 == 9:  # print every 2000 mini-batches
            print(f'pretrain loss: {running_loss / 2000:.7f}')
            running_loss = 0.0

running_loss = 0
minibatch_counter = 0



for epoch in range(1000):
    # episodes
    for episode in range(1):
        obs = env.reset()
        obs_nocheat = obs[:, :14][0]
        student_obs_queue = deque([obs_nocheat] * HISTORY_LEN, maxlen=HISTORY_LEN)

        # steps
        for step in range(N_STEPS):

            student_obs = np.array(list(student_obs_queue))[np.newaxis, ...]
            # print("student_obs", student_obs.shape)

            action, _ = model_student.predict(student_obs, deterministic=True)
            # assert np.allclose(action.cpu().detach().numpy(), action_NOGRAD), f"NOT CLOSE {action} and {action_NOGRAD}"

            # print("obs", obs.shape)
            latent_student = get_latent(model_student, student_obs)
            latent_teacher = get_latent(model, obs).numpy()

            action_teacher, _ = model.predict(obs, deterministic=True)
            # print("action_teacher", action_teacher.shape)

            # print("action", action)
            obs, reward, done, info = env.step(action)

            env.render()
            # time.sleep(.03)
            if done:
                break

            dataset.appendleft(student_obs_queue)
            labels_action.appendleft(action_teacher)
            labels_latent.appendleft(latent_teacher)

            obs_nocheat = obs[:, :14][0]
            student_obs_queue.appendleft(obs_nocheat)

    th_dataset = torch.as_tensor(dataset)
    th_labels_action = torch.as_tensor(labels_action)
    th_labels_latent = torch.as_tensor(labels_latent)

    # print("th_dataset", th_dataset.shape)
    # print("th_labels_action", th_labels_action.shape)
    # training
    print("Training for", len(th_dataset) // 64, "batches")
    print("dataset size", len(th_dataset))
    for idx in DataLoader(range(len(th_dataset)), shuffle=True, batch_size=64):
        # print(idx)
        # zero the parameter gradients
        optimizer.zero_grad()

        data = th_dataset[idx]
        action_teacher = th_labels_action[idx]
        latent_teacher = th_labels_latent[idx]

        action = []
        latent_student = []
        for sample in data:
            # print("data", data.shape)
            # print("sample", sample.shape)
            action_s = predict_grad(model_student, sample.unsqueeze(0))
            latent_student_s = get_latent(model_student, sample.unsqueeze(0))
            action.append(action_s)
            latent_student.append(latent_student_s)

        # print(action)
        action = torch.stack(action)
        latent_student = torch.stack(latent_student)

        loss_action = criterion(action, torch.as_tensor(action_teacher))
        loss_latent = criterion(latent_student, latent_teacher)
        loss = loss_action + loss_latent
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        minibatch_counter += 1
        if minibatch_counter % 10 == 9:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.7f' %
                  (epoch + 1, episode + 1, running_loss / 2000))
            running_loss = 0.0
