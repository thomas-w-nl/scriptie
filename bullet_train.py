import os

import pybullet_envs.bullet.minitaur_gym_env as e
import numpy as np

import pybullet



class ExtendedMinitaurBulletEnv(e.MinitaurBulletEnv):
    def step(self, action):
        ob, reward, done, info = super().step(action)



        vel = pybullet.getBaseVelocity(self.minitaur.quadruped)[0] # select linear velocity [x,y,z]
        info.update(dict(avg_speed=np.linalg.norm(vel),
                         checkpoints=-1))

        return ob, reward, done, info


###
### V2
###
import stable_baselines3
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch




def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == '__main__':
    import argparse

    WORKSTATION = not os.getlogin() == "thomas"

    parser = argparse.ArgumentParser(description='Train AntEnv on terrain.')
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    # parser.add_argument('--env', type=str, required=True, help='Env name')
    parser.add_argument('--terrain', type=str, default="flat", help='The name of the terrain for the environment')
    parser.add_argument('--algo', type=str, default="sac", help='The name of the RL algorithm to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint from which to start')
    parser.add_argument('--cheat', action='store_true', help='Train the agent with cheating observations')

    config = parser.parse_args()
    terrain = config.terrain

    render = not WORKSTATION
    print("Rendering:", render)

    hardcore = config.terrain == "hardcore"

    experiment_conf = {"render": render,
                       "terrain": "flat",
                       "hardcore": hardcore,
                       "cheat": config.cheat,
                       "desc": config.name,
                       }


    env = ExtendedMinitaurBulletEnv(render=render)


    # env = HumanoidEnv(experiment_conf)
    # env = HumanoidEnvGym(experiment_conf)

    # env = InvertedPendulumEnvR(experiment_conf)

    # env = OLD_ANYmalEnv(experiment_conf)

    name = f"{config.algo}_{env.__class__.__name__}_{'cheat_' if config.cheat else ''}{terrain}_{config.name}"

    print("File", name)

    # Create log dir
    log_dir = "/tmp/gym/" + name + "/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir, info_keywords=("checkpoints", "avg_speed"))

    # env = DummyVecEnv([lambda:env])
    # env = VecNormalize(env)

    seed = np.random.randint(0, 200)

    print("Seed:", seed)

    if config.algo == "sac":
        Algo = SAC
    elif config.algo == "ppo":
        Algo = PPO
    else:
        raise NotImplementedError(f"Unknown {config.algo}")

    sched = linear_schedule(2.5e-4)
    clip_sched = linear_schedule(.2)

    # Hardcore BipedalWalker Tuned PPO
    # model = Algo("MlpPolicy", env, verbose=1, tensorboard_log="./logs/all/",
    #             n_steps=2048,
    #             batch_size=64,
    #             gamma=0.99,
    #             n_epochs=10,
    #             ent_coef=0.001,
    #             learning_rate=sched,
    #             clip_range=.2
    #             )

    model = Algo("MlpPolicy", env, verbose=1, tensorboard_log="./logs/all/",
                 # use_sde=True,
                 seed=seed,
                 device="cuda",
                 ent_coef=0.005,
                 tau=0.01,
                 learning_starts=100,
                 learning_rate=sched,
                 )



    if config.checkpoint:
        # --checkpoint pretrained/anymal_base.zip
        print("Starting from checkpoint", config.checkpoint)
        model = SAC.load(config.checkpoint)
        model.set_env(env)

    steps = 1_000_000 if WORKSTATION else 200_000

    # callback = SaveOnBestTrainingRewardCallback(monitor=env, check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=steps, log_interval=10, tb_log_name=name)
    model.save(name)

    # num_cpu = 4  # 80% gpu usage expected
    # l = []
    # for i in range(num_cpu):
    #     conf = experiment_conf.copy()
    #     conf["render"] = i == 0
    #     l.append(lambda: Env(experiment_conf))
    # env = SubprocVecEnv(l)
