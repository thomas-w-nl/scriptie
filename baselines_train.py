###
### V2
###

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from envs import *


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the name created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, monitor: Monitor, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.monitor = monitor

        print("Model save path:", self.save_path)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            rewards = self.monitor.get_episode_rewards()

            if len(rewards) > 0:
                # Mean training reward over the last 10 episodes
                mean_reward = np.mean(rewards[-10:])
                self.logger.record('reward', mean_reward)
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last episode mean: {mean_reward:.2f}")

                # Log last 100 steps average speed and checkpoints
                avg_speed = np.mean(list(map(lambda x: x["avg_speed"], self.model.ep_info_buffer)))
                avg_checkpoints = np.mean(list(map(lambda x: x["checkpoints"], self.model.ep_info_buffer)))
                print("avg_speed", avg_speed)
                self.logger.record('avg_speed', avg_speed)
                self.logger.record('avg_checkpoints', avg_checkpoints)

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


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

    # env = ANYMalStandupEnv(experiment_conf)

    env = HumanoidEnv(experiment_conf)
    # env = HumanoidEnvGym(experiment_conf)

    # env = InvertedPendulumEnvR(experiment_conf)

    # env = BipedalWalkerEnv(experiment_conf)



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

    model = Algo("MlpPolicy", env, verbose=1, tensorboard_log="./logs/all/",
                 use_sde=True,
                 seed=seed,
                 device="cuda",
                 )

    if config.checkpoint:
        print("Starting from checkpoint", config.checkpoint)
        model.load(config.checkpoint)

    steps = 10_000_000 if WORKSTATION else 200_000

    callback = SaveOnBestTrainingRewardCallback(monitor=env, check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=steps, log_interval=10, tb_log_name=name, callback=callback)
    model.save(name)




    # num_cpu = 4  # 80% gpu usage expected
    # l = []
    # for i in range(num_cpu):
    #     conf = experiment_conf.copy()
    #     conf["render"] = i == 0
    #     l.append(lambda: Env(experiment_conf))
    # env = SubprocVecEnv(l)