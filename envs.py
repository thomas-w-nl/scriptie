###
### V2
###

import gym.envs.box2d as box2d


import socket
hostname = socket.gethostname()
if hostname == "robolabws4":
    pass
else:
    from gym.envs.mujoco import *
    from gym.envs.mujoco.humanoid import mass_center
    from gym.envs.mujoco import HumanoidEnv as HumanoidEnvORIGINAL

from base_envs import *

WORKSTATION = not os.getlogin() == "thomas"

"""
TODO
touch sensors for cheating agent

rangefinder sensor for height above ground for ant?




"""


class BipedalWalkerEnv(gym.envs.box2d.BipedalWalker):

    def __init__(self, config):
        self.velocities = []
        self.do_render = config.get("render", False)
        self.cheat = config.get("cheat", False)
        super().__init__()
        self.hardcore = config.get("hardcore", False)

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

        # clip lidar for non cheating agent
        if not self.cheat:
            ob = ob[:14]
        return ob, reward, done, info

    def reset(self):
        self.velocities = []
        return super().reset()





class ANYMalStandupEnv(BaseExperimentEnv):
    def __init__(self, config):
        # config["model_file"] = "anymal.xml"
        # self.start_height = .6

        config["model_file"] = "anymal_flat.xml"
        self.start_height = .13

        # config["model_file"] = "anymal_servo.xml"
        # self.start_height = .6

        config["model_start_height"] = self.start_height
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.model.nu,), dtype=np.float32)

        super().__init__(config)


    def step(self, a):
        ob, reward, done, info = super().step(a)

        reward = zpos = self.get_body_com("torso")[2]

        # Too tilted over
        xmat = self.data.get_body_xmat("torso")
        # print(xmat[-1, -1])
        if xmat[-1, -1] < .5:
            print("tilt")
            done = True
            reward = -50
            return ob, reward, done, info

        # Too low
        # if zpos < .1:
        #     print("height")
        #     done = True
        #     reward = -50
        #     return ob, reward, done, info

        # Haarnoja reward resacle
        reward *= 10

        return ob, reward, done, info

    def get_height_above_ground(self):
        return self.data.sensordata[9]

    def compute_reward(self, action):
        zpos = self.get_body_com("torso")[2]
        return zpos, dict()

    def reset(self):
        x = super().reset()
        qpos_init = np.array(
            [1.57135e-06, -3.20295e-06, self.start_height, 1, 3.78503e-06, -1.6587e-07, 3.16825e-06, 1.16648,
             -0.000206107, -0.036613, -1.16587, -0.000139531, -0.0445421, 1.16624, 0.000126223, 0.0397481, -1.16597,
             0.000204427, 0.0432443])
        self.sim.data.qpos[:] = qpos_init
        return x



class HumanoidEnv(BaseExperimentEnv):
    def __init__(self, config):
        config["model_file"] = "humanoid.xml"
        config["model_start_height"] = 1.4
        super().__init__(config)

    def step(self, a):
        obs, reward, done, info = super().step(a)

        vel_vec = self.get_velocity()

        # reached = self.update_target()
        #
        # target_reached_reward = 1000 if reached else 0
        #
        # target_vec = self.get_target_vector()
        # body_vec = self.get_body_vector()
        # dir = np.dot(target_vec[:2], body_vec[:2])
        # direction_reward = dir
        # speed_reward = np.linalg.norm(vel_vec) * dir

        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * np.linalg.norm(vel_vec)

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        n = dict(reward_linvel=lin_vel_cost,
                 reward_quadctrl=-quad_ctrl_cost,
                 reward_alive=alive_bonus,
                 reward_impact=-quad_impact_cost)
        info.update(n)

        return obs, reward, done, info



########################### TODO REWORK
#
#
# class InvertedPendulumEnvR(InvertedPendulumEnv):
#     def __init__(self, config):
#         self.do_render = config.get("render", False)
#         self.cheat = config.get("cheat", False)
#         self.perturb_magnitude = config.get("perturb_magnitude", 1)
#         self.perturbation = None
#
#         super().__init__()
#
#     def _get_obs(self):
#         ob = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
#         if self.cheat:
#             return np.concatenate([ob, [self.perturbation]])
#         else:
#             return ob
#
#     def step(self, a):
#         if self.data.time % 1 < 0.001:
#             self.perturbation = np.random.uniform(size=None, low=-.1, high=.1) * self.perturb_magnitude
#             # print(self.perturbation)
#
#         self.data.qvel[1] += self.perturbation
#
#         ob, reward, done, info = super().step(a)
#
#         info["checkpoints"] = 0
#         info["avg_speed"] = 0
#         if self.do_render:
#             self.render()
#
#         # limt length to 10 seconds
#         duration = self.sim.data.time
#         if duration > 10:
#             done = True
#
#         return ob, reward, done, info


class AntEnv(BaseExperimentEnv):
    def __init__(self, config):
        config["model_file"] = "ant.xml"
        config["model_start_height"] = .75
        super().__init__(config)

    def step(self, a):
        ob, reward, done, info = super().step(a)

        xmat = self.data.get_body_xmat("torso")
        if xmat[-1, -1] < -.7:
            # flipped over
            done = True
            reward = -5

        return ob, reward, done, info


class AntRandomForceEnv(RandomForceEnv):
    def __init__(self, config):
        config["model_file"] = "ant.xml"
        config["model_start_height"] = .75
        super().__init__(config)

    def step(self, a):
        ob, reward, done, info = super().step(a)

        xmat = self.data.get_body_xmat("torso")
        if xmat[-1, -1] < -.7:
            # flipped over
            done = True
            reward = -5

        return ob, reward, done, info

    def _get_cheat_obs(self):
        cheat_ob = np.concatenate([
            self._get_default_obs(),
            self.force
        ])
        return cheat_ob


class ANYMalEnv(BaseExperimentEnv):
    def __init__(self, config):
        config["model_file"] = "anymal.xml"
        config["model_start_height"] = .6
        super().__init__(config)

    def step(self, a):
        ob, reward, done, info = super().step(a)

        # Too tilted over
        xmat = self.data.get_body_xmat("torso")
        # print(xmat[-1, -1])
        if xmat[-1, -1] < .5:
            print("tilt")
            done = True
            reward = -50
            return ob, reward, done, info

        # Too low
        zpos = self.get_body_com("torso")[2]
        if zpos < .1:
            print("height")
            done = True
            reward = -50
            return ob, reward, done, info

        return ob, reward, done, info

    def get_height_above_ground(self):
        return self.data.sensordata[9]

    # def compute_reward(self, action):
    #     vel_vec = self.get_velocity()
    #
    #     reached = self.update_target()
    #
    #     target_reached_reward = 1000 if reached else 0
    #
    #     target_vec = self.get_target_vector()
    #     body_vec = self.get_body_vector()
    #
    #     dir = np.dot(target_vec[:2], body_vec[:2])
    #
    #     direction_reward = dir
    #
    #     speed_reward = np.linalg.norm(vel_vec) * dir * 5
    #
    #     ctrl_cost = .5 * np.square(action).sum()
    #     contact_cost = 0.5 * 1e-3 * np.sum(
    #         np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
    #
    #     height = self.get_height_above_ground()
    #
    #     ### Reward shaping
    #     # direction_reward estimate [-1, 1]
    #     # speed_reward estimate [-6, 6]
    #     # ctrl_cost estimate [-10, 0]
    #     # contact_cost estimate [-6, 0]
    #     # survive_reward estimate [-6, 0]
    #
    #     survive_reward = .01
    #     reward = direction_reward + speed_reward - ctrl_cost - contact_cost + survive_reward + target_reached_reward + height
    #
    #     reward_info = dict(
    #         speed_reward=speed_reward,
    #         direction_reward=direction_reward,
    #         reward_ctrl=-ctrl_cost,
    #         reward_contact=-contact_cost,
    #         reward_survive=survive_reward)
    #
    #     for k,v in reward_info.items():
    #         print(f"{k}:", v)
    #     return reward, reward_info

    def compute_reward(self, action):
        vel_vec = self.get_velocity()

        reached = self.update_target()

        target_reached_reward = 1000 if reached else 0

        target_vec = self.get_target_vector()
        body_vec = self.get_body_vector()

        dir = np.dot(target_vec[:2], body_vec[:2])

        direction_reward = dir

        speed_reward = vel_vec[0] * 5

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = .5e-6 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        height = self.get_height_above_ground()

        ### Reward shaping
        # direction_reward estimate [-1, 1]
        # speed_reward estimate [-6, 6]
        # ctrl_cost estimate [-10, 0]
        # contact_cost estimate [-6, 0]
        # survive_reward estimate [-6, 0]

        survive_reward = 5
        reward = speed_reward - ctrl_cost - contact_cost + survive_reward + target_reached_reward

        reward_info = dict(
            speed_reward=speed_reward,
            direction_reward=direction_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

        return reward, reward_info



class ANYMalRandomForceEnv(RandomForceEnv):
    def __init__(self, config):
        config["model_file"] = "anymal.xml"
        config["model_start_height"] = .6
        config["random_force_norm"] = 100
        super().__init__(config=config)

    def step(self, a):
        ob, reward, done, info = super().step(a)

        # Too tilted over
        xmat = self.data.get_body_xmat("torso")
        if xmat[-1, -1] < .9:
            done = True
            reward = -5
            return ob, reward, done, info

        # Too low
        zpos = self.get_body_com("torso")[2]
        if zpos < .4:
            done = True
            reward = -5
            return ob, reward, done, info

        return ob, reward, done, info


class ANYMalEnvBASIC(BaseExperimentEnv):
    def __init__(self, config):
        config["model_file"] = "anymal.xml"
        config["model_start_height"] = .6
        self.counter = 0
        super().__init__(config=config)

    def step(self, a):
        ob, reward, done, info = super().step(a)

        xvel = self.sim.data.sensordata[0]
        self.velocities.append(xvel)
        reward = xvel
        info = dict(avg_speed=xvel,
                    checkpoints=-1)

        zpos = self.get_body_com("torso")[2]

        # print("zpos", zpos)
        done = zpos < .35

        self.counter += 1
        if self.counter > 10_000:
            print("ping")
            done = True

        if self.do_render:
            self.render()

        return ob, reward, done, info

    def reset(self):
        self.counter = 0
        return super().reset()


class ANYMalRandomForceCheatEnv(ANYMalRandomForceEnv):

    def _get_obs(self):
        ob = super()._get_obs()
        cheat_ob = np.concatenate([
            ob,
            self.force
        ])
        return cheat_ob


class NaoEnv(BaseExperimentEnv):
    def __init__(self, config):
        config["model_file"] = "nao.xml"
        config["model_start_height"] = .35
        super().__init__(config)

    def step(self, a):
        obs, reward, done, info = super().step(a)

        zpos = self.get_body_com("torso")[2]
        done = bool((zpos < .23) or (zpos > 2.0))

        info.update(dict(avg_speed=-1,
                         checkpoints=-1))

        return obs, reward, done, info



#
# class HumanoidEnvGym(HumanoidEnvORIGINAL):
#
#     def __init__(self, config):
#         self.velocities = []
#         self.do_render = config.get("render", False)
#         super().__init__()
#
#     def step(self, a):
#         pos_before = mass_center(self.model, self.sim)
#         ob, reward, done, info = super().step(a)
#         pos_after = mass_center(self.model, self.sim)
#
#         vel = (pos_after - pos_before) / self.dt
#
#         self.velocities.append(vel)
#
#         info.update(dict(avg_speed=np.mean(self.velocities),
#                          checkpoints=-1))
#
#         done = False
#         # limt length to 60 seconds
#         duration = self.sim.data.time
#         if duration > 60:
#             done = True
#
#         if self.do_render:
#             self.render()
#
#         # Haarnoja et al reward scale
#         reward *= 10
#
#         return ob, reward, done, info
#
#     def reset(self):
#         self.velocities = []
#         return super().reset()

