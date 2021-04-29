###
### V2
###

import os
import random
import socket
from abc import ABC

from PIL import Image
from noise import pnoise2
import gym

hostname = socket.gethostname()
if hostname == "robolabws4":
    MujocoEnv = ABC
else:
    import mujoco_py
    from mujoco_py.generated import const
    from gym.envs.mujoco import MujocoEnv


from scipy.spatial.transform import Rotation as R
from scipy.signal import convolve2d
import numpy as np
from gym import utils

WORKSTATION = not os.getlogin() == "thomas"


def euler2mat(euler, degrees=True):
    r = R.from_euler('xyz', euler, degrees=degrees)
    return r.as_matrix()


def normalize(v):
    return np.array(v) / np.linalg.norm(v)


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


class MujocoEnvFromString(MujocoEnv):
    """
    Build MuJoCo environment from a string containing the xml instead of a name.
    """

    def __init__(self, xml, frame_skip):
        self.model = mujoco_py.load_model_from_xml(xml)

        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self.frame_skip = frame_skip

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def step(self, action):
        # Note: Step is completely reimplemented in BaseExperimentEnv due to modifications
        raise NotImplementedError("This method should be overridden by BaseExperimentEnv")

    def reset_model(self):
        # TODO is this corrrect random initialization?
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()


class TerrainEnv(MujocoEnvFromString):
    """
    Base class for models that have multiple terrains to train on.
    """

    def __init__(self, config):

        self.terrain = config["terrain"]
        self.model_file = config["model_file"]
        self.model_start_height = config["model_start_height"]

        self.terrain_seed = config.get("terrain_seed", 0)
        self.octaves = config.get("terrain_octaves", 10)
        self.slope_angle = config.get("slope_angle", 10)
        self.hfield_height = config.get("hfield_height", 5)
        self.do_render = config.get("render", False)

        # TODO Clean up path system
        if WORKSTATION:
            print("warn: NotImplementedError(Fix path system)")
            root = "/home/twiggers/walking"
        else:
            root = "/home/thomas/uva/scriptie/versie 3"

        mean_height_start = self.generate_terrain(root)

        # TODO this is not really the place to load xml but its tightly coupled anyway to terrain
        with open(root + '/worlds/' + self.model_file, 'r') as f:
            xml = f.read()
        hfield_height = self.hfield_height

        hfield_xoffset = 0

        if self.terrain == "steps":
            hfield_height = .3  # TODO HARDCOOOODE

        xml = xml.replace("{{START_HEIGHT}}", str(mean_height_start * hfield_height + self.model_start_height))
        xml = xml.replace("{{HFIELD_HEIGHT}}", str(hfield_height))
        xml = xml.replace("{{HFIELD_XOFFSET}}", str(hfield_xoffset))

        MujocoEnvFromString.__init__(self, xml, config["frame_skip"])
        utils.EzPickle.__init__(self)

        # Does not work
        # data = np.array(data).ravel()
        # data = np.interp(data, (data.min(), data.max()), (0, 1))  # rescale to proper input range [0, 1] for hfield
        # assert data.min() >= 0 and data.max() <= 1
        # self.sim.model.hfield_data[:] = data

    def clear_patch(self, hfield):
        ''' Clears a patch shaped like box, assuming robot is placed in center of hfield
        '''

        start_area_size = int(0.02 * hfield.shape[0])

        # clear patch
        h_center = int(0.5 * hfield.shape[0])
        w_center = int(0.5 * hfield.shape[1])
        fromrow, torow = w_center - start_area_size, w_center + start_area_size
        fromcol, tocol = h_center - start_area_size, h_center + start_area_size

        mean_height = np.mean(hfield[fromrow:torow, fromcol:tocol])

        hfield[fromrow:torow, fromcol:tocol] = mean_height

        # convolve to smoothen edges somewhat, in case hills were cut off
        K = np.ones((32, 32)) / (32 * 32)
        s = convolve2d(hfield[fromrow - 9:torow + 9, fromcol - 9:tocol + 9], K, mode='same', boundary='symm')
        hfield[fromrow - 9:torow + 9, fromcol - 9:tocol + 9] = s

        return hfield, mean_height

    def generate_terrain(self, file_root):

        map_width = map_length = 1000

        if self.terrain == "hills" or self.terrain == "hills_steps":

            x = np.linspace(0, 1, map_width)
            y = np.linspace(0, 1, map_length)

            # offset the sample area by 'seed' map area's. Not the prettiest solution
            x += self.terrain_seed * map_length
            y += self.terrain_seed * map_length

            print("Generating perlin ", end="")
            data = np.array([[pnoise2(i, j, octaves=self.octaves) for j in x] for i in y])
            print("done")

            if self.terrain == "hills_steps":
                data *= .1  # shrink range, when scaled by mujoco detail is lost and a step like hfield is created

        elif self.terrain == "steps":
            n_steps = 200
            steps = np.random.random((n_steps, n_steps))
            data = np.array(Image.fromarray(steps).resize((map_width, map_length), Image.NEAREST))

        elif self.terrain == "stairs":
            step_width = 10
            step_height = .4
            n_steps = 30

            start = map_width // 2 + int(map_width * .1)
            trace = np.zeros(map_width)
            trace[start:start + n_steps * step_width] = np.repeat(np.linspace(0, 1, n_steps), step_width) * step_height
            steps = np.expand_dims(trace, 0)
            data = np.array(Image.fromarray(steps).resize((map_width, map_length), Image.NEAREST))

        elif self.terrain == "flat" or self.terrain is None:
            data = np.zeros((map_width, map_length))
        elif self.terrain == "slope":
            # TODO
            raise NotImplementedError()
        else:
            raise NotImplementedError("Unkown terrain", self.terrain)

        if np.max(data) > 0:
            data = np.interp(data, (np.min(data), np.max(data)), (0, 255))

        data, mean_height_start = self.clear_patch(data)

        img = Image.fromarray(data).convert("RGB")
        # img.save(file_root + '/worlds/' + "generated_terrain.png")
        img.save("/tmp/generated_terrain.png")

        return mean_height_start / 255


class BaseExperimentEnv(TerrainEnv):
    """
    Base class for models that expects them to have a torso, follow checkpoints and measure velocity.
    """

    def __init__(self, config):
        self.frame_skip = config["frame_skip"] = 5  # TODO WHY IS THIS?
        self.desc = config["desc"]
        self.cheat = config.get("cheat", False)

        self.num_targets_reached = 0
        self.target_checkpoint_range = 3

        self.velocities = []

        self.targets = [
            [30, 0, 0],
            [30, 30, 0],
            [-30, 30, 0],
            [-30, -30, 0],
            [30, -30, 0],
        ]

        self.reset_targets()

        super().__init__(config)

    def get_name(self):
        c = "_cheat" if self.cheat else ""
        return f"{self.__class__.__name__}{c}_{self.desc}"

    def reset_targets(self):
        self.target = random.choice(self.targets)
        self.num_targets_reached = 0

    def reset(self):
        print(f"Avg vel: {np.mean(self.velocities)}, num checkpoints: {self.num_targets_reached}")

        self.reset_targets()
        self.velocities = []
        return super().reset()

    def next_target(self):
        self.target = random.choice(self.targets)

    def get_target_vector(self):
        pos = self.get_body_com("torso")
        target_vec = np.array(self.target) - pos

        return normalize(target_vec)

    def update_target(self):
        pos = self.get_body_com("torso")
        target_vec = np.array(self.target) - pos

        # check if in range, ignoring z axis
        if np.linalg.norm(target_vec[:2]) < self.target_checkpoint_range:
            self.next_target()
            self.num_targets_reached += 1

            return True
        return False

    def get_body_vector(self):
        fwd_vec = np.array([1, 0, 0])
        body_mat = self.data.get_body_xmat("torso")
        return body_mat @ fwd_vec

    def compute_reward(self, action):
        vel_vec = self.get_velocity()

        reached = self.update_target()

        target_reached_reward = 1000 if reached else 0

        target_vec = self.get_target_vector()
        body_vec = self.get_body_vector()

        dir = np.dot(target_vec[:2], body_vec[:2])

        direction_reward = dir

        speed_reward = np.linalg.norm(vel_vec) * dir

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        survive_reward = .01
        reward = direction_reward + speed_reward - ctrl_cost - contact_cost + survive_reward + target_reached_reward

        reward_info = dict(
            speed_reward=speed_reward,
            direction_reward=direction_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

        return reward, reward_info

    def show_target_vector(self):
        pos = self.get_body_com("torso")
        body_mat = self.data.get_body_xmat("torso")
        viewer = self._get_viewer("human")

        target_vec = self.get_target_vector()

        up_vec = np.array([0, 0, 1])

        # Show vector towards target
        mat = rotation_matrix_from_vectors(up_vec, target_vec)
        viewer.add_marker(pos=pos + np.array([0, 0, 1]),  # position of the arrow
                          size=np.array([0.05, 0.05, 2.4]),  # size of the arrow
                          mat=mat,  # orientation as a matrix
                          rgba=np.array([1., 1., 1., 1.]),  # color of the arrow
                          type=const.GEOM_ARROW,
                          label="")

        # Show forward facing vector
        rot_forward = euler2mat([0, 90, 0])
        viewer.add_marker(pos=pos + np.array([0, 0, 1]),  # position of the arrow
                          size=np.array([0.05, 0.05, 2.4]),  # size of the arrow
                          mat=body_mat @ rot_forward,  # orientation as a matrix
                          rgba=np.array([1., 1., 1., .5]),  # color of the arrow
                          type=const.GEOM_ARROW,
                          label="")

    def _get_obs(self):
        if self.cheat:
            return self._get_cheat_obs()
        else:
            return self._get_default_obs()

    def _get_default_obs(self):
        target_vec = self.get_target_vector()
        body_vec = self.get_body_vector()

        # global z pos of torso is also stripped
        # Maybe include torso range sensor
        return np.concatenate([
            self.sim.data.qpos.flat[3:],
            self.sim.data.qvel.flat,
            body_vec,
            target_vec,
        ])

    def _get_cheat_obs(self):
        return np.concatenate([
            self._get_default_obs(),
            self.sim.data.cfrc_ext.flat,
        ])

    def get_velocity(self):
        # Relies on velocimeter being the first sensor
        # This is nasty but there really is no beter way than to know sensor offset
        return self.data.sensordata[:3]

    def step(self, a):


        self.do_simulation(a, self.frame_skip)
        # Limit joint velocity to prevent clipping, but do not limit body speed
        # self.sim.data.qvel[3:] = np.clip(self.sim.data.qvel[3:], -1, 1)

        ob = self._get_obs()

        # Keep track of robot velocity during episode
        vel = np.linalg.norm(self.get_velocity())
        self.velocities.append(vel)

        info = dict(avg_speed=np.mean(self.velocities),
                    checkpoints=self.num_targets_reached)

        reward, reward_info = self.compute_reward(a)
        info.update(reward_info)

        done = False
        # limt length to 60 seconds
        duration = self.sim.data.time
        if duration > 60:
            done = True

        if self.do_render:
            self.render()

        return ob, reward, done, info

    def render(self, **kwargs):
        self.show_target_vector()
        if self.cheat:
            self.show_sensors()
        return super().render(**kwargs)

    def show_sensors(self):
        pass

    #
    #     for i, leg in enumerate(["rangefinder_frontleft",
    #                              "rangefinder_frontright",
    #                              "rangefinder_backright",
    #                              "rangefinder_backleft"]):
    #
    #         for s in range(1, 5):
    #             name = f"{leg}{s}"
    #
    #             sensor_pos = self.data.get_site_xpos(name)
    #             sensor_mat = self.data.get_site_xmat(name)
    #             # self.show_axis(sensor_pos, sensor_mat)
    #
    #             viewer = self._get_viewer("human")
    #
    #             distance = self.sim.data.get_sensor("sensor_" + name)
    #             # distance = self.get_cheats()[i]
    #
    #             # Show forward facing vector
    #             rot_forward = euler2mat([0, 0, 0])
    #             viewer.add_marker(pos=sensor_pos,  # position of the arrow
    #                               size=np.array([0.01, 0.01, distance]),  # size of the arrow
    #                               mat=sensor_mat @ rot_forward,  # orientation as a matrix
    #                               rgba=np.array([0, 0, 1., .5]),  # color of the arrow
    #                               type=const.GEOM_ARROW1,
    #                               label="")

    def show_vector(self, xpos, xmat, size=np.array([.05, .05, 2])):
        viewer = self._get_viewer("human")

        viewer.add_marker(pos=xpos,  # position of the arrow
                          size=size,  # size of the arrow
                          mat=xmat,  # orientation as a matrix
                          rgba=np.array([0, 0, 1, .5]),  # color of the arrow
                          type=const.GEOM_ARROW1,
                          label="")

    def show_axis(self, xpos, xmat):
        viewer = self._get_viewer("human")

        # Z - Blue
        rot_forward = euler2mat([0, 0, 0])
        viewer.add_marker(pos=xpos,  # position of the arrow
                          size=np.array([0.01, 0.01, 2]),  # size of the arrow
                          mat=xmat @ rot_forward,  # orientation as a matrix
                          rgba=np.array([0, 0, 1, .5]),  # color of the arrow
                          type=const.GEOM_ARROW1,
                          label="")

        # Y - Green
        rot_forward = euler2mat([0, 90, 0])
        viewer.add_marker(pos=xpos,  # position of the arrow
                          size=np.array([0.01, 0.01, 2]),  # size of the arrow
                          mat=xmat @ rot_forward,  # orientation as a matrix
                          rgba=np.array([0, 1, 0, .5]),  # color of the arrow
                          type=const.GEOM_ARROW1,
                          label="")

        rot_forward = euler2mat([90, 0, 0])
        viewer.add_marker(pos=xpos,  # position of the arrow
                          size=np.array([0.01, 0.01, 2]),  # size of the arrow
                          mat=xmat @ rot_forward,  # orientation as a matrix
                          rgba=np.array([1, 0, 0, .5]),  # color of the arrow
                          type=const.GEOM_ARROW1,
                          label="")


class RandomForceEnv(BaseExperimentEnv):

    def __init__(self, config):
        self.force_norm = config["random_force_norm"]
        self.force = np.array([0, 0, 0])
        self.randomforce()
        super(RandomForceEnv, self).__init__(config)

    def randomforce(self):
        dir = np.random.uniform(-np.pi * 2, np.pi * 2)
        self.force = np.array([np.sin(dir), np.cos(dir), 0]) * self.force_norm

    def _get_cheat_obs(self):
        x = super(RandomForceEnv, self)._get_cheat_obs()
        return np.concatenate([x, self.force])

    def step(self, a):
        pos = self.data.get_body_xpos("torso")
        i = self.model.body_name2id("torso")

        if self.data.time % 5 < 0.001:
            self.randomforce()

        self.data.xfrc_applied[i, :3] = self.force

        mat = rotation_matrix_from_vectors([0, 0, 1], self.force)
        if self.do_render:
            disp_vec = np.array([.05, .05, 2])
            self.show_vector(pos, mat, disp_vec)
        return super(RandomForceEnv, self).step(a)
