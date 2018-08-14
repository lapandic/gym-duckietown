import math
import numpy as np
import gym
import h5py
import os
from gym import spaces
from datetime import datetime

CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160



class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Turn left
        if action == 0:
            vels = [0.6, +1.0]
        # Turn right
        elif action == 1:
            vels = [0.6, -1.0]
        # Go forward
        elif action == 2:
            vels = [0.7, 0.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

class PyTorchObsWrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)

class LoggingWrapper(gym.Wrapper):
    """
    Logs data while the simulator is running.
    Stored are:
        - The input Image (# Camera image size WIDTH = 160, HEIGHT = 120)
        - The received reward signal (if any)
        - The output commands
        - The pose and velocities of the Duckiebot
        - The ideal rule-based trajectory
    """

    def __init__(self, env):
        super(LoggingWrapper, self).__init__(env)
        self.data_folder = 'recordings'
        self.path = self.data_folder + '/data.hdf5'
        self.chunk_size = 1024
        self.initial_size = 0
        self.tags = {
                     'image': [[self.initial_size, CAMERA_HEIGHT, CAMERA_WIDTH, 3], np.uint8],
                     'reward': [[self.initial_size, 1], np.float32],
                     'actions': [[self.initial_size, 2], np.float32],
                     'position': [[self.initial_size, 3], np.float32],
                     'dir': [[self.initial_size], np.float32],
                     'speed': [[self.initial_size], np.float32],
                     'ref_position': [[self.initial_size, 3], np.float32],
                     'ref_dir': [[self.initial_size], np.float32],
                     # 'time_stamp': [[self.initial_size], np.string_]
                    }


        self.buffer = {k: np.zeros(tuple([self.chunk_size]+v[0][1:]),
                        dtype=v[1]) for (k,v) in self.tags.items()}
        self.buffer_counter = 0

        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        if not os.path.exists(self.path):
            f = h5py.File(self.path, "w")
            f.close()
            with h5py.File(self.path, "a") as f:
                print("self.tags", self.tags)
                for tag, attribute in self.tags.items():
                    print("tag", tag)
                    print("attribute", attribute)
                    maxshape_value = list(attribute[0])
                    maxshape_value[0] = None
                    print("Max shape:", tuple(maxshape_value))
                    f.create_dataset(tag, shape=attribute[0],
                                     dtype=attribute[1],
                                     maxshape=tuple(maxshape_value))


    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        ref_point, tangent = self.env.closest_curve_point(self.env.cur_pos)
        i, j = self.env.get_grid_coords(self.env.cur_pos)
        tile = self.env._get_tile(i, j)
        tile_angle = tile['angle'] * np.pi / 2.0

        input_data = dict(zip(self.tags.keys(), [observation, reward, action,
                          self.env.cur_pos, self.env.cur_angle,
                          self.env.speed, ref_point, tile_angle,
                          datetime.now().strftime('%Y-%m-%d %H:%M:%S')]))
        for tag in self.buffer:
            self.buffer[tag][self.buffer_counter] = input_data[tag]

        self.buffer_counter += 1  # increment to receive next datapoint

        if self.buffer_counter == self.chunk_size:
            self.buffer_counter = 0  # reset counter
            with h5py.File(self.path, mode='a') as f:
                for tag in self.tags:
                    # Resize file and write data to file
                    f[tag].resize(f[tag].shape[0] + self.chunk_size, axis=0)
                    f[tag][-self.chunk_size:] = self.buffer[tag]
                    f.flush()

        return observation, reward, done, info

    #TODO: Are the functions below strictly necessary?
    def get_grid_coords(self, position):
        return self.env.get_grid_coords(position)

    def __getattr__(self, attr):
        if attr == 'cur_pos':
            return self.env.cur_pos
        elif attr == 'cur_angle':
            return self.env.cur_angle
        else:
            AttributeError("Wrong attribute")

    def _get_ref_trajectory(self, i, j):
        return self.env._get_ref_trajectory(i, j)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return observation

