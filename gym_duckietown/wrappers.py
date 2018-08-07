import math
import numpy as np
import gym
import h5py
import os
from gym import spaces

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

    def __init__(self, env=None):
        super().__init__(env)
        self.data_folder = 'recordings'
        self.path = self.data_folder + '/data.hdf5'
        self.chunk_size = 1024
        self.initial_size = 0
        self.tags = {
                    'Images': [(self.initial_size, 120, 160, 3), 'i1'],
                     'Reward': [(self.initial_size, 1), 'float'],
                     'Output': [(self.initial_size, 2), 'float'],
                     'Position': [(self.initial_size, 3), 'float'],
                     'Angle': [(self.initial_size,), 'float'],
                     'Velocity': [(self.initial_size,), 'float'],
                     'Ref-Position': [(self.initial_size, 3), 'float']}
        self.buffer = {
                     'Images': np.zeros((self.chunk_size, 120, 160, 3),
                        dtype=np.dtype('i1')),
                     'Reward': np.zeros((self.chunk_size, 1),
                        dtype='float'),
                     'Output': np.zeros((self.chunk_size, 2),
                        dtype='float'),
                     'Position': np.zeros((self.chunk_size, 3),
                        dtype='float'),
                     'Angle': np.zeros((self.chunk_size,),
                        dtype='float'),
                     'Velocity': np.zeros((self.chunk_size,),
                        dtype='float'),
                     'Ref-Position': np.zeros((self.chunk_size, 3),
                        dtype='float')}
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
        point, tangent = self.env.closest_curve_point(self.env.cur_pos)

        input_data = dict(zip(self.tags.keys(), [observation, reward, action,
                          self.env.cur_pos, self.env.cur_angle,
                          self.env.speed, point]))
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
