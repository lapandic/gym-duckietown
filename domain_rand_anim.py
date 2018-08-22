#!/usr/bin/env python3

import numpy as np
import gym
from gym.wrappers import Monitor
import gym_duckietown

env = gym.make('Duckietown-udem1-v0')

#env = wrappers.Monitor(env, './videos/' + str(time()) + '/', video_callable=lambda x: True)
