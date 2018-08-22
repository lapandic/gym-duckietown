#!/usr/bin/env python3

import math
import time
import numpy as np
import gym
from gym.wrappers import Monitor
import gym_duckietown

env = gym.make('Duckietown-udem1-v0')

#env = wrappers.Monitor(env, './videos/' + str(time()) + '/', video_callable=lambda x: True)


while True:

    env.reset()
    env.cur_pos = np.array([0.84, 0, 3.31])
    env.cur_angle = 50 * math.pi / 180

    env.render('human')
    time.sleep(1)
