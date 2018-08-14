#!/usr/bin/env python3

"""
Pure pursuit
Control the simulator or Duckiebot using a a PID controller (heuristic).

References:
Path Tracking for a Miniature Robot - http://www8.cs.umu.se/kurser/TDBD17/VT06/utdelat/Assignment%20Papers/Path%20Tracking%20for%20a%20Miniature%20Robot.pdf
Implementation of the Pure Pursuit Tracking Algorithm: https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf
"""

import time
import argparse
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import LoggingWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
parser.add_argument('--draw-bbox', default=False)
parser.add_argument('--log-data', default=True)
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = False,
        draw_bbox = args.draw_bbox,
    )
else:
    env = gym.make(args.env_name)

if args.log_data is True:
    env = LoggingWrapper(env)
    print("Data logger is being used!")

obs = env.reset()
env.render()

#TODO: Move function to planning utilities

def se2(pose):
    """
    pose = x - m
           y - m
           angle - rad
    """
    return np.array([[np.cos(pose[2]), -np.sin(pose[2]), pose[0]],
                     [np.sin(pose[2]), np.cos(pose[2]), pose[1]],
                     [0, 0, 1]])

def get_lookahead(traj, dist_r=0.2):
    lookahead = None
    distances = np.zeros((traj.shape[0],))
    for count in range(traj.shape[0]):
        distances[count] = np.linalg.norm(traj[count])

    pair_diff = np.diff(distances, n=1)
    pair_diff = np.append(pair_diff, [0], axis=0)
    dist_r = np.median(distances[pair_diff>0])
    prev = traj[0]
    lo = np.linalg.norm(prev)

    for count in range(1, traj.shape[0]):
        next = traj[count]
        hi = np.linalg.norm(next)

        if (lo <= dist_r) and (dist_r <= hi):
            lambda_interp = (dist_r - lo)/(hi-lo)
            return prev * (1-lambda_interp) + lambda_interp * next

        prev = next
        lo = hi
    assert lookahead is not None, lookahead
    return lookahead


def global2local(global_coord):
    """
    This function transforms global map coordinates into a coordinate system
    centered on the robot.
    It is assumes that the robot operates in standard x, y coordinates.
    x - forward
    y - left
    This is ensured by using function "coordinate_corr"
    input: global_coord - numpy array (3,)
    output: local_coord - numpy array (3,)
    """
    pose = coordinate_corr(env.cur_pos, env.cur_angle)
    M_inv = np.linalg.inv(se2(pose))
    vec_glob = np.array([global_coord[0], global_coord[1], 1])
    local_coord = np.matmul(M_inv, vec_glob)
    return local_coord[:2]


def local2global(local_coord):
    pose = coordinate_corr(env.cur_pos, env.cur_angle)
    M = se2(pose)
    global_coord = np.matmul(M, local_coord)
    return global_coord


def coordinate_corr(zxy, angle):
    return np.array([zxy[2], zxy[0], angle + np.pi/2.0])

def inv_coordinate_corr(xyz):
    return xyz[[1, 2, 0]]

def ratiopositive_x(lookAhead_arr):
    x = lookAhead_arr[0]
    y = lookAhead_arr[1]

    if x > 0:
        assert x > 0, print("x:", x)
        angle = np.arctan2(y, x)
        return np.sin(2*angle) / x

    assert x > 0, print("x:", x)
    return None


if __name__ == '__main__':
    follow_dist = 0.4  # currently not used

    while True:

        # ----------------------------------------------
        # PURE PURSUIT:
        # ----------------------------------------------
        # GET Trajectory
        i, j = env.get_grid_coords(env.cur_pos)
        trajectory = env._get_ref_trajectory(i, j) # global coordinates output

        # Transform trajectory to local coordinates
        canon_traj = np.zeros((trajectory.shape[0], 2))
        for i in range(canon_traj.shape[0]):
            canon_traj[i] = global2local(coordinate_corr(trajectory[i], 0))
        lookahead = get_lookahead(canon_traj, follow_dist)

        # Get steering angle
        #TODO: double check this
        steer_angle = ratiopositive_x(lookahead)
        # ----------------------------------------------
        # omega is determined -> velocity can be determined by the maximum possible commands

        velocity = 0.4
        omega_steer = velocity * steer_angle #np.tan(steer_angle) / env.wheel_dist

        obs, reward, done, info = env.step([velocity, omega_steer])

        env.render()

        if done:
            if reward < 0:
                print('*** FAILED ***')
                if not args.no_pause:
                    time.sleep(0.7)
            obs = env.reset()
            env.render()
