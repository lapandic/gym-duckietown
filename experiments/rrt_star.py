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
from experiments.RRTStar import rrts


# Road tile dimensions (2ft x 2ft, 61cm wide)
ROAD_TILE_SIZE = 0.61

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
parser.add_argument('--draw-bbox', default=False)
parser.add_argument('--log-data', default=False)
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
    # env = LoggingWrapper(env)
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


def inverse_kinematics(env, vel, omega):
    # Distance between the wheels
    baseline = env.unwrapped.wheel_dist

    # assuming same motor constants k for both motors
    k_r = env.k
    k_l = env.k

    # adjusting k by gain and trim
    k_r_inv = (env.gain + env.trim) / k_r
    k_l_inv = (env.gain - env.trim) / k_l

    omega_r = (vel + 0.5 * omega * baseline) / env.radius
    omega_l = (vel - 0.5 * omega * baseline) / env.radius

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv
    return u_r, u_l

def get_obstacles(env):
    """
    Return list of undrivable tiles formated as obstacles
    :param env:
    :return:
    """
    obstacle_list = []

    for j in range(env.grid_height):
        for i in range(env.grid_width):
            if env._get_tile(i,j)["drivable"] == False:
                obstacle_list.append(rrts.Obstacle(get_corners_from_coord(i,j)))

    return obstacle_list

def get_corners_from_coord(i,j):
    """
    Given the tile coordinates in grid return positions of 4 corners of the rectangle in the clockwise order
    :param i:
    :param j:
    :return:
    """
    return [rrts.Point(i*ROAD_TILE_SIZE,j*ROAD_TILE_SIZE),rrts.Point(i*ROAD_TILE_SIZE,(j+1)*ROAD_TILE_SIZE),
            rrts.Point((i+1)*ROAD_TILE_SIZE,(j+1)*ROAD_TILE_SIZE),rrts.Point((i+1)*ROAD_TILE_SIZE,j*ROAD_TILE_SIZE)]




if __name__ == '__main__':
    follow_dist = 0.4  # currently not used

    obstacle_list = get_obstacles(env)

    # ----------------------------------------------
    # RRT Star:
    # ----------------------------------------------
    # GET Trajectory

    goals = get_goal_path()
    for goal in goals:
        x, _, y = env.cur_pos

        sample_area = get_sample_area()

        rrt_star = rrts.RRTStar(start=[x, y], goal=goal,
                  sample_area=sample_area, obstacle_list=obstacle_list)
        path = rrt_star.planner(animation=False)

        
        execute_path(path)

        velocity = 0.3
        omega_steer = velocity * steer_angle

        u_r, u_l = inverse_kinematics(env, velocity, omega_steer) #np.tan(steer_angle) / env.wheel_dist

        obs, reward, done, info = env.step([u_r, u_l])

        env.render()

        if done:
            if reward < 0:
                print('*** FAILED ***')
                if not args.no_pause:
                    time.sleep(0.7)
            obs = env.reset()
            env.render()
