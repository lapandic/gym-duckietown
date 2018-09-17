#!/usr/bin/env python3

"""
Pure pursuit
Control the simulator or Duckiebot using a a PID controller (heuristic).

References:
Path Tracking for a Miniature Robot - http://www8.cs.umu.se/kurser/TDBD17/VT06/utdelat/Assignment%20Papers/Path%20Tracking%20for%20a%20Miniature%20Robot.pdf
Implementation of the Pure Pursuit Tracking Algorithm: https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf
"""
import sys
sys.path.append('../')

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
                obstacle_list.append(rrts.Obstacle(get_corners_from_coord(j,i)))

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


def get_goal(step_forward = 0.5,step_left = 0.1):
    #hack
    pose = coordinate_corr(env.cur_pos, env.cur_angle)
    return [pose[0]+np.cos(pose[2])*step_forward,pose[1]+np.sin(pose[2])*step_left]

def get_sample_area(start,goal,eps=0.2):
    #hack
    xmin = np.min([start[0],goal[0]])
    xmax = np.max([start[0],goal[0]])
    ymin = np.min([start[1],goal[1]])
    ymax = np.max([start[1],goal[1]])
    return [xmin-eps,xmax+eps,ymin-eps,ymax+eps]

def distance(point1,point2):
    return np.sqrt((point2[0]-point1[0])**2 +(point2[1]-point1[1])**2)

def get_desired_angle(point1,point2):
    return np.arctan2(point2[1]-point1[1],point2[0]-point1[0])-np.pi/2.0

# def get_velocities(env,x_ref,y_ref,theta_ref):
#     L =  env.unwrapped.wheel_dist/2.0
#     R = env.radius
#     x,y = get_current_pos()
#     theta = env.cur_angle + np.pi/2.0
#     delta_time = 1.0/ env.frame_rate
#     u_r = ((x - x_ref)*( x_ref + np.cos(theta_ref)) + (y - y_ref)*(y_ref + np.sin(theta_ref)) + (theta - theta_ref)*2.0/delta_time)*L/R
#     u_l = ((x - x_ref)*(-x_ref + np.cos(theta_ref)) + (y - y_ref)*(y_ref + np.sin(theta_ref)) + (theta - theta_ref)*2.0/delta_time)*L/R
#     return u_r,u_l

def get_vel_and_omega(env,start,goal,velocity,step,executed_vels,prev_error_d,prev_error_theta):
    integral_d = 0
    integral_theta = 0
    x1,y1 = start
    x2,y2 = goal
    x,y = get_current_pos()
    theta = np.mod(env.cur_angle + np.pi/2.0,2*np.pi)
    if theta > np.pi:
        theta -= 2*np.pi
    theta_ref = np.arctan2(y2-y1,x2-x1)
    print('Theta =  ',np.rad2deg(theta), ' Theta_ref = ', np.rad2deg(theta_ref))

    error_theta = theta_ref - theta

    error_d = np.abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)/np.sqrt((y2-y1)**2 + (x2-x1)**2)

    print('Error_d = ',error_d, ' error_theta = ',np.rad2deg(error_theta))

    delta_time = 1.0/ env.frame_rate

    #coefficients taken from lane controller node param file
    k_d = -3.5
    k_theta = -1*15
    k_Id = 1*10
    k_Itheta = 0

    #magic coefficient
    k_omega = 4.75
    
    if step > 0:
        integral_d = error_d * delta_time
        integral_theta = error_theta * delta_time

    integral_d_top_cutoff = 1
    integral_d_bottom_cutoff = -1
    integral_theta_top_cutoff = 1.2
    integral_theta_bottom_cutoff = -1.2

    if integral_d > integral_d_top_cutoff:
        integral_d = integral_d_top_cutoff
    if integral_d < integral_d_bottom_cutoff:
        integral_d = integral_d_bottom_cutoff
    
    if integral_theta > integral_theta_top_cutoff:
        integral_theta = integral_theta_top_cutoff
    if integral_theta < integral_theta_bottom_cutoff:
        integral_theta = integral_theta_bottom_cutoff
    
    if abs(error_d) <= 0.011:  # TODO: replace '<= 0.011' by '< delta_d' (but delta_d might need to be sent by the lane_filter_node.py or even lane_filter.py)
        integral_d = 0
    if abs(error_theta) <= 0.051:  # TODO: replace '<= 0.051' by '< delta_phi' (but delta_phi might need to be sent by the lane_filter_node.py or even lane_filter.py)
        integral_theta = 0
    if np.sign(error_d) != np.sign(prev_error_d):  # sign of error changed => error passed zero
        integral_d = 0
    if np.sign(error_theta) != np.sign(prev_error_theta):  # sign of error changed => error passed zero
        integral_theta = 0
    if executed_vels[0] == 0 and executed_vels[1] == 0:  # if actual velocity sent to the motors is zero
        integral_d = 0
        integral_theta = 0


    omega = -k_omega*(k_d * error_d + k_theta * error_theta - k_Id * integral_d  - k_Itheta * integral_theta)

    #if velocity - 0.5 * np.fabs(omega) * 0.1 < 0.065:
    #   velocity = 0.065 + 0.5 * np.fabs(omega) * 0.1

    omega_max = 999
    omega_min =-999
    if omega > omega_max: omega = omega_max
    if omega < omega_min: omega = omega_min

    return velocity, omega, error_d, error_theta

def get_omega_steer(desired_angle):
    delta_time = 1.0/ env.frame_rate
    current_angle = env.cur_angle + np.pi/2.0
    return (desired_angle-current_angle)/delta_time

def execute_path(path):
    print('Executing the path!')
    eps = 0.02
    executed_vels = [0,0]
    prev_error_d = 0
    prev_error_theta = 0
    for i in range(len(path)):
        if i == 0:
            x,y = get_current_pos()
            start = [x,y]
        else:
            start = path[i-1]

        goal = path[i]
        dist = distance(goal, get_current_pos())
        step = 0

        while dist > eps:
            current_pos = get_current_pos()
            desired_angle = get_desired_angle(current_pos,goal)
            steer_angle = ratiopositive_x(goal)
            print('Distance: ',dist, '  Desired angle: ',np.rad2deg(desired_angle), ' Steer angle: ', np.rad2deg(steer_angle))
            velocity = 0.5
            #omega_steer = get_omega_steer(desired_angle)
            omega_steer = velocity * steer_angle

            #u_r, u_l = inverse_kinematics(env, velocity, omega_steer) #np.tan(steer_angle) / env.wheel_dist
            velocity, omega, prev_error_d, prev_error_theta = get_vel_and_omega(env,start,goal,velocity,step,executed_vels, prev_error_d, prev_error_theta)
            u_r, u_l = inverse_kinematics(env, velocity, omega)

            obs, reward, done, info = env.step([u_r, u_l])

            env.render()

            if done:
                if reward < 0:
                    print('*** FAILED ***')
                    if not args.no_pause:
                        time.sleep(0.7)
                obs = env.reset()
                env.render()
                return False

            dist = distance(goal, get_current_pos())
            step += 1
            executed_vels = [u_r,u_l]
            rrts.draw_path(path,rrts.Point(current_pos[0],current_pos[1]),env.cur_angle +np.pi/2.0,prev_error_d,prev_error_theta,i,dist)

    return True

def get_current_pos():
    return [env.cur_pos[2],env.cur_pos[0]]


goal_pose = [0,0]
def onclick(event):
    global goal_pose
    goal_pose = [event.xdata, event.ydata]


if __name__ == '__main__':
    follow_dist = 0.4  # currently not used

    obstacle_list = get_obstacles(env)

    set_abs = True
    # ----------------------------------------------
    # RRT Star:
    # ----------------------------------------------
    # GET Trajectory

    #goals = get_goal_path()
    while True:
        start = get_current_pos()
        print('Starting position: ', start)
        print('Starting angle: ',env.cur_angle)
        fig = rrts.draw_obstacles(obstacle_list,rrts.Point(start[0],start[1]),env.cur_angle +np.pi/2.0)

        if set_abs:
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            fig.waitforbuttonpress()
            goal = goal_pose
        else:
            step_forward = float(input('Set step_forward:'))
            step_left = float(input('Set step_left:'))
            goal = get_goal(step_forward,step_left)

        sample_area = get_sample_area(start,goal,eps = 0.4)
        print('*********** RRT* ***********')
        print('Starting position: ', start)
        print('Goal position: ', goal)
        print('Sample area: ', sample_area)

        rrt_star = rrts.RRTStar(start=start, goal=goal,
                  sample_area=sample_area, obstacle_list=obstacle_list,steer_step=0.1,d=0.01,max_iter=50)
        path = rrt_star.planner(animation=True)
        if path == None:
            print('No possible paths...')
            done = False
        else:
            print('Calculated path!\n')
            print('Path: ',path)
            done = execute_path(path)

        print('Finished successfully: ', done)
        print('****************************')


