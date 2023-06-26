"""
Partial code source: https://github.com/f1tenth/f1tenth_gym
Example waypoint_follow.py from f1tenth_gym
Specific function used:
- nearest_point_on_trajectory_py2
- first_point_on_trajectory_intersecting_circle
- get_actuation

Adjustments have been made

"""

import numpy as np
from numba import njit
import os
from LocalMapRacing.planner_utils.TrackLine import TrackLine
from LocalMapRacing.planner_utils.VehicleStateHistory import VehicleStateHistory

LOOKAHEAD_DISTANCE = 0.8
WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 8
GRAVITY = 9.81

class PurePursuit:
    def __init__(self, map_name, test_name):
        path = f"Data/" + test_name + "/"
        if not os.path.exists(path):
            os.mkdir(path)
            
        self.track_line = TrackLine(map_name, True, False)
        # self.track_line.plot_wpts()

        self.vehicle_state_history = VehicleStateHistory(test_name, map_name)

        self.counter = 0

    def plan(self, obs):
        position = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        theta = obs['poses_theta'][0]

        # self.track_line.plot_vehicle(position, theta)
        
        lookahead_distance = 0.3 + obs['linear_vels_x'][0] * 0.15
        lookahead_point = self.track_line.get_lookahead_point(position, lookahead_distance)

        if obs['linear_vels_x'][0] < 1:
            return np.array([0.0, 4])

        speed_raceline, steering_angle = get_actuation(theta, lookahead_point, position, LOOKAHEAD_DISTANCE, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
            
        speed = min(speed_raceline, MAX_SPEED) # cap the speed
        max_speed = calculate_speed_limit(steering_angle)
        speed = min(speed, max_speed)
        action = np.array([steering_angle, speed])
        
        self.vehicle_state_history.add_memory_entry(obs, action)

        return action

    def done_callback(self, final_obs):
        self.vehicle_state_history.add_memory_entry(final_obs, np.array([0, 0]))
        self.vehicle_state_history.save_history()
        
        progress = self.track_line.calculate_progress_percent([final_obs['poses_x'][0], final_obs['poses_y'][0]]) * 100
        
        print(f"Lap complete ({self.track_line.map_name.upper()}) --> Time: {final_obs['lap_times'][0]:.2f}, Progress: {progress:.1f}%")



@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

     
@njit(cache=True)
def calculate_speed_limit(delta, friction_limit=1.2):
    if abs(delta) < 0.03:
        return MAX_SPEED

    V = np.sqrt(friction_limit*GRAVITY*WHEELBASE/np.tan(abs(delta)))
    V = min(V, MAX_SPEED)

    return V