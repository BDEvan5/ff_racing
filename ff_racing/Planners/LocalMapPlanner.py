import numpy as np 
import matplotlib.pyplot as plt
from ff_racing.planner_utils.VehicleStateHistory import VehicleStateHistory
from ff_racing.planner_utils.TrackLine import TrackLine
from numba import njit  

from ff_racing.local_mapping.local_map_utils import *
from ff_racing.local_mapping.LocalMap import LocalMap
from ff_racing.local_mapping.LocalMapGenerator import LocalMapGenerator
from ff_racing.local_mapping.LocalRaceline import LocalRaceline

np.set_printoptions(precision=4)


LOOKAHEAD_DISTANCE = 1.1
WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 8


class LocalMapPlanner:
    def __init__(self, name, path, map_name):
        self.name = name
        self.path = path
        
        ensure_path_exists(path)
        self.vehicle_state_history = VehicleStateHistory(name, map_name)
        self.counter = 0
                
        self.local_map_generator = LocalMapGenerator(self.path)
        self.local_map = None
        self.local_raceline = LocalRaceline(self.path)
        
    def plan(self, obs):
        self.local_map = self.local_map_generator.generate_line_local_map(obs['scans'][0])
        self.local_raceline.generate_raceline(self.local_map)

        # action = self.pure_pursuit_center_line()
        action, lhd = self.pure_pursuit_racing_line()

        self.vehicle_state_history.add_memory_entry(obs, action)

        self.counter += 1
        return action
        
    def pure_pursuit_center_line(self):
        current_progress = np.linalg.norm(self.local_map.track[0, 0:2])
        lookahead = LOOKAHEAD_DISTANCE + current_progress
        lookahead = min(lookahead, self.local_map.s_track[-1]) 
        lookahead_point = interp_2d_points(lookahead, self.local_map.s_track, self.local_map.track[:, 0:2])

        steering_angle = get_local_steering_actuation(lookahead_point, LOOKAHEAD_DISTANCE, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        speed = 3
        
        return np.array([steering_angle, speed])

    def pure_pursuit_racing_line(self):
        current_progress = np.linalg.norm(self.local_raceline.raceline[0, :])
        lookahead = LOOKAHEAD_DISTANCE + current_progress
        lookahead = min(lookahead, self.local_raceline.s_track[-1]) 
        lookahead_point = interp_2d_points(lookahead, self.local_raceline.s_track, self.local_raceline.raceline)

        steering_angle = get_local_steering_actuation(lookahead_point, LOOKAHEAD_DISTANCE, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        speed = np.interp(current_progress, self.local_raceline.s_track, self.local_raceline.vs) #* 0.8
        max_turning_speed = calculate_speed(steering_angle, f_s=0.99, max_v=8)
        print(f"Speed: {speed:.3f}, Max turning speed: {max_turning_speed:.3f} --> Difference: {(speed - max_turning_speed):.3f}")
        speed = min(speed, max_turning_speed)


        return np.array([steering_angle, speed]), lookahead_point
    
        
    def done_callback(self, obs):
        self.vehicle_state_history.save_history()
        pass
        
     
@njit(cache=True)
def calculate_speed(delta, f_s=0.8, max_v=7):
    b = 0.523
    g = 9.81
    l_d = 0.329

    if abs(delta) < 0.03:
        return max_v
    if abs(delta) > 0.4:
        return 0

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    V = min(V, max_v)

    return V
    
# @njit(fastmath=False, cache=True)
def get_steering_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle

@njit(fastmath=False, cache=True)
def get_local_steering_actuation(lookahead_point, lookahead_distance, wheelbase):
    waypoint_y = lookahead_point[1]
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle
   
