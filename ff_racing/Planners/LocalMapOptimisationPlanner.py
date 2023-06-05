import numpy as np 
import matplotlib.pyplot as plt
from ff_racing.PlannerUtils.VehicleStateHistory import VehicleStateHistory
from ff_racing.PlannerUtils.TrackLine import TrackLine
from numba import njit  

import cv2 as cv
from PIL import Image
import os
from ff_racing.PlannerUtils.LocalMap import LocalMap
from ff_racing.PlannerUtils.OptimiseLocalMap import LocalMap


LOOKAHEAD_DISTANCE = 1.5
WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 8

    
def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    
    return xs, ys

def ensure_path_exists(path):
    if not os.path.exists(path): os.mkdir(path)

class LocalOptimisationPlanner:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        
        ensure_path_exists(path)
        ensure_path_exists(path + "Scans/")
        self.vehicle_state_history = VehicleStateHistory(name, "LocalMap")
                
        self.counter = 0
        self.local_map = LocalMap(self.path)
        
    def plan(self, obs):
        scan = obs['scans'][0]
        
        self.local_map.generate_local_map(scan)
        self.local_map.plot_save_local_map()
        self.local_map.generate_minimum_curvature_path()
        self.local_map.generate_max_speed_profile()
        self.local_map.plot_save_raceline()

        action = self.local_map_pure_pursuit()

        self.vehicle_state_history.add_memory_entry(obs, action)

        self.counter += 1
        return action
        
    def local_map_pure_pursuit(self):
        assert self.local_map is not None, "No local map has been created"
        
        lookahead = min(LOOKAHEAD_DISTANCE, self.local_map.s_raceline[-1]) 
        lookahead_point = interp_2d_points(lookahead, self.local_map.s_raceline, self.local_map.raceline)
        # self.local_map.plot_local_raceline()
        # plt.plot(lookahead_point[0], lookahead_point[1], 'o', color='green', label="Lookahead")
        
        theta = 0 #! TODO: get calculate theta relative to center line.
        position = np.array([0, 0])
        steering_angle = get_steering_actuation(theta, lookahead_point, position, LOOKAHEAD_DISTANCE, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        
        speed = 3

        return np.array([steering_angle, speed])
        
    def done_callback(self, obs):
        self.vehicle_state_history.save_memory()
        pass
        
     
    
# @njit(fastmath=False, cache=True)
def get_steering_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle
   
