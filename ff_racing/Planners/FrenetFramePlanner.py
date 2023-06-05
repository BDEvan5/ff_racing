import numpy as np 
import matplotlib.pyplot as plt
from ff_racing.PlannerUtils.VehicleStateHistory import VehicleStateHistory
from ff_racing.PlannerUtils.TrackLine import TrackLine
from numba import njit  

import cv2 as cv
from PIL import Image
import os
from ff_racing.PlannerUtils.LocalMap import LocalMap

LOOKAHEAD_DISTANCE = 1
WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 6

    
def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    
    return xs, ys

class FrenetFramePlanner:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        
        if not os.path.exists(path): os.mkdir(path)
        if not os.path.exists(path + "Scans/"): os.mkdir(path+ "Scans/")
        if not os.path.exists(path + "ScanData/"): os.mkdir(path+ "ScanData/")
        if not os.path.exists(path + "LocalMaps/"): os.mkdir(path+ "LocalMaps/")
        # self.vehicle_state_history = VehicleStateHistory(name, "ff")
        
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)
        self.sines = np.sin(self.angles)
        
        self.counter = 0
        
        self.local_map = None
        
    def plan(self, obs):
        scan = obs['scans'][0]
        pose = np.array([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]])
        
        np.save(self.path + "ScanData/" + f"{self.name}_{self.counter}.npy", obs['scans'][0])
        center_line = self.run_scan_centerline(scan)
        self.local_map.plot_map_img(scan, pose)

        action = self.pure_pursuit()

        plt.savefig(self.path + "LocalMaps/" + f"{self.name}_{self.counter}.svg")
        plt.pause(0.0001)
        self.counter += 1
        return action
        
    def run_scan_centerline(self, scan):
        n_pts = 20
        xs = self.coses[scan < 10] * scan[scan < 10]
        ys = self.sines[scan < 10] * scan[scan < 10]
        xs = xs[180:-180]
        ys = ys[180:-180]

        pts = np.hstack((xs[:, None], ys[:, None]))
        pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        mid_idx = np.argmax(pt_distances)

        l1_cs = np.cumsum(pt_distances[:mid_idx+1])
        l2_cs = np.cumsum(pt_distances[mid_idx:])
        
        l1_ss = np.linspace(0, l1_cs[-1], n_pts)
        l2_ss = np.linspace(0, l2_cs[-1], n_pts)

        l1_xs, l1_ys = interp_2d_points(l1_ss, l1_cs, pts[:mid_idx+1])
        l2_xs, l2_ys = interp_2d_points(l2_ss, l2_cs, pts[mid_idx+1:])
        
        c_xs = (l1_xs + l2_xs[::-1])/2
        c_ys = (l1_ys + l2_ys[::-1])/2
        center_line = np.hstack((c_xs[:, None], c_ys[:, None]))
        
        #Reregularise the center line distances
        cl_dists = np.linalg.norm(center_line[1:] - center_line[:-1], axis=1)
        cl_cs = np.cumsum(cl_dists)
        cl_cs = np.insert(cl_cs, 0, 0)
        cl_ss = np.linspace(0, cl_cs[-1], n_pts)
        cl_xs, cl_ys = interp_2d_points(cl_ss, cl_cs, center_line)
        
        center_line = np.hstack((cl_xs[:, None], cl_ys[:, None]))

        ws = np.ones(n_pts) * 0.8
        
        self.local_map = LocalMap(center_line, ws)
        self.local_map.save_map(self.path, self.counter)
        # self.local_map.plot_map(scan)
       
    def pure_pursuit(self, ):
        assert self.local_map is not None, "No local map has been created"
        
        center_line = self.local_map.pts
        
        distances = np.linalg.norm(center_line[1:] - center_line[:-1], axis=1)
        lengths = np.cumsum(distances)
        lengths = np.insert(lengths, 0, 0)
        
        position = np.array([0, 0])
        lookahead = 2
        
        lookahead = min(lookahead, lengths[-1]) 
         
        lookahead_point = interp_2d_points(lookahead, lengths, center_line)
        plt.plot(lookahead_point[0], lookahead_point[1], 'o', color='green', label="Lookahead")
        
        theta = 0 #! TODO: get calculate theta relative to center line.
        position = np.array([0, 0])
        steering_angle = get_steering_actuation(theta, lookahead_point, position, LOOKAHEAD_DISTANCE, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        
        speed = 3

        return np.array([steering_angle, speed])
        
    def plan_frenet_action(self, center_line, speed):
        
        
        return np.array([0, 5])
        
        
    def done_callback(self, obs):
        pass
        
     
    
# @njit(fastmath=False, cache=True)
def get_steering_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle
   
