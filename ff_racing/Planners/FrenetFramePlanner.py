import numpy as np 
import matplotlib.pyplot as plt
from ff_racing.PlannerUtils.VehicleStateHistory import VehicleStateHistory
from ff_racing.PlannerUtils.TrackLine import TrackLine
from numba import njit  

import cv2 as cv
from PIL import Image
import os

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
        # if not os.path.exists(path + "ScanData/"): os.mkdir(path+ "ScanData/")
        if not os.path.exists(path + "LocalMaps/"): os.mkdir(path+ "LocalMaps/")
        # self.vehicle_state_history = VehicleStateHistory(name, "ff")
        
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)
        self.sines = np.sin(self.angles)
        
        self.counter = 0
        
    def plan(self, obs):
        speed = obs['linear_vels_x'][0]
        # scan = np.clip(obs['scans'][0], 0, 10)
        scan = obs['scans'][0]
        
        
        np.save(self.path + "ScanData/" + f"{self.name}_{self.counter}.npy", obs['scans'][0])
        center_line = self.run_scan_centerline(scan)
        self.plot_centerline(scan, center_line)

        action = self.pure_pursuit(center_line)

        plt.savefig(self.path + "LocalMaps/" + f"{self.name}_{self.counter}.svg")
        self.counter += 1
        return action
        
        center_line = self.run_centerline_extraction(obs['scans'][0])
        
        action = self.plan_frenet_action(center_line, speed)
        
        
        return action
    
    def run_scan_centerline(self, scan):
        # scan = np.clip(scan, 0, 10)
        # scan = scan[scan < 10]
        xs = self.coses[scan < 10] * scan[scan < 10]
        ys = self.sines[scan < 10] * scan[scan < 10]
        xs = xs[180:-180]
        ys = ys[180:-180]

        pts = np.hstack((xs[:, None], ys[:, None]))
        pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        mid_idx = np.argmax(pt_distances)

        l1_cs = np.cumsum(pt_distances[:mid_idx+1])
        l2_cs = np.cumsum(pt_distances[mid_idx:])
        
        l1_ss = np.linspace(0, l1_cs[-1], 50)
        l1_ss = np.linspace(0, l2_cs[-1], 50)

        l1_xs, l1_ys = interp_2d_points(l1_ss, l1_cs, pts[:mid_idx+1])
        l2_xs, l2_ys = interp_2d_points(l1_ss, l2_cs, pts[mid_idx+1:])

        c_xs = (l1_xs + l2_xs[::-1])/2
        c_ys = (l1_ys + l2_ys[::-1])/2
        
        center_line = np.hstack((c_xs[:, None], c_ys[:, None]))
        
        return center_line
        
    def plot_centerline(self, scan, centerline):
        xs = self.coses[scan < 10] * scan[scan < 10]
        ys = self.sines[scan < 10] * scan[scan < 10]
        xs = xs[180:-180]
        ys = ys[180:-180]
        
        plt.figure(1)
        plt.clf()
        plt.plot(xs, ys, 'x', label="Lidar")
        
        plt.plot(centerline[:, 0], centerline[:, 1], '-', color='red', label="Center", linewidth=3)
        plt.plot(0, 0, 'x', color='black', label="Origin")

        plt.gca().set_aspect('equal', adjustable='box')

        
        
    def pure_pursuit(self, center_line):
        distances = np.linalg.norm(center_line[1:] - center_line[:-1], axis=1)
        lengths = np.cumsum(distances)
        lengths = np.insert(lengths, 0, 0)
        
        position = np.array([0, 0])
        progress = np.linalg.norm(position - center_line[0])
        
        lookahead = 2
        lookahead = min(lookahead, lengths[-1]) # make sure lookahead within visible distance.
         
        lookahead_point = interp_2d_points(lookahead, lengths, center_line)
        plt.plot(lookahead_point[0], lookahead_point[1], 'o', color='green', label="Lookahead")
        
        theta = 0 #! TODO: get calculate theta relative to center line.
        position = np.array([0, 0])
        steering_angle = get_steering_actuation(theta, lookahead_point, position, LOOKAHEAD_DISTANCE, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        
        speed = 3

        return np.array([steering_angle, speed])
        
    def run_centerline_extraction(self, scan):
        save_path = self.path + "Scans/"
        plt.figure()
        plt.plot(scan*self.coses, scan*self.sines)
        
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path + f"{self.name}_{self.counter}.png", bounding_box_inches='tight', pad_inches=0, dpi=50)
        
        self.process_img()
        
        plt.show()
    
    def process_img(self):
        img = cv.imread(self.path + "Scans/" + f"{self.name}_{self.counter}.png")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        edges = cv.Canny(img, 100, 250, apertureSize=5, L2gradient=True)

        # Apply Hough Line Transform to detect lines
        lines = cv.HoughLines(edges, 1, np.pi/180, 200)
        
        line_img = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Display the image with the detected lines
        cv.imshow('Lines', line_img)
        cv.waitKey(0)
        
        # blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
        # _, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        # ret, thresh = cv.threshold(blur, 200, 255,
                                # cv.THRESH_BINARY_INV)


        # cv.imshow('ocv', gray)
        # cv.waitKey(0)
        

        # c, _ = cv.findContours(gray,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        # c, _ = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        
        cv.drawContours(img, c, -1, (0,255,0), 3)
        cv.imshow('ocv', img)
        cv.waitKey(10 * 1000)
        
        return img
    
        
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
   
