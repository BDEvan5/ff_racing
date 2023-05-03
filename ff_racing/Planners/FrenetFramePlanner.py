import numpy as np 
import matplotlib.pyplot as plt
from ff_racing.PlannerUtils.VehicleStateHistory import VehicleStateHistory
from ff_racing.PlannerUtils.TrackLine import TrackLine

import cv2 as cv
from PIL import Image
import os

LOOKAHEAD_DISTANCE = 1
WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 6


class FrenetFramePlanner:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        
        if not os.path.exists(path): os.mkdir(path)
        if not os.path.exists(path + "Scans/"): os.mkdir(path+ "Scans/")
        if not os.path.exists(path + "ScanData/"): os.mkdir(path+ "ScanData/")
        # self.vehicle_state_history = VehicleStateHistory(name, "ff")
        
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)
        self.sines = np.sin(self.angles)
        
        self.counter = 0
        
    def plan(self, obs):
        speed = obs['linear_vels_x'][0]
        
        np.save(self.path + "ScanData/" + f"{self.name}_{self.counter}.npy", obs['scans'][0])
        self.run_scan_centerline(obs['scans'][0])

        self.counter += 1
        return np.array([0, 5])
        
        center_line = self.run_centerline_extraction(obs['scans'][0])
        
        action = self.plan_frenet_action(center_line, speed)
        
        
        return action
    
    def run_scan_centerline(self, scan):
        n2 = 540

        xs = self.coses * scan
        ys = self.sines * scan

        c_xs = (xs[:n2] + xs[n2:][::-1])/2
        c_ys = (ys[:n2] + ys[n2:][::-1])/2

        plt.figure(1)
        plt.clf()
        plt.plot(xs, ys, label="Lidar")
        plt.plot(c_xs, c_ys, label="Center")

        plt.pause(0.001)
        
        
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
        
        
