from typing import Any
import numpy as np
import os
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
from PIL import Image
import yaml
    
def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    
    return xs, ys

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)

class LocalMap:
    def __init__(self, pts, ws) -> None:
        self.xs = pts[:, 0]
        self.ys = pts[:, 1]
        self.pts = pts
        self.ws = ws
        
        self.calculate_nvecs()
        
        self.distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        self.lengths = np.insert(np.cumsum(self.distances), 0, 0)
        
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)
        self.sines = np.sin(self.angles)
        
        map_name = "maps/mco.yaml"
        self.map_img = None
        self.set_map(map_name, ".png")
        
    def set_map(self, map_path, map_ext):
        map_img_path = os.path.splitext(map_path)[0] + map_ext
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)

        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        with open(map_path, 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.map_origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)

    def calculate_nvecs(self):
        el_lengths = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(self.pts, el_lengths, False)
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi-np.pi/2)
        
    def get_lookahead_point(self, lookahead_distance):
        lookahead = min(lookahead, self.lengths[-1]) 
         
        lookahead_point = interp_2d_points(lookahead, self.lengths, self.pts)
        
        return lookahead_point
    
    def plot_map(self, scan, location):
        l1 = self.pts + self.nvecs * self.ws[:, None]
        l2 = self.pts - self.nvecs * self.ws[:, None]
        
        plt.figure(1)
        plt.clf()
        xs = self.coses[scan < 10] * scan[scan < 10]
        ys = self.sines[scan < 10] * scan[scan < 10]
        plt.plot(xs, ys, '.', color='blue', label="Scan")
        
        plt.plot(self.pts[:, 0], self.pts[:, 1], '-', color='red', label="Center", linewidth=3)
        plt.plot(0, 0, 'x', color='black', label="Origin")

        plt.plot(l1[:, 0], l1[:, 1], color='green')
        plt.plot(l2[:, 0], l2[:, 1], color='green')

        for i in range(len(self.ws)):
            xs = [l1[i, 0], l2[i, 0]]
            ys = [l1[i, 1], l2[i, 1]]
            plt.plot(xs, ys)

        plt.gca().set_aspect('equal', adjustable='box')
        
    def plot_map_img(self, scan, pose):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.map_img, cmap='gray', origin='lower', alpha=0.5)
        
        l1 = self.pts + self.nvecs * self.ws[:, None]
        l1 = transpose_point(l1, pose)
        l2 = self.pts - self.nvecs * self.ws[:, None]
        l2 = transpose_point(l2, pose)
        
        xs = self.coses[scan < 10] * scan[scan < 10]
        ys = self.sines[scan < 10] * scan[scan < 10]
        
        pts = np.hstack((xs[:, None], ys[:, None]))
        pts = transpose_point(pts, pose)
        xs, ys = convert_pts(pts, self.map_origin, self.map_resolution)
        plt.plot(xs, ys, '.', color='blue', label="Scan")
        
        plt.plot(self.pts[:, 0], self.pts[:, 1], '-', color='red', label="Center", linewidth=3)
        plt.plot(0, 0, 'x', color='black', label="Origin")

        xs, ys = convert_pts(l1, self.map_origin, self.map_resolution)
        plt.plot(xs, ys, color='green')
        xs, ys = convert_pts(l2, self.map_origin, self.map_resolution)
        plt.plot(xs, ys, color='green')

        for i in range(len(self.ws)):
            xs = np.array([l1[i, 0], l2[i, 0]])
            ys = np.array([l1[i, 1], l2[i, 1]])
            convert_xs_ys(xs, ys, self.map_origin, self.map_resolution)
            plt.plot(xs, ys)

        plt.gca().set_aspect('equal', adjustable='box')
        
    def convert_xy_to_rc(self, xs, ys):
        for x, y in zip(xs, ys):
            r = np.sqrt(x**2 + y**2)
            c = np.arctan2(y, x)
            
            yield r, c
        
    def save_map(self, path, number):
        # save the data into a folder fo later use
        ensure_path_exists(path + "LocalMapData/")
        data = np.concatenate([self.pts, self.ws[:, None]], axis=1)
        np.save(path + "LocalMapData/" + f"local_map_{number}.npy", data)
        
def convert_pts(points, origin, resolution):
    xs = (points[:, 0] - origin[0])/resolution
    ys = (points[:, 1] - origin[1])/resolution
    return xs, ys      
  
def convert_xs_ys(xs, ys, origin, resolution):
    xs = (xs - origin[0])/resolution
    ys = (ys - origin[1])/resolution
    return xs, ys

def transpose_point(pts, pose):
    position = pose[0:2]
    angle_rad = -pose[2]
    rotation_mtrx = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]])
    
    pts = np.dot(pts, rotation_mtrx) + position
    
    return pts
    
