from typing import Any
import numpy as np
import os
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
    
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
        
    def calculate_nvecs(self):
        el_lengths = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(self.pts, el_lengths, False)
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi-np.pi/2)
        
    def get_lookahead_point(self, lookahead_distance):
        lookahead = min(lookahead, self.lengths[-1]) 
         
        lookahead_point = interp_2d_points(lookahead, self.lengths, self.pts)
        
        return lookahead_point
    
    def plot_map(self):
        l1 = self.pts + self.nvecs * self.ws[:, None]
        l2 = self.pts - self.nvecs * self.ws[:, None]
        
        plt.figure(1)
        plt.clf()
        plt.plot(self.pts[:, 0], self.pts[:, 1], '-', color='red', label="Center", linewidth=3)
        plt.plot(0, 0, 'x', color='black', label="Origin")

        plt.plot(l1[:, 0], l1[:, 1], color='green')
        plt.plot(l2[:, 0], l2[:, 1], color='green')

        for i in range(len(self.ws)):
            xs = [l1[i, 0], l2[i, 0]]
            ys = [l1[i, 1], l2[i, 1]]
            plt.plot(xs, ys)

        # plt.xlim(-1, 10)
        # plt.ylim(-10, 10)

        plt.gca().set_aspect('equal', adjustable='box')
        
    def save_map(self, path, number):
        # save the data into a folder fo later use
        ensure_path_exists(path + "LocalMapData/")
        data = np.concatenate([self.pts, self.ws[:, None]], axis=1)
        np.save(path + "LocalMapData/" + f"local_map_{number}.npy", data)
        
