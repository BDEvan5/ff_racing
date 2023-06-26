import numpy as np
import os
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
import glob
from matplotlib.collections import LineCollection
    
def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    
    return xs, ys

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)



class LocalMapExtraction:
    def __init__(self, scan, counter) -> None:
        self.counter = counter
        self.pts = None
        self.ws = None
        self.scan = scan[180:-180]
        
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)[180:-180]
        self.sines = np.sin(self.angles)[180:-180]
        
        self.n_pts = 20
        
    def extract_scan_centerline(self):
        scan = self.scan
        xs = self.coses[scan < 10] * scan[scan < 10]
        ys = self.sines[scan < 10] * scan[scan < 10]

        pts = np.hstack((xs[:, None], ys[:, None]))
        pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        mid_idx = np.argmax(pt_distances)

        l1_cs = np.cumsum(pt_distances[:mid_idx+1])
        l2_cs = np.cumsum(pt_distances[mid_idx:])
        
        l1_ss = np.linspace(0, l1_cs[-1], self.n_pts)
        l2_ss = np.linspace(0, l2_cs[-1], self.n_pts)

        l1_xs, l1_ys = interp_2d_points(l1_ss, l1_cs, pts[:mid_idx+1])
        l2_xs, l2_ys = interp_2d_points(l2_ss, l2_cs, pts[mid_idx+1:])
        
        c_xs = (l1_xs + l2_xs[::-1])/2
        c_ys = (l1_ys + l2_ys[::-1])/2
        center_line = np.hstack((c_xs[:, None], c_ys[:, None]))
        
        #Reregularise the center line distances
        cl_dists = np.linalg.norm(center_line[1:] - center_line[:-1], axis=1)
        cl_cs = np.cumsum(cl_dists)
        cl_cs = np.insert(cl_cs, 0, 0)
        cl_ss = np.linspace(0, cl_cs[-1], self.n_pts)
        cl_xs, cl_ys = interp_2d_points(cl_ss, cl_cs, center_line)
        
        self.pts = np.hstack((cl_xs[:, None], cl_ys[:, None]))
        self.ws = np.ones(self.n_pts) * 0.8
                
    def calculate_nvecs(self):
        el_lengths = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(self.pts, el_lengths, False)
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi-np.pi/2)
    
    def plot_map(self):
        l1 = self.pts + self.nvecs * self.ws[:, None]
        l2 = self.pts - self.nvecs * self.ws[:, None]
        
        plt.figure(1)
        plt.clf()
        xs = self.coses[self.scan < 10] * self.scan[self.scan < 10]
        ys = self.sines[self.scan < 10] * self.scan[self.scan < 10]
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
        plt.pause(0.1)
        
    
def run_loop(path="Data/LocalMapPlanner2/ScanData/"):
    laps = glob.glob(path + "LocalMapPlanner*.npy")
    laps.sort()
    
    for i, lap in enumerate(laps):
        print(f"Processing lap {i}")
        data = np.load(lap)
        local_map = LocalMapExtraction(data, i)
        local_map.extract_scan_centerline()
        local_map.calculate_nvecs()
        local_map.plot_map()
        
        # if i > 20:
        #     break



if __name__ == "__main__":
    run_loop()
    plt.show()
    pass