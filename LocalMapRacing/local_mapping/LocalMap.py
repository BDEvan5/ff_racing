import numpy as np
import LocalMapRacing.tph_utils as tph
import matplotlib.pyplot as plt
from LocalMapRacing.local_mapping.local_map_utils import *

from scipy import interpolate
from scipy import optimize
from scipy import spatial

class LocalMap:
    def __init__(self, track):
        self.track = track
        self.el_lengths = None
        self.psi = None
        self.kappa = None
        self.nvecs = None
        self.s_track = None

        self.calculate_length_heading_nvecs()

    def calculate_length_heading_nvecs(self):
        self.el_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        self.s_track = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi-np.pi/2)

    def calculate_s(self, point):
        distances = np.linalg.norm(self.track[:, 0:2] - point, axis=1)
        idx = np.argmin(distances)
        x, h = self.interp_pts(idx, distances)
        s = (self.s_track[idx] + x) 

        return s, h

    def interp_pts(self, idx, dists):
        d_ss = self.s_track[idx+1] - self.s_track[idx]
        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else:     # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            if Area_square < 0:  # negative due to floating point precision
                h = 0
                x = d_ss + d1
            else:
                Area = Area_square**0.5
                h = Area * 2/d_ss
                x = (d1**2 - h**2)**0.5

        return x, h
    
    def interpolate_track(self, spacing=0.8):
        new_s = np.linspace(0, self.s_track[-1], int(self.s_track[-1]/spacing))
        self.track = interp_nd_points(new_s, self.s_track, self.track)

        self.calculate_length_heading_nvecs()
    
    def apply_required_smoothing(self, counter, path):
        self.interpolate_track(0.4)
        no_points_track = len(self.track)
        
        old_track = np.copy(self.track)
        old_nvecs = np.copy(self.nvecs)

        crossing_horizon = min(5, len(self.track)//2 -1)

        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, crossing_horizon)
        if not crossing: return

        i = 0
        while i < 20 and crossing:
            i += 1
            smoothing = i *0.05
            print("Smoothing: ", smoothing)
            ws = np.ones(len(self.track))
            ws[0] = 100
            ws[-1] = 100
            tck, t_glob = interpolate.splprep([self.track[:, 0], self.track[:, 1]], w=ws, k=3, s=smoothing)[:2]

            # Over extend the smooth path so that the true path length can be found
            smooth_path = np.array(interpolate.splev(np.linspace(0.0, 1.2, no_points_track*4), tck, ext=0)).T[:-1]
            dists = np.linalg.norm(smooth_path - self.track[-1, :2], axis=1)
            min_point = np.argmin(dists, axis=0)
            smooth_path = smooth_path[:min_point+1]

            # get normal path 
            tck = interpolate.splprep([smooth_path[:, 0], smooth_path[:, 1]], k=3, s=0)[0]
            smooth_path = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_track), tck, ext=0)).T

            # find new widths
            closest_t_glob = np.zeros(no_points_track)
            closest_point = np.zeros((no_points_track, 2))
            dists = np.zeros(len(self.track))
            t_glob_guess = self.s_track / self.s_track[-1]
            for z in range(no_points_track):
                closest_t_glob[z] = optimize.fmin(dist_to_p, x0=t_glob_guess[z], args=(tck, self.track[z, :2]), disp=False)

                closest_point[z] = interpolate.splev(closest_t_glob[z], tck)
                dists[z] = np.linalg.norm(closest_point[z] - self.track[z, :2])


            sides = np.zeros(no_points_track)
            for z in range(1, no_points_track):
                sides[z-1] = side_of_line(a=self.track[z-1, :2], b=self.track[z, :2], z=closest_point[z])
                
            w_tr_right_new = self.track[:, 2] + sides * dists
            w_tr_left_new = self.track[:, 3] - sides * dists

            smooth_track = np.hstack((smooth_path, w_tr_left_new.reshape(-1, 1), w_tr_right_new.reshape(-1, 1)))

            self.track = smooth_track
            self.calculate_length_heading_nvecs()

            crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, crossing_horizon)

        self.plot_smoothing(old_track, old_nvecs, counter, path)

        # self.interpolate_track()

    def plot_smoothing(self, old_track, old_nvecs, counter, path):
        plt.figure(2)
        plt.clf()
        plt.plot(old_track[:, 0], old_track[:, 1], 'r', linewidth=2)
        l1 = old_track[:, :2] + old_track[:, 2][:, None] * old_nvecs
        l2 = old_track[:, :2] - old_track[:, 3][:, None] * old_nvecs
        plt.plot(l1[:, 0], l1[:, 1], 'r', linestyle='--', linewidth=1)
        plt.plot(l2[:, 0], l2[:, 1], 'r', linestyle='--', linewidth=1)
        for z in range(len(old_track)):
            xs = [l1[z, 0], l2[z, 0]]
            ys = [l1[z, 1], l2[z, 1]]
            plt.plot(xs, ys, color='orange', linewidth=1)

        plt.plot(self.track[:, 0], self.track[:, 1], 'b', linewidth=2)
        l1 = self.track[:, :2] + self.track[:, 2][:, None] * self.nvecs
        l2 = self.track[:, :2] - self.track[:, 3][:, None] * self.nvecs
        plt.plot(l1[:, 0], l1[:, 1], 'b', linestyle='--', linewidth=1)
        plt.plot(l2[:, 0], l2[:, 1], 'b', linestyle='--', linewidth=1)
        for z in range(len(self.track)):
            xs = [l1[z, 0], l2[z, 0]]
            ys = [l1[z, 1], l2[z, 1]]
            plt.plot(xs, ys, color='green', linewidth=1)

        plt.axis('equal')
        # plt.show()
        # plt.pause(0.0001)
        plt.savefig(path + f"Smoothing_{counter}.svg")


def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
    s = interpolate.splev(t_glob, path)
    return spatial.distance.euclidean(p, s)

from typing import Union
def side_of_line(a: Union[tuple, np.ndarray],
                 b: Union[tuple, np.ndarray],
                 z: Union[tuple, np.ndarray]) -> float:

    side = np.sign((b[0] - a[0]) * (z[1] - a[1]) - (b[1] - a[1]) * (z[0] - a[0]))

    return side

class PlotLocalMap(LocalMap):
    def __init__(self, track):
        super().__init__(track)

        # self.local_map_img_path = self.path + "LocalMapImgs/"
        # ensure_path_exists(self.local_map_img_path)
    
    def plot_local_map(self, save_path=None, counter=0, xs=None, ys=None):
        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]

        plt.figure(1)
        plt.clf()
        if xs is not None and ys is not None:
            plt.plot(xs, ys, '.', color='#0057e7', alpha=0.7)
        plt.plot(self.track[:, 0], self.track[:, 1], '-', color='#E74C3C', label="Center", linewidth=3)
        plt.plot(0, 0, 'x', color='black', markersize=10)

        plt.plot(l1[:, 0], l1[:, 1], color='#ffa700')
        plt.plot(l2[:, 0], l2[:, 1], color='#ffa700')

        for i in range(len(self.track)):
            xs = [l1[i, 0], l2[i, 0]]
            ys = [l1[i, 1], l2[i, 1]]
            plt.plot(xs, ys, '#ffa700')

        plt.title("Local Map")
        plt.gca().set_aspect('equal', adjustable='box')

        if save_path is not None:
            plt.savefig(save_path + f"Local_map_{counter}.svg")

    def plot_local_map_offset(self, offset_pos, offset_theta, origin, resolution, save_path=None, counter=0):
        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]

        rotation = np.array([[np.cos(offset_theta), -np.sin(offset_theta)],
                                [np.sin(offset_theta), np.cos(offset_theta)]])
        
        l1 = np.matmul(rotation, l1.T).T
        l2 = np.matmul(rotation, l2.T).T

        l1 = l1 + offset_pos
        l2 = l2 + offset_pos

        l1 = (l1 - origin) / resolution
        l2 = (l2 - origin) / resolution

        track = np.matmul(rotation, self.track[:, :2].T).T
        track = track + offset_pos
        track = (track - origin) / resolution

        position = (offset_pos - origin) / resolution

        # plt.figure(1)
        # plt.clf()
        plt.plot(track[:, 0], track[:, 1], '--', color='#E74C3C', label="Center", linewidth=2)
        plt.plot(track[0, 0], track[0, 1], '*', color='#E74C3C', markersize=10)

        plt.plot(l1[:, 0], l1[:, 1], color='#ffa700')
        plt.plot(l2[:, 0], l2[:, 1], color='#ffa700')

        plt.gca().set_aspect('equal', adjustable='box')

        buffer = 50

        plt.xlim([np.min(track[:, 0]) - buffer, np.max(track[:, 0]) + buffer])
        plt.ylim([np.min(track[:, 1]) - buffer, np.max(track[:, 1]) + buffer])

        if save_path is not None:
            plt.savefig(save_path + f"Local_map_{counter}.svg")


