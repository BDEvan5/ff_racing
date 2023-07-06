import numpy as np
import matplotlib.pyplot as plt
import LocalMapRacing.tph_utils as tph
from LocalMapRacing.local_mapping.local_map_utils import *


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

    def interpolate_track(self, spacing=0.8):
        new_s = np.linspace(0, self.s_track[-1], int(self.s_track[-1]/spacing))
        self.track = interp_nd_points(new_s, self.s_track, self.track)

        self.calculate_length_heading_nvecs()

    def plot_local_map(self):
        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]

        plt.figure(1)
        plt.clf()
        plt.plot(self.track[:, 0], self.track[:, 1], '-', color='#E74C3C', label="Center", linewidth=3)
        plt.plot(0, 0, 'x', color='black', markersize=10)

        plt.plot(l1[:, 0], l1[:, 1], color='#ffa700')
        plt.plot(l2[:, 0], l2[:, 1], color='#ffa700')

        for i in range(len(self.track)):
            xs = [l1[i, 0], l2[i, 0]]
            ys = [l1[i, 1], l2[i, 1]]
            plt.plot(xs, ys, '#ffa700')

        plt.axis('equal')
        plt.show()

    def build_smooth_track(self):
        crossing_horizon = min(5, len(self.track)//2 -1)
        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, crossing_horizon)
        
        if not crossing: return
        self.interpolate_track(0.4)
        old_track = np.copy(self.track)
        old_nvecs = np.copy(self.nvecs)
        
        tb_l = self.track[:, :2] + self.nvecs * self.track[:, 2].reshape(-1, 1)
        tb_r = self.track[:, :2] - self.nvecs * self.track[:, 3].reshape(-1, 1)

        # plt.pause(0.001)

        d_kappa = np.diff(self.kappa)
        n_anlges = self.psi - np.pi/2
        for i in range(1, len(d_kappa)):
            if abs(self.kappa[i]) > 0.8:
                if d_kappa[i] > 0:
                    n_anlges[i] += 0.2
                    n_anlges[i+1] += 0.15
                else:
                    n_anlges[i-1] -= 0.15
                    n_anlges[i] -= 0.22

        self.nvecs[:, 0] = np.cos(n_anlges)
        self.nvecs[:, 1] = np.sin(n_anlges)

        # old_track = np.copy(self.track)
        # old_nvecs = np.copy(self.nvecs)

        tb_l = self.track[:, :2] + self.nvecs * self.track[:, 2].reshape(-1, 1)
        tb_r = self.track[:, :2] - self.nvecs * self.track[:, 3].reshape(-1, 1)

        frozen_track = np.copy(self.track)
        for i in range(len(self.track)-1):
            forward_vec = np.array([-self.nvecs[i, 1], self.nvecs[i, 0]])
            forward_pt = self.track[i, :2] + forward_vec * 1 # search distance
            forward_line = np.array([self.track[i, :2], forward_pt])
            nvec_line = np.array([tb_l[i+1], tb_r[i+1]])

            intersection_pt = calculate_intersection(forward_line, nvec_line)
            if intersection_pt is None: raise ValueError('No intersection found')

            distance = np.linalg.norm(intersection_pt - self.track[i+1, :2])
            # sign = np.sign(np.dot(intersection_pt - self.track[i+1, :2], self.nvecs[i+1]))
            sign = side_of_line(self.track[i, :2], self.track[i+1, :2], intersection_pt)
            self.track[i+1, :2] = intersection_pt
            self.track[i+1, 2] += distance * sign
            self.track[i+1, 3] -= distance * sign


        self.plot_smoothing(old_track, old_nvecs)
        # plt.show()
        plt.pause(0.001)

    def plot_smoothing(self, old_track, old_nvecs):
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


def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
    s = interpolate.splev(t_glob, path)
    return spatial.distance.euclidean(p, s)

from typing import Union
def side_of_line(a: Union[tuple, np.ndarray],
                 b: Union[tuple, np.ndarray],
                 z: Union[tuple, np.ndarray]) -> float:

    side = np.sign((b[0] - a[0]) * (z[1] - a[1]) - (b[1] - a[1]) * (z[0] - a[0]))

    return side

def do_lines_intersect(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # Calculate the slopes of the lines
    slope1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    slope2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')

    # Check if the lines are parallel
    if slope1 == slope2:
        return False

    # Calculate the y-intercepts of the lines
    intercept1 = y1 - slope1 * x1
    intercept2 = y3 - slope2 * x3

    # Calculate the intersection point (x, y)
    x = (intercept2 - intercept1) / (slope1 - slope2) if slope1 != float('inf') and slope2 != float('inf') else float('inf')
    y = slope1 * x + intercept1 if slope1 != float('inf') else slope2 * x + intercept2

    # Check if the intersection point lies within the line segments
    if (min(x1, x2) <= x <= max(x1, x2)) and (min(y1, y2) <= y <= max(y1, y2)) and \
       (min(x3, x4) <= x <= max(x3, x4)) and (min(y3, y4) <= y <= max(y3, y4)):
        return True

    return False

def calculate_intersection(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # Calculate the slopes of the lines
    slope1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    slope2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')

    # Check if the lines are parallel
    if slope1 == slope2:
        return None

    # Calculate the y-intercepts of the lines
    intercept1 = y1 - slope1 * x1
    intercept2 = y3 - slope2 * x3

    # Calculate the intersection point (x, y)
    x = (intercept2 - intercept1) / (slope1 - slope2) if slope1 != float('inf') and slope2 != float('inf') else float('inf')
    y = slope1 * x + intercept1 if slope1 != float('inf') else slope2 * x + intercept2

    return np.array([x, y])


def make_c_line():
    pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4.2, 0.5], [4.5, 1], [4.5, 2], [4.5, 3], [5, 4], [6, 4], [7, 4]])
    width = 1.5
    ws = np.ones_like(pts) * width 
    track = np.hstack((pts, ws))
    local_map = LocalMap(track)
    local_map.interpolate_track(0.4)

    local_map.build_smooth_track()


    local_map.plot_local_map()

make_c_line()
