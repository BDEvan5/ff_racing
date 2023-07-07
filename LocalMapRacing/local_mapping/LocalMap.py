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
        tck = interpolate.splprep([self.track[:, 0], self.track[:, 1]], k=3, s=0)[0]
        idx, distances = self.get_trackline_segment(point)
        x, h = self.interp_pts(idx, distances)
        s = (self.s_track[idx] + x) 

        c_point = interpolate.splev(s/self.s_track[-1], tck)
        h_true = np.linalg.norm(c_point - point)

        return s, h_true, c_point

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
    

    def get_trackline_segment(self, point):
        """Returns the first index representing the line segment that is closest to the point.

        wpt1 = pts[idx]
        wpt2 = pts[idx+1]

        dists: the distance from the point to each of the wpts.
        """
        dists = np.linalg.norm(point - self.track[:, :2], axis=1)

        min_dist_segment = np.argmin(dists)
        if min_dist_segment == 0:
            return 0, dists
        elif min_dist_segment == len(dists)-1:
            return len(dists)-2, dists 

        if dists[min_dist_segment+1] < dists[min_dist_segment-1]:
            return min_dist_segment, dists
        else: 
            return min_dist_segment - 1, dists

    def interpolate_track(self, spacing=0.8):
        new_s = np.linspace(0, self.s_track[-1], int(self.s_track[-1]/spacing))
        self.track = interp_nd_points(new_s, self.s_track, self.track)

        self.calculate_length_heading_nvecs()

    def interpolate_track_scipy(self, n_pts=None, s=0):
        ws = np.ones_like(self.track[:, 0])
        ws[0:2] = 100
        ws[-2:] = 100
        tck = interpolate.splprep([self.track[:, 0], self.track[:, 1]], w=ws, k=3, s=s)[0]
        if n_pts is None: n_pts = len(self.track)
        self.track[:, :2] = np.array(interpolate.splev(np.linspace(0, 1, n_pts), tck)).T

        self.calculate_length_heading_nvecs()
    
    def extend_track_end(self, n_pts, distance=0.8):
        nvec_forward = self.nvecs[-1][::-1]
        nvec_forward[1] = - nvec_forward[1]
        extended_track = np.copy(self.track)
        for i in range(1, n_pts+1):
            new_pt = np.copy(self.track[-1])
            new_pt[:2] = new_pt[:2] + nvec_forward * distance * i
            extended_track = np.append(extended_track, new_pt.reshape(1, -1), axis=0)

        self.track = extended_track
        self.calculate_length_heading_nvecs()

    def build_smooth_track(self, counter, path):
        crossing_horizon = min(5, len(self.track)//2 -1)
        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, crossing_horizon)
        
        if not crossing: return
        self.interpolate_track(0.4)
        old_track = np.copy(self.track)
        old_nvecs = np.copy(self.nvecs)
        
        tb_l = self.track[:, :2] + self.nvecs * self.track[:, 2].reshape(-1, 1)
        tb_r = self.track[:, :2] - self.nvecs * self.track[:, 3].reshape(-1, 1)

        # fig, axs = plt.subplots(2, 1, figsize=(10, 10), num=5)
        # # axs[0].plot(self.s_track, self.kappa, 'k')
        # # axs[1].plot(self.s_track[:-1], np.diff(self.kappa), '')

        # axs[0].plot(self.s_track, self.kappa, 'r')
        # axs[1].plot(self.s_track[:-1], np.diff(self.kappa), 'r')
        
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
        # self.plot_smoothing(old_track, old_nvecs, counter, path)
        # plt.show()

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


        # self.calculate_length_heading_nvecs()
        # fig, axs = plt.subplots(2, 1, figsize=(10, 10), num=5)

        # axs[0].plot(self.s_track, self.kappa, 'b')
        # axs[1].plot(self.s_track[:-1crossing_horizon = min(5, len(self.track)//2 -1)
        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, crossing_horizon)
        
        if not crossing: return
        self.interpolate_track(0.4)
        old_track = np.copy(self.track)
        old_nvecs = np.copy(self.nvecs)
        
        no_points_track = len(self.track)
        ws = np.ones_like(self.track[:, 0])
        ws[0:2] = 100
        ws[-2:] = 100
        tck = interpolate.splprep([self.track[:, 0], self.track[:, 1]], w=ws, k=3, s=0.1)[0]
        smooth_path = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_track), tck, ext=0)).T
        self.track[:, :2] = smooth_path

        self.calculate_length_heading_nvecs()

        for i in range(1, no_points_track-1):
            if np.abs(self.kappa[i]) < 0.2: continue

            distance_magnitude =  (np.abs(self.kappa[i]) + 1) **0.3 - 1
            d_pt = self.nvecs[i] * distance_magnitude * np.sign(self.kappa[i])

            self.track[i, :2] += d_pt
            self.track[i, 2] -= distance_magnitude * np.sign(self.kappa[i])
            self.track[i, 3] += distance_magnitude * np.sign(self.kappa[i])

        self.calculate_length_heading_nvecs()

        self.plot_smoothing(old_track, old_nvecs)

        plt.pause(0.001)


    def adjust_center_line_smoothing(self, counter, path):
        crossing_horizon = min(5, len(self.track)//2 -1)
        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, crossing_horizon)
        
        if not crossing: return
        self.interpolate_track(0.8)
        old_track = np.copy(self.track)
        old_nvecs = np.copy(self.nvecs)
        
        # ws = np.ones_like(self.track[:, 0])
        # ws[0:2] = 100
        # ws[-2:] = 100
        # tck = interpolate.splprep([self.track[:, 0], self.track[:, 1]], w=ws, k=3, s=0.1)[0]
        # smooth_path = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_track), tck, ext=0)).T
        # self.track[:, :2] = smooth_path
        self.interpolate_track_scipy(None, 0.1)

        self.calculate_length_heading_nvecs()

        no_points_track = len(self.track)
        for i in range(1, no_points_track-1):
            if np.abs(self.kappa[i]) < 0.2: continue

            distance_magnitude =  (np.abs(self.kappa[i]) + 1) **0.3 - 1
            d_pt = self.nvecs[i] * distance_magnitude * np.sign(self.kappa[i])

            self.track[i, :2] += d_pt
            self.track[i, 2] -= distance_magnitude * np.sign(self.kappa[i])
            self.track[i, 3] += distance_magnitude * np.sign(self.kappa[i])

        self.calculate_length_heading_nvecs()

        self.plot_smoothing(old_track, old_nvecs, counter, path)

        crossing_horizon = min(5, len(self.track)//2 -1)
        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, crossing_horizon)

        if crossing:
            print(f'Crossing after smoothing: {counter}')

        # plt.pause(0.001)


    def apply_required_smoothing(self, counter, path):
        self.interpolate_track(0.4)
        
        old_track = np.copy(self.track)
        old_nvecs = np.copy(self.nvecs)

        self.extend_track_end(1, 0.5)
        no_points_track = len(self.track)

        crossing_horizon = min(5, len(self.track)//2 -1)

        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, crossing_horizon)
        if not crossing: return

        i = 0
        smoothing = 0
        refactored = False
        while i < 50 and crossing:
            i += 1
            smoothing = i**2 *0.1
            if smoothing > 1 and not refactored:
                self.track = self.track[:-1]
                self.calculate_length_heading_nvecs()
                self.interpolate_track(0.8)
                self.extend_track_end(1, 0.8)
                no_points_track = len(self.track)
                refactored = True
            if len(self.track) < 6:
                print("Track too short")
                return
            ws = np.ones(len(self.track))
            ws[0] = 1000
            ws[1] = 1000
            ws[-2] = 1000
            ws[-1] = 1000
            tck, t_glob = interpolate.splprep([self.track[:, 0], self.track[:, 1]], w=ws, k=5, s=smoothing)[:2]

            # Over extend the smooth path so that the true path length can be found
            smooth_path = np.array(interpolate.splev(np.linspace(0.0, 1.2, no_points_track*4), tck, ext=0)).T
            dists = np.linalg.norm(smooth_path - self.track[-2, :2], axis=1)
            min_point = np.argmin(dists, axis=0)
            smooth_path = smooth_path[:min_point+1]

            # get normal path 
            tck = interpolate.splprep([smooth_path[:, 0], smooth_path[:, 1]], k=3, s=0)[0]
            smooth_path = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_track), tck, ext=0)).T

            # find new widths
            # closest_t_glob = np.zeros(no_points_track)
            # closest_point = np.zeros((no_points_track, 2))
            # dists = np.zeros(len(self.track))
            # t_glob_guess = self.s_track / self.s_track[-1]
            # for z in range(no_points_track):
            #     closest_t_glob[z] = optimize.fmin(dist_to_p, x0=t_glob_guess[z], args=(tck, self.track[z, :2]), disp=False)

            #     closest_point[z] = interpolate.splev(closest_t_glob[z], tck)
            #     dists[z] = np.linalg.norm(closest_point[z] - self.track[z, :2])

            closest_point = np.zeros((no_points_track, 2))
            dists = np.zeros(no_points_track)
            for z in range(no_points_track):
                s, h, pt = self.calculate_s(smooth_path[z, :2])
                closest_point[z] = pt
                dists[z] = h

            sides = np.zeros(no_points_track)
            for z in range(1, no_points_track):
                sides[z-1] = side_of_line(a=self.track[z-1, :2], b=self.track[z, :2], z=closest_point[z])
                
            w_tr_right_new = self.track[:, 2] + sides * dists /2
            w_tr_left_new = self.track[:, 3] - sides * dists /2

            smooth_track = np.hstack((smooth_path, w_tr_left_new.reshape(-1, 1), w_tr_right_new.reshape(-1, 1)))

            # self.track = smooth_track
            # self.calculate_length_heading_nvecs()

            crossing = check_normals_crossing_complete(smooth_track)

        print(f"{counter} Smoothing: ", smoothing)
        self.track = smooth_track
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


def check_normals_crossing_complete(track):
    crossing_horizon = min(5, len(track)//2 -1)

    el_lengths = np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1)
    s_track = np.insert(np.cumsum(el_lengths), 0, 0)
    psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(track, el_lengths, False)
    nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi-np.pi/2)

    crossing = tph.check_normals_crossing.check_normals_crossing(track, nvecs, crossing_horizon)

    return crossing
