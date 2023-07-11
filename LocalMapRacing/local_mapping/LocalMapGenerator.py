import numpy as np
import matplotlib.pyplot as plt
import LocalMapRacing.tph_utils as tph
from matplotlib.collections import LineCollection
np.set_printoptions(precision=4)
from LocalMapRacing.local_mapping.local_map_utils import *
from LocalMapRacing.local_mapping.LocalMap import LocalMap, PlotLocalMap


DISTNACE_THRESHOLD = 1.4 # distance in m for an exception
TRACK_WIDTH = 1.9 # use fixed width
POINT_SEP_DISTANCE = 0.8


class LocalMapGenerator:
    def __init__(self, path, map_name) -> None:
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)
        self.sines = np.sin(self.angles)
        self.xs, self.ys = None, None 

        self.local_map_data_path = path + f"LocalMapData_{map_name.upper()}/"
        ensure_path_exists(self.local_map_data_path)
        self.counter = 0

    def generate_line_local_map(self, scan, save=True):
        xs_f = self.coses * scan
        ys_f = self.sines * scan

        pts, pt_distances, inds = self.extract_track_lines(xs_f, ys_f)
        long_side, short_side = self.extract_boundaries(pts, pt_distances, inds)
        # track = self.estimate_center_line_devel(long_side, short_side, xs=xs_f, ys=ys_f)
        left_pts, right_pts = self.estimate_center_line_clean(long_side, short_side)
        true_center_line = (left_pts + right_pts) / 2
        # track = self.build_true_track(left_epts, right_pts)
        smooth_track = self.build_smooth_track(left_pts, right_pts)
        # track = self.project_side_to_track(long_side)

        local_map = PlotLocalMap(smooth_track)
        # lm = LocalMap(track)
        local_map.plot_local_map(xs=xs_f, ys=ys_f)

        plt.plot(left_pts[:, 0], left_pts[:, 1], 'x', color='black', markersize=10)
        plt.plot(right_pts[:, 0], right_pts[:, 1], 'x', color='black', markersize=10)
        
        plt.plot(true_center_line[:, 0], true_center_line[:, 1], '*', color='green', markersize=10)


        plt.show()


        if save: np.save(self.local_map_data_path + f"local_map_{self.counter}", local_map.track)
        self.counter += 1

        return local_map
    
    def extract_track_lines(self, xs, ys):
        pts = np.hstack((xs[:, None], ys[:, None]))
        pts = pts[pts[:, 0] > -2] # remove points behind the car
        pts = pts[np.logical_or(pts[:, 0] > 0, np.abs(pts[:, 1]) < 2)] # remove points behind the car or too far away
        pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        inds = np.array(np.where(pt_distances > DISTNACE_THRESHOLD))
        exclusion_zone = 2
        length = len(pts)
        inds = np.delete(inds, np.where(inds <= exclusion_zone)) 
        inds = np.delete(inds, np.where(inds >= length-exclusion_zone)) 

        if len(inds) == 0:
            raise IOError("Problem with full scan, no gaps found")

        return pts, pt_distances, inds

    def extract_boundaries(self, pts, pt_distances, inds):
        arr_inds = np.arange(len(pt_distances))[inds]
        min_ind = np.min(arr_inds) + 1
        max_ind = np.max(arr_inds) + 1

        l1_cs = np.cumsum(pt_distances[:min_ind-1])
        l2_cs = np.cumsum(pt_distances[max_ind:])

        if l1_cs[-1] > l2_cs[-1]:
            long_pts = pts[:min_ind]
            long_length = l1_cs[-1]

            short_pts = pts[max_ind:]
            short_pts = short_pts[::-1]
            short_length = l2_cs[-1]
        else:
            long_pts = pts[max_ind:]
            long_length = l2_cs[-1]

            short_pts = pts[:min_ind]
            short_length = l1_cs[-1]

        smoothing_s = 0.5
        n_pts = int(short_length / POINT_SEP_DISTANCE)
        short_side = interpolate_track_new(short_pts, None, smoothing_s)
        short_side = interpolate_track_new(short_side, n_pts*2, 0)

        n_pts = int(long_length / POINT_SEP_DISTANCE)
        long_side = interpolate_track_new(long_pts, None, smoothing_s)
        long_side = interpolate_track_new(long_side, n_pts*2, 0)
        #NOTE: the double interpolation ensures that the lengths are correct.
        # the first step smooths the points and the second step ensures the correct spacing.
        # TODO: this could be achieved using the same tck and just recalculating the s values based on the new lengths. Do this for computational speedup.

        return long_side, short_side
    
    def project_side_to_track(self, side):
        side_lm = LocalMap(side)
        center_line = side + side_lm.nvecs * TRACK_WIDTH / 2
        n_pts = int(side.shape[0] / 2)
        center_line = interpolate_track(center_line, n_pts, 0)

        center_line = center_line[center_line[:, 0] > -1] # remove points behind the car

        pt_init = np.linalg.norm(center_line[0, :])
        pt_final = np.linalg.norm(center_line[-1, :])
        if pt_final < pt_init: center_line = np.flip(center_line, axis=0)

        ws = np.ones_like(center_line) * TRACK_WIDTH / 2
        track = np.concatenate((center_line, ws), axis=1)

        return track

    def estimate_center_line_devel(self, long_side, short_side, xs=None, ys=None):
        plt.figure(1)
        plt.clf()
        if xs is not None and ys is not None:
            plt.plot(xs, ys, '.', color='#45aaf2', alpha=0.1)

        plt.plot(long_side[:, 0], long_side[:, 1], '.', markersize=10, color="#20bf6b")
        plt.plot(short_side[:, 0], short_side[:, 1], '.', markersize=10, color="#20bf6b")
        plt.plot(0, 0, '.', markersize=12, color='red')

        long_bound =  TrackBoundary(long_side)
        short_bound = TrackBoundary(short_side)
        
        center_pt = np.zeros(2)
        step_size = 0.6
        theta = 0
        left_pts = []
        right_pts = []
        at_the_end = False
        while not at_the_end and len(left_pts) < 20:
            long_pt = long_bound.find_closest_point(center_pt)
            short_pt = short_bound.find_closest_point(center_pt)

            left_pts.append(long_pt)
            right_pts.append(short_pt)
            
            long_distance = np.linalg.norm(long_pt - long_bound.points[-1])
            short_distance = np.linalg.norm(short_pt - short_bound.points[-1])
            threshold = 0.1
            if long_distance < threshold and short_distance < threshold:
                at_the_end = True

            n_diff = long_pt - short_pt
            heading = np.arctan2(n_diff[1], n_diff[0])
            new_theta = heading - np.pi/2
            d_theta = new_theta - theta
            theta = new_theta

            line_center = (long_pt + short_pt) / 2
            weighting = np.clip(abs(d_theta) / 0.2, 0, 0.8)
            print(f"Weighting: {weighting}")
            if d_theta > 0:
                adjusted_line_center = (short_pt * (weighting) + line_center * (1- weighting)) 
            else:
                adjusted_line_center = (long_pt * (weighting) + line_center * (1- weighting)) 

            print(f"Line c: {line_center} --> adjusted: {adjusted_line_center}")

            center_pt = adjusted_line_center + step_size * np.array([np.cos(theta), np.sin(theta)])

            plt.plot(center_pt[0], center_pt[1], '*', color='blue', markersize=10)
            plt.plot(line_center[0], line_center[1], '*', color='orange', markersize=10)
            plt.plot(adjusted_line_center[0], adjusted_line_center[1], '*', color='purple', markersize=10)
            c_xs = [adjusted_line_center[0], center_pt[0]]
            c_ys = [adjusted_line_center[1], center_pt[1]]
            plt.plot(c_xs, c_ys, '-', color='purple')

            plt.plot(long_pt[0], long_pt[1], 'x', color='black', markersize=10)
            plt.plot(short_pt[0], short_pt[1], 'x', color='black', markersize=10)
            xs = [long_pt[0], short_pt[0]]
            ys = [long_pt[1], short_pt[1]]
            plt.plot(xs, ys, '-', color='black')


        plt.axis('equal')
        plt.show()

    def estimate_center_line_clean(self, long_side, short_side):
        long_bound =  TrackBoundary(long_side)
        short_bound = TrackBoundary(short_side)
        
        center_pt = np.zeros(2)
        step_size = 0.6
        theta = 0
        left_pts = []
        right_pts = []
        at_the_end = False
        while not at_the_end and len(left_pts) < 30:
            long_pt = long_bound.find_closest_point(center_pt)
            short_pt = short_bound.find_closest_point(center_pt)

            left_pts.append(long_pt)
            right_pts.append(short_pt)
            
            long_distance = np.linalg.norm(long_pt - long_bound.points[-1])
            short_distance = np.linalg.norm(short_pt - short_bound.points[-1])
            threshold = 0.1
            if long_distance < threshold and short_distance < threshold:
                at_the_end = True

            n_diff = long_pt - short_pt
            heading = np.arctan2(n_diff[1], n_diff[0])
            new_theta = heading - np.pi/2
            d_theta = new_theta - theta
            theta = new_theta

            line_center = (long_pt + short_pt) / 2
            weighting = np.clip(abs(d_theta) / 0.2, 0, 0.8)
            if d_theta > 0:
                adjusted_line_center = (short_pt * (weighting) + line_center * (1- weighting)) 
            else:
                adjusted_line_center = (long_pt * (weighting) + line_center * (1- weighting)) 

            center_pt = adjusted_line_center + step_size * np.array([np.cos(theta), np.sin(theta)])

        left_pts = np.array(left_pts)
        right_pts = np.array(right_pts)

        return left_pts, right_pts

    # def build_true_track(self, left_pts, right_pts):
    #     c_line = (left_pts + right_pts) / 2
    #     ws = np.ones_like(c_line) * np.linalg.norm(c_line - left_pts, axis=1)[:, None]

    #     track = np.concatenate((c_line, ws), axis=1)

    #     return track

    def build_smooth_track(self, left_pts, right_pts):
        c_line = np.zeros_like(left_pts)
        c_line[0] = (left_pts[0] + right_pts[0]) / 2
        search_size = 2
        for i in range(1, len(left_pts)):
            diff = (left_pts[i-1] - right_pts[i-1])
            heading = np.arctan2(diff[1], diff[0])
            new_theta = heading - np.pi/2

            line1 = [left_pts[i], right_pts[i]]
            line2 = [c_line[i-1], c_line[i-1] + np.array([np.cos(new_theta), np.sin(new_theta)]) * search_size]

            intersection = calculate_intersection(line1, line2)
            c_line[i] = intersection

        ws_1 = np.linalg.norm(c_line - left_pts, axis=1)[:, None]
        ws_2 = np.linalg.norm(c_line - right_pts, axis=1)[:, None]

        track = np.concatenate((c_line, ws_1, ws_2), axis=1)

        return track


from scipy import optimize
from scipy import spatial
class TrackBoundary:
    def __init__(self, points) -> None:
        dists = np.linalg.norm(points - np.zeros(2), axis=1)
        if dists[0] > dists[-1]:
            self.points = np.flip(points, axis=0)
        else:
            self.points = points

        self.el = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cs = np.insert(np.cumsum(self.el), 0, 0)

        self.tck = interpolate.splprep([self.points[:, 0], self.points[:, 1]], k=3, s=0)[0]

    def find_closest_point(self, pt):
        dists = np.linalg.norm(self.points - pt, axis=1)
        closest_ind = np.argmin(dists)
        t_guess = self.cs[closest_ind] / self.cs[-1]
        # print(f"Closest ind: {closest_ind}, cs[ind]: {self.cs[closest_ind]};; guess pt: {self.points[closest_ind]}")

        # print(f"Init pt: {pt}, t_guess: {t_guess}")
        closest_t = optimize.fmin(dist_to_p, x0=t_guess, args=(self.tck, pt), disp=False)

        closest_pt = np.array(interpolate.splev(closest_t, self.tck, ext=3)).T[0]
        # print(f"Closest t: {closest_t} --> closest pt: {closest_pt}")

        return closest_pt
        
    def plot_line(self):
        ss = np.linspace(0, 1, 100)
        new_pts = np.array(interpolate.splev(ss, self.tck)).T
        plt.plot(new_pts[:, 0], new_pts[:, 1])
        plt.plot(self.points[:, 0], self.points[:, 1], 'x', color='blue', markersize=10)


def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
    s = interpolate.splev(t_glob, path)
    s = np.concatenate(s)
    return spatial.distance.euclidean(p, s)

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

if __name__ == "__main__":
    pass