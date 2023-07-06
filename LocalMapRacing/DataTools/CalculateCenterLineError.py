import numpy as np
from matplotlib import pyplot as plt
import csv
from scipy import interpolate
from LocalMapRacing.planner_utils.utils import ensure_path_exists


class MapCenterline:
    def __init__(self, map_name):
        track = []
        with open("maps/" + map_name + "_trueCenterline.csv", 'r') as file:
            csvFile = csv.reader(file)
            for i, line in enumerate(csvFile):
                if i ==0: continue
                track.append(line)

        track = np.array(track).astype(np.float64)
        self.track = np.insert(track, 0, [0, 0, 0.95, 0.95, 0, -1], axis=0)
        self.nvecs = self.track[:, 4:6]

        diffs = np.diff(self.track[:, :2], axis=0)
        seg_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.s_track = np.insert(np.cumsum(seg_lengths), 0, 0)

        self.tck, t_glob = interpolate.splprep([self.track[:, 0], self.track[:, 1]], k=3, s=0)[:2]

        print(self.track[0])

    def plot_centerline(self):
        plt.figure(1)
        plt.plot(self.track[:,0], self.track[:,1], 'r-')
        l1 = self.track[:, :2] + self.track[:, 2][:, None] * self.nvecs
        l2 = self.track[:, :2] - self.track[:, 3][:, None] * self.nvecs
        plt.plot(l1[:,0], l1[:,1], 'b-')
        plt.plot(l2[:,0], l2[:,1], 'b-')

        plt.show()

    def calucalte_center_point(self, point):
        idx, distances = self.get_trackline_segment(point)
        x, h = self.interp_pts(idx, distances)
        s = (self.s_track[idx] + x) / self.s_track[-1]

        c_point = interpolate.splev(s, self.tck)
        h_true = np.linalg.norm(c_point - point)

        return c_point, s, h_true

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

    def interp_pts(self, idx, dists):
        if idx == len(self.s_track) - 1:
            return 0, 0
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

    def plot_projection(self, pt):
        c_point, s, h_true = self.calucalte_center_point(pt)
        plt.plot(c_point[0], c_point[1], 'ro')
        plt.plot(pt[0], pt[1], 'go', markersize=5)
        xs = [pt[0], c_point[0]]
        ys = [pt[1], c_point[1]]
        plt.plot(xs, ys, '--', color='orange')

def calculate_centerline_area_error(name):
    map_name = "aut"
    map_centerline = MapCenterline(map_name)

    path = f"Data/{name}/"

    lm_path = path + "LocalMapError/"
    ensure_path_exists(lm_path)
    
    map_root = path + f"LocalMapData_{map_name.upper()}/"
    history = np.load(path + "TestingAUT/" + f"Lap_0_history_{name}.npy")
    states = history[:, 0:7]
    actions = history[:, 7:9]

    lm_errors = []
    for i in range(1, 260):
        file = map_root + f"local_map_{i}.npy"
        try:
            local_track = np.load(file)
        except Exception as e: 
            print(e)
            break

        distances, progresses = [], []
        c_points = []
        position = states[i, 0:2]
        transformed_lm = []
        heading = states[i, 4]
        rotation_matrix = np.array([[np.cos(heading), -np.sin(heading)], 
                                [np.sin(heading), np.cos(heading)]])
        t_lm = np.matmul(rotation_matrix, local_track[:, :2].T).T
        for k in range(len(local_track)):
            pt = position + t_lm[k]
            transformed_lm.append(pt)
            c_point, s, h_true = map_centerline.calucalte_center_point(pt)
            distances.append(h_true)
            progresses.append(s)
            c_points.append(c_point)

        mean_dist = np.mean(distances)
        print(f"Step {i}: {mean_dist}")
        if mean_dist < 2:
            lm_errors.append(mean_dist)

    np.save(path + f"local_map_error_{name}.npy", lm_errors)

    plt.plot(lm_errors)

    plt.savefig(path + f"local_map_error_{name}.svg")


def calculate_centerline_area_error_plot(name):
    map_name = "aut"
    map_centerline = MapCenterline(map_name)

    path = f"Data/{name}/"

    lm_path = path + "LocalMapError/"
    ensure_path_exists(lm_path)
    
    map_root = path + f"LocalMapData_{map_name.upper()}/"
    history = np.load(path + "TestingAUT/" + f"Lap_0_history_{name}.npy")
    states = history[:, 0:7]
    actions = history[:, 7:9]

    lm_errors = []
    for i in range(1, 260):
        file = map_root + f"local_map_{i}.npy"
        try:
            local_track = np.load(file)
        except: break

        plt.figure(1)
        plt.clf()
        plt.title(f"Step {i}")
        plt.plot(map_centerline.track[:,0], map_centerline.track[:,1], 'r-')

        distances, progresses = [], []
        c_points = []
        position = states[i, 0:2]
        transformed_lm = []
        heading = states[i, 4]
        rotation_matrix = np.array([[np.cos(heading), -np.sin(heading)], 
                                [np.sin(heading), np.cos(heading)]])
        t_lm = np.matmul(rotation_matrix, local_track[:, :2].T).T
        for k in range(len(local_track)):
            pt = position + t_lm[k]
            transformed_lm.append(pt)
            c_point, s, h_true = map_centerline.calucalte_center_point(pt)
            distances.append(h_true)
            progresses.append(s)
            c_points.append(c_point)
            map_centerline.plot_projection(pt)

        transformed_lm = np.array(transformed_lm)
        plt.plot(transformed_lm[:,0], transformed_lm[:,1], 'g-')
        plt.xlim(np.min(transformed_lm[:, 0]) - 1, np.max(transformed_lm[:, 0]) + 1)
        plt.ylim(np.min(transformed_lm[:, 1]) - 1, np.max(transformed_lm[:, 1]) + 1)
        # plt.show()

        plt.savefig(lm_path + f"local_map_error_{i}.svg")


    np.save(path + f"local_map_error_{name}.npy", lm_errors)

    plt.plot(lm_errors)

    plt.savefig(path + f"local_map_error_{name}.svg")


# calculate_centerline_area_error("LocalCenter_1")
calculate_centerline_area_error_plot("LocalCenter_1")