import numpy as np
import matplotlib.pyplot as plt
import LocalMapRacing.tph_utils as tph
from matplotlib.collections import LineCollection
np.set_printoptions(precision=4)
from LocalMapRacing.local_mapping.local_map_utils import *
from LocalMapRacing.local_mapping.LocalMap import LocalMap, PlotLocalMap

from scipy import optimize
from scipy import spatial
from numba import njit

DISTNACE_THRESHOLD = 1.4 # distance in m for an exception
TRACK_WIDTH = 1.7 # use fixed width
POINT_SEP_DISTANCE = 0.8

PLOT_DEVEL = True
# PLOT_DEVEL = False

class LocalMapGenerator:
    def __init__(self, path, map_name) -> None:
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)
        self.sines = np.sin(self.angles)
        self.scan_xs, self.scan_ys = None, None 

        self.local_map_data_path = path + f"LocalMapData_{map_name.upper()}/"
        ensure_path_exists(self.local_map_data_path)
        self.local_map_imgs = path + f"LocalMapGenerators_{map_name.upper()}/"
        ensure_path_exists(self.local_map_imgs)
        self.counter = 0

        self.line_1 = None
        self.line_2 = None
        self.max_s_1 = 0
        self.max_s_2 = 0
        self.boundary_1 = None
        self.boundary_2 = None
        self.boundary_extension_1 = None
        self.boundary_extension_2 = None
        self.smooth_track = None

    def generate_line_local_map(self, scan, save=True, counter=None):
        self.scan_xs = self.coses * scan
        self.scan_ys = self.sines * scan

        pts, pt_distances, inds = self.extract_track_lines()
        self.extract_boundaries(pts, pt_distances, inds)
        self.estimate_center_line_dual_boundary()
        self.extend_center_line_projection()

        local_track = self.build_local_track()
        local_map = PlotLocalMap(local_track)
        # lm = LocalMap(track)

        self.smooth_track = local_map
        if counter != None:
            self.plot_local_map_generation(counter)

        if save: np.save(self.local_map_data_path + f"local_map_{self.counter}", local_map.track)
        self.counter += 1

        return local_map

    def extract_track_lines(self):
        pts = np.hstack((self.scan_xs[:, None], self.scan_ys[:, None]))
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
        min_ind = np.min(arr_inds) +1
        max_ind = np.max(arr_inds) + 1

        line_1_pts = pts[:min_ind]
        line_2_pts = pts[max_ind:]
        i = 1
        while (np.all(line_1_pts[:, 0] < -0.8) or np.all(np.abs(line_1_pts[:, 1]) > 2.5)) and i < len(inds):
            min_ind2 = np.min(arr_inds[i:]) 
            line_1_pts = pts[min_ind+2:min_ind2]
            min_ind = min_ind2
            i += 1

        line_2_pts = pts[max_ind:]
        i = 1
        while (np.all(line_1_pts[:, 0] < -0.8) or np.all(np.abs(line_1_pts[:, 1]) > 2.5)) and i < len(inds):
            max_ind2 = np.max(arr_inds[:-i])
            line_1_pts = pts[max_ind2+2:max_ind]
            max_ind = max_ind2
            i += 1

        self.line_1 = TrackBoundary(line_1_pts, True)
        self.line_2 = TrackBoundary(line_2_pts, True)

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

    def estimate_center_line_dual_boundary(self):
        search_pt = [-1, 0]
        max_pts = 50
        end_threshold = 0.05
        self.boundary_1 = np.zeros((max_pts, 2))
        self.boundary_2 = np.zeros((max_pts, 2))
        center_pts = np.zeros((max_pts, 2))
        self.max_s_1, self.max_s_2 = 0, 0 
        for i in range(max_pts):
            if i == 0:
                pt_1, self.max_s_1 = self.line_1.find_closest_point(search_pt, self.max_s_1, "Line1")
                pt_2, self.max_s_2 = self.line_2.find_closest_point(search_pt, self.max_s_2, "Line2")
            else:
                pt_1, pt_2, search_pt = self.calculate_next_boundaries(pt_1, pt_2)

            line_center = (pt_1 + pt_2) / 2
            center_pts[i] = line_center

            if np.all(np.isclose(pt_1, self.boundary_1[i-1])) and np.all(np.isclose(pt_2, self.boundary_2[i-1])): 
                i -= 1
                break

            self.boundary_1[i] = pt_1
            self.boundary_2[i] = pt_2

            long_distance = np.linalg.norm(pt_1 - self.line_1.points[-1])
            short_distance = np.linalg.norm(pt_2 - self.line_2.points[-1])
            if long_distance < end_threshold and short_distance < end_threshold:
                # print(f"{i}-> Breaking because of long ({long_distance}) and short ({short_distance}) distances >>> Pt1: {pt_1} :: Pt2: {pt_2}")
                break

        self.boundary_1 = self.boundary_1[:i+1]
        self.boundary_2 = self.boundary_2[:i+1]

        if len(self.boundary_1) < 2:
            raise RuntimeError("Not enough points found -> {len(self.boundary_1)}")

    def extend_center_line_projection(self):
        if self.max_s_1 > 0.99 and self.max_s_2 > 0.99:
            self.boundary_extension_1 = None
            self.boundary_extension_2 = None
            return # no extension required
        
        true_center_line = (self.boundary_1 + self.boundary_2) / 2
        dists = np.linalg.norm(np.diff(true_center_line, axis=0), axis=1)
        center_point_threshold = 0.2
        removal_n = np.sum(dists < center_point_threshold)

        self.boundary_1 = self.boundary_1[:-removal_n]
        self.boundary_2 = self.boundary_2[:-removal_n]
        true_center_line = (self.boundary_1 + self.boundary_2) / 2

        if self.max_s_1 > self.max_s_2:
            projection_line = self.line_2
            boundary = self.boundary_2
            direction = -1
        else:
            projection_line = self.line_1
            boundary = self.boundary_1
            direction = 1

        _pt, current_s = projection_line.find_closest_point(boundary[-1], 0)
        length_remaining = (1-current_s[0]) * projection_line.cs[-1]
        if length_remaining < 1:
            self.boundary_extension_1 = None
            self.boundary_extension_2 = None
            return # no extension required

        step_size = 0.5
        n_pts = int((1-current_s[0]) * projection_line.cs[-1] / step_size + 1)
        new_boundary_points = projection_line.extract_line_portion(np.linspace(current_s[0], 1, n_pts))
        new_projection_line = LocalLine(new_boundary_points)

        # project to center line
        extra_center_line = new_projection_line.track + new_projection_line.nvecs * TRACK_WIDTH/2 * direction

        extra_projected_boundary = new_projection_line.track + new_projection_line.nvecs * TRACK_WIDTH * direction

        if self.max_s_1 > self.max_s_2:
            self.boundary_extension_1 = extra_projected_boundary
            self.boundary_extension_2 = new_projection_line.track
        else:
            self.boundary_extension_1 = new_projection_line.track
            self.boundary_extension_2 = extra_projected_boundary

        # remove last point since it has now been replaced.
        self.boundary_1 = self.boundary_1[:-1]
        self.boundary_2 = self.boundary_2[:-1]

    def calculate_next_boundaries(self, pt_1, pt_2):
        step_size = 0.6
        line_center = (pt_1 + pt_2) / 2
        theta = calculate_track_direction(pt_1, pt_2)

        weighting = 0.7
        search_pt_a = (pt_2 * (weighting) + line_center * (1- weighting)) 
        search_pt_b = (pt_1 * (weighting) + line_center * (1- weighting)) 
        search_pt_a = search_pt_a + step_size * np.array([np.cos(theta), np.sin(theta)])
        search_pt_b = search_pt_b + step_size * np.array([np.cos(theta), np.sin(theta)])

        pt_1_a, max_s_1_a = self.line_1.find_closest_point(search_pt_a, self.max_s_1, "Line1_a")
        pt_2_a, max_s_2_a = self.line_2.find_closest_point(search_pt_a, self.max_s_2, "Line2_a")

        pt_1_b, max_s_1_b = self.line_1.find_closest_point(search_pt_b, self.max_s_1, "Line1_b")
        pt_2_b, max_s_2_b = self.line_2.find_closest_point(search_pt_b, self.max_s_2, "Line2_b")

        # test to find the best candidate
        sum_s_a = max_s_1_a + max_s_2_a
        sum_s_b = max_s_1_b + max_s_2_b

        if sum_s_a < sum_s_b:
            pt_1, self.max_s_1 = pt_1_a, max_s_1_a
            pt_2, self.max_s_2 = pt_2_a, max_s_2_a
            search_pt = search_pt_a
        else:
            pt_1, self.max_s_1 = pt_1_b, max_s_1_b
            pt_2, self.max_s_2 = pt_2_b, max_s_2_b
            search_pt = search_pt_b

        return pt_1, pt_2, search_pt

    def build_local_track(self):
        """
        This generates the list of center line points that keep the nvecs the same as they were before
        - if the track is a dual-build track, then an extra point is added to preserve the last nvec
        - if the track is extended, the last nvec is adjusted to keep the correct directions.
        """
        if self.boundary_extension_1 is not None:
            boundary_1 = np.append(self.boundary_1, self.boundary_extension_1, axis=0)
            boundary_2 = np.append(self.boundary_2, self.boundary_extension_2, axis=0)

            nvecs = boundary_1 - boundary_2
            psi_nvecs = np.arctan2(nvecs[:, 1], nvecs[:, 0])
            psi_tanvecs = psi_nvecs + np.pi/2
            c_line = np.zeros((boundary_1.shape[0], 2))
        else:
            boundary_1 = self.boundary_1
            boundary_2 = self.boundary_2

            nvecs = boundary_1 - boundary_2
            psi_nvecs = np.arctan2(nvecs[:, 1], nvecs[:, 0])
            psi_tanvecs = psi_nvecs + np.pi/2

            # extension = 0.8 * np.array([np.cos(psi_tanvecs[-1]), np.sin(psi_tanvecs[-1])])[:, None]
            # pt_1 = (boundary_1[-1, :] + extension)
            # pt_2 = (boundary_2[-1, :] + extension)
            # boundary_1 = np.append(boundary_1, pt_1, axis=0)
            # boundary_2 = np.append(boundary_2, pt_2, axis=0)
            # c_line = np.zeros((boundary_1.shape[0] - 1, 2))
            c_line = np.zeros((boundary_1.shape[0], 2))

        c_line[0] = (boundary_1[0] + boundary_2[0]) / 2
        search_size = 2
        for i in range(1, len(c_line)):
            theta = psi_tanvecs[i-1]
            line1 = [boundary_1[i], boundary_2[i]]
            if i == 1:
                line2 = [c_line[i-1], c_line[i-1] + np.array([np.cos(theta), np.sin(theta)]) * search_size]
            else:
                line2 = [c_line[i-2], c_line[i-2] + np.array([np.cos(theta), np.sin(theta)]) * search_size]

            intersection = calculate_intersection(line1, line2)
            if intersection is None: # or intersection[0] == 1e9:
                raise ValueError(f"No intersection found between {line1} and {line2}")
            c_line[i] = intersection

        ws_1 = np.linalg.norm(c_line - boundary_1[:len(c_line)], axis=1)[:, None]
        ws_2 = np.linalg.norm(c_line - boundary_2[:len(c_line)], axis=1)[:, None]

        track = np.concatenate((c_line, ws_2, ws_1), axis=1)

        return track

    def build_true_center_track(self):
        true_center_line = (self.boundary_1 + self.boundary_2) / 2
        ws = np.linalg.norm(self.boundary_1 - true_center_line, axis=1)
        ws = ws[:, None] * np.ones_like(true_center_line)
        track = np.concatenate([true_center_line, ws], axis=1)

        return track

    def plot_local_map_generation(self, counter):
        plt.figure(2)
        plt.clf()

        plt.plot(self.scan_xs, self.scan_ys, '.', color='#45aaf2', alpha=0.2)
        plt.plot(0, 0, '*', markersize=12, color='red')

        self.line_1.plot_line()
        self.line_2.plot_line()

        true_center_line = (self.boundary_1 + self.boundary_2) / 2

        plt.plot(true_center_line[:, 0], true_center_line[:, 1], 'X', color='orange', markersize=10)
        for i in range(len(self.boundary_1)):
            xs = [self.boundary_1[i, 0], self.boundary_2[i, 0]]
            ys = [self.boundary_1[i, 1], self.boundary_2[i, 1]]
            plt.plot(xs, ys, '-x', color='black', markersize=10)
        
        if self.boundary_extension_1 is not None:
            extended_true_center = (self.boundary_extension_1 + self.boundary_extension_2) / 2
            plt.plot(extended_true_center[:, 0], extended_true_center[:, 1], 'X', color='pink', markersize=10)

            for i in range(len(self.boundary_extension_1)):
                xs = [self.boundary_extension_1[i, 0], self.boundary_extension_2[i, 0]]
                ys = [self.boundary_extension_1[i, 1], self.boundary_extension_2[i, 1]]
                plt.plot(xs, ys, '-x', color='blue', markersize=10)

        plt.plot(self.smooth_track.track[:, 0], self.smooth_track.track[:, 1], '-', color='red', linewidth=3)

        l1 = self.smooth_track.track[:, :2] + self.smooth_track.nvecs * self.smooth_track.track[:, 2][:, None] 
        l2 = self.smooth_track.track[:, :2] - self.smooth_track.nvecs * self.smooth_track.track[:, 3][:, None]

        plt.plot(l1[:, 0], l1[:, 1], '-', color='red', linewidth=1) 
        plt.plot(l2[:, 0], l2[:, 1], '-', color='red', linewidth=1) 

        for i in range(len(l1)):
            xs = [l1[i, 0], l2[i, 0]]
            ys = [l1[i, 1], l2[i, 1]]
            plt.plot(xs, ys, '-+', color='orange', markersize=10)

        plt.axis('equal')
        plt.tight_layout()
        name = self.local_map_imgs + f"LocalMapGeneration_{counter}"
        plt.savefig(name + ".svg", bbox_inches="tight")


class TrackBoundary:
    def __init__(self, points, smoothing=False) -> None:        
        self.smoothing_s = 0.5
        if points[0, 0] > points[-1, 0]:
            self.points = np.flip(points, axis=0)
        else:
            self.points = points

        if smoothing:
            self.apply_smoothing()

        self.el = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cs = np.insert(np.cumsum(self.el), 0, 0)

        order_k = min(3, len(self.points) - 1)
        self.tck = interpolate.splprep([self.points[:, 0], self.points[:, 1]], k=order_k, s=0)[0]

    def find_closest_point(self, pt, previous_maximum, string=""):
        dists = np.linalg.norm(self.points - pt, axis=1)
        closest_ind = np.argmin(dists)
        t_guess = self.cs[closest_ind] / self.cs[-1]

        closest_t = optimize.fmin(dist_to_p, x0=t_guess, args=(self.tck, pt), disp=False)
        if closest_t < 0:
            return self.points[0], closest_t
        t_pt = max(closest_t, previous_maximum)

        interp_return = interpolate.splev(t_pt, self.tck)
        closest_pt = np.array(interp_return).T
        if len(closest_pt.shape) > 1:
            closest_pt = closest_pt[0]

        return closest_pt, t_pt
    
    def extract_line_portion(self, s_array):
        assert np.min(s_array) >= 0, "S must be positive"
        assert  np.max(s_array) <= 1, "S must be < 1"
        point_set = np.array(interpolate.splev(s_array, self.tck)).T

        return point_set
        
    def plot_line(self):
        plt.plot(self.points[:, 0], self.points[:, 1], '.', markersize=10, color="#20bf6b")
        plt.plot(self.points[0, 0], self.points[0, 1], "o", color='#20bf6b', markersize=15)

    def apply_smoothing(self):
        line_length = np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))
        n_pts = max(int(line_length / POINT_SEP_DISTANCE), 2)
        smooth_line = interpolate_track_new(self.points, None, self.smoothing_s)
        self.points = interpolate_track_new(smooth_line, n_pts*2, 0)
        #NOTE: the double interpolation ensures that the lengths are correct.
        # the first step smooths the points and the second step ensures the correct spacing.
        # TODO: this could be achieved using the same tck and just recalculating the s values based on the new lengths. Do this for computational speedup.


class LocalLine:
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




if __name__ == "__main__":
    pass