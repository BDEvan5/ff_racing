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
        line_1, line_2 = self.extract_boundaries(pts, pt_distances, inds)
        left_pts, right_pts = self.estimate_center_line_devel(line_1, line_2, xs_f, ys_f)
        # left_pts, right_pts = self.estimate_center_line_clean(long_side, short_side)
        true_center_line = (left_pts + right_pts) / 2
        smooth_track = self.build_smooth_track(left_pts, right_pts)

        local_map = PlotLocalMap(smooth_track)
        # lm = LocalMap(track)
        # local_map.plot_local_map(xs=xs_f, ys=ys_f)

        # plt.plot(left_pts[:, 0], left_pts[:, 1], 'x', color='black', markersize=10)
        # plt.plot(right_pts[:, 0], right_pts[:, 1], 'x', color='black', markersize=10)
        
        # plt.plot(true_center_line[:, 0], true_center_line[:, 1], '*', color='green', markersize=10)

        # for i in range(left_pts.shape[0]):
        #     plt.plot([left_pts[i, 0], right_pts[i, 0]], [left_pts[i, 1], right_pts[i, 1]], color='black', linewidth=1)


        # line_1.plot_line()
        # line_2.plot_line()

        plt.show()
        # plt.pause(0.0001)

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

        line_1 = TrackBoundary(pts[:min_ind], True)
        line_2 = TrackBoundary(pts[max_ind:], True)

        return line_1, line_2
    
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

    def estimate_center_line_devel(self, line_1, line_2, xs=None, ys=None):
        plt.figure(2)
        plt.clf()
        if xs is not None and ys is not None:
            plt.plot(xs, ys, '.', color='#45aaf2', alpha=0.1)
        plt.plot(0, 0, 'x', markersize=14, color='red')

        line_1.plot_line()
        line_2.plot_line()
        
        center_pt = np.zeros(2)
        center_pt[0] = -1 # start before beginning
        step_size = 0.6
        max_pts = 20
        end_threshold = 0.1
        theta = 0
        boundary_1, boundary_2 = np.zeros((max_pts, 2)), np.zeros((max_pts, 2))
        center_pts = np.zeros((max_pts, 2))
        max_s_1, max_s_2 = 0, 0
        z = 0
        through_positive_corner = False
        ready_to_extend_line = False
        for i in range(max_pts):
            plt.plot(center_pt[0], center_pt[1], '.', color='red', markersize=10)
            pt_1, max_s_1 = line_1.find_closest_point(center_pt, max_s_1, "Line1")
            pt_2, max_s_2 = line_2.find_closest_point(center_pt, max_s_2, "Line2")

            line_center = (pt_1 + pt_2) / 2
            center_pts[i] = line_center

            if np.all(np.isclose(pt_1, boundary_1[i-1])) and np.all(np.isclose(pt_2, boundary_2[i-1])):
                print("Adding redundant points -- > move to projection")
                ready_to_extend_line = True
                i -= 4 # remove last two points
                # break

            if max_s_2 > 0.95:
                # consider a problem...
                previous_edge_diff = pt_1 - boundary_1[i-1] 
                previous_edge_heading = np.arctan2(previous_edge_diff[1], previous_edge_diff[0])
                center_curvature = calculate_curvature(center_pts[:i+1])

                n_diff = pt_1 - pt_2
                proposed_track_heading = np.arctan2(n_diff[1], n_diff[0]) - np.pi/2
                d_track_heading = proposed_track_heading - previous_edge_heading
                # print(f"{i} -> d_head {d_track_heading} --> Curvature: {center_curvature}")

                if d_track_heading > 0:
                    through_positive_corner = True
                
                if through_positive_corner and d_track_heading < 0:
                    print(f"Ready to extend line - {d_track_heading}")
                    ready_to_extend_line = True
                    break

                if d_track_heading < -0.2 and abs(center_curvature) < 0.01:
                    print(f"Ready to extend line -- heading difference too large")
                    ready_to_extend_line = True
                    break

            boundary_1[i] = pt_1
            boundary_2[i] = pt_2
            
            long_distance = np.linalg.norm(pt_1 - line_1.points[-1])
            short_distance = np.linalg.norm(pt_2 - line_2.points[-1])
            if long_distance < end_threshold and short_distance < end_threshold:
                print(f"Breaking because of long ({long_distance}) and short ({short_distance}) distances")
                print(f"Pt1: {pt_1} :: Pt2: {pt_2}")
                # i += 1 # include last point
                break

            n_diff = pt_1 - pt_2
            heading = np.arctan2(n_diff[1], n_diff[0])
            # print(f"Heading: {heading}")
            new_theta = heading + np.pi/2

            d_theta = new_theta - theta
            if new_theta > np.pi:
                new_theta = new_theta - 2 * np.pi
            elif new_theta < -np.pi:
                new_theta = new_theta + 2 * np.pi

            # print(f"Theta: {new_theta}")
            theta = new_theta

            if i == z: # first point
                d_theta = - np.abs(d_theta) * np.sign(theta)

            weighting = np.clip(abs(d_theta) / 0.2, 0, 0.7)
            # if theta < 0:
            if max_s_1 > 0.95:
                adjusted_line_center = (pt_2 * (weighting) + line_center * (1- weighting)) 
            elif max_s_2 > 0.95:
                adjusted_line_center = (pt_1 * (weighting) + line_center * (1- weighting)) 
            else:
                adjusted_line_center = line_center 


            center_pt = adjusted_line_center + step_size * np.array([np.cos(theta), np.sin(theta)])

            plt.plot(center_pt[0], center_pt[1], '*', color='blue', markersize=10)
            plt.plot(line_center[0], line_center[1], '*', color='orange', markersize=10)
            plt.plot(adjusted_line_center[0], adjusted_line_center[1], '*', color='purple', markersize=10)
            c_xs = [adjusted_line_center[0], center_pt[0]]
            c_ys = [adjusted_line_center[1], center_pt[1]]
            plt.plot(c_xs, c_ys, '-', color='purple')

            plt.plot(pt_1[0], pt_1[1], 'x', color='black', markersize=10)
            plt.plot(pt_2[0], pt_2[1], 'x', color='black', markersize=10)
            xs = [pt_1[0], pt_2[0]]
            ys = [pt_1[1], pt_2[1]]
            plt.plot(xs, ys, '-', color='black')

        if i == max_pts - 1:
            # plt.show()
            print(f"No points found - make via extension")
            ready_to_extend_line = True
            i = 0
            z = 0
            plt.axis('equal')
            plt.show()

        k = 0
        # if ready_to_extend_line:/
        if False:
            print(f"Extending lines.... from {i} onwards")
            for k in range(i, max_pts):
                pt_1, max_s_1 = long_bound.find_closest_point(center_pt, max_s_1)
                
                previous_edge_diff = pt_1 - boundary_1[k-1] 
                previous_edge_heading = np.arctan2(previous_edge_diff[1], previous_edge_diff[0])
                nvec_angle = previous_edge_heading + np.pi/2
                pt_2 = pt_1 - TRACK_WIDTH * np.array([np.cos(nvec_angle), np.sin(nvec_angle)])
                # print(f"New proposed point: {short_pt}")

                new_line = [pt_2, pt_1]
                old_line = [boundary_1[k-1], boundary_2[k-1]]
                if do_lines_intersect(new_line, old_line):
                    print(f"Lines intersect: Move on....")
                    break

                boundary_1[k] = pt_1
                boundary_2[k] = pt_2
                
                long_distance = np.linalg.norm(pt_1 - long_bound.points[-1])
                if long_distance < end_threshold:
                    print(f"Breaking long distance")
                    break

                n_diff = pt_1 - pt_2
                heading = np.arctan2(n_diff[1], n_diff[0])
                new_theta = heading - np.pi/2
                d_theta = new_theta - theta
                theta = new_theta

                line_center = (pt_1 + pt_2) / 2
                weighting = np.clip(abs(d_theta) / 0.2, 0, 0.8)
                # print(f"Weighting: {weighting}")
                if d_theta > 0:
                    adjusted_line_center = (pt_2 * (weighting) + line_center * (1- weighting)) 
                else:
                    adjusted_line_center = (pt_1 * (weighting) + line_center * (1- weighting)) 

                center_pt = adjusted_line_center + step_size * np.array([np.cos(theta), np.sin(theta)])

                plt.plot(center_pt[0], center_pt[1], '*', color='blue', markersize=10)
                plt.plot(line_center[0], line_center[1], '*', color='orange', markersize=10)
                plt.plot(adjusted_line_center[0], adjusted_line_center[1], '*', color='purple', markersize=10)
                c_xs = [adjusted_line_center[0], center_pt[0]]
                c_ys = [adjusted_line_center[1], center_pt[1]]
                plt.plot(c_xs, c_ys, '-', color='purple')

                plt.plot(pt_1[0], pt_1[1], 'x', color='black', markersize=10)
                plt.plot(pt_2[0], pt_2[1], 'x', color='black', markersize=10)
                xs = [pt_1[0], pt_2[0]]
                ys = [pt_1[1], pt_2[1]]
                plt.plot(xs, ys, '-', color='black')


        plt.axis('equal')
        # plt.show()
        plt.pause(0.00001)
        end_ind = max(i, k)
        boundary_1 = np.array(boundary_1[z:end_ind+1])
        boundary_2 = np.array(boundary_2[z:end_ind+1])

        if len(boundary_1) < 2:
            print(f"Only {len(boundary_1)} points found. This is a problem")


        return boundary_1, boundary_2

    def estimate_center_line_clean(self, long_side, short_side):
        long_bound =  TrackBoundary(long_side)
        short_bound = TrackBoundary(short_side)
        long_bound.plot_line()
        short_bound.plot_line()
        
        center_pt = np.zeros(2)
        step_size = 0.6
        max_pts = 30
        end_threshold = 0.1
        theta = 0
        left_pts, right_pts = np.zeros((max_pts, 2)), np.zeros((max_pts, 2))
        max_long_s, max_short_s = 0, 0
        for i in range(max_pts):
            long_pt, max_long_s = long_bound.find_closest_point(center_pt, max_long_s)
            short_pt, max_short_s = short_bound.find_closest_point(center_pt, max_short_s)



            left_pts[i] = long_pt
            right_pts[i] = short_pt
            
            long_distance = np.linalg.norm(long_pt - long_bound.points[-1])
            short_distance = np.linalg.norm(short_pt - short_bound.points[-1])
            if long_distance < end_threshold and short_distance < end_threshold:
                break

            n_diff = long_pt - short_pt
            new_theta = np.arctan2(n_diff[1], n_diff[0]) - np.pi/2
            d_theta = new_theta - theta
            theta = new_theta

            line_center = (long_pt + short_pt) / 2
            weighting = np.clip(abs(d_theta) / 0.2, 0, 0.8)
            if d_theta > 0:
                adjusted_line_center = (short_pt * (weighting) + line_center * (1- weighting)) 
            else:
                adjusted_line_center = (long_pt * (weighting) + line_center * (1- weighting)) 

            center_pt = adjusted_line_center + step_size * np.array([np.cos(theta), np.sin(theta)])

        left_pts = np.array(left_pts[:i+1])
        right_pts = np.array(right_pts[:i+1])

        return left_pts, right_pts

    def build_smooth_track(self, left_pts, right_pts):
        c_line = np.zeros_like(left_pts)
        c_line[0] = (left_pts[0] + right_pts[0]) / 2
        search_size = 2
        for i in range(1, len(left_pts)):
            diff = (left_pts[i-1] - right_pts[i-1])
            theta_1 = np.arctan2(diff[1], diff[0]) - np.pi/2
            diff = (left_pts[i] - right_pts[i])
            theta_2 = np.arctan2(diff[1], diff[0]) - np.pi/2
            new_theta = (theta_1 + theta_2) / 2

            line1 = [left_pts[i], right_pts[i]]
            line2 = [c_line[i-1], c_line[i-1] + np.array([np.cos(new_theta), np.sin(new_theta)]) * search_size]

            intersection = calculate_intersection(line1, line2)
            # print(f"Intersection: {intersection}")
            if intersection is None: # or intersection[0] == 1e9:
                print(f"Line 1: {line1}")
                print(f"Line 2: {line2}")
                raise ValueError("No intersection found")
            c_line[i] = intersection

        ws_1 = np.linalg.norm(c_line - left_pts, axis=1)[:, None]
        ws_2 = np.linalg.norm(c_line - right_pts, axis=1)[:, None]

        track = np.concatenate((c_line, ws_1, ws_2), axis=1)

        return track

def calculate_curvature(pts):
    d1 = pts[-1] - pts[-2]
    d2 = pts[-2] - pts[-3]
    head1 = np.arctan2(d1[1], d1[0])
    head2 = np.arctan2(d2[1], d2[0])
    distance = np.linalg.norm(pts[1] - pts[-3])
    curvature = (head2 - head1)/distance

    return curvature
    

class TrackBoundary:
    def __init__(self, points, smoothing=False) -> None:        
        self.smoothing_s = 0.5
        # print(f"Pts start: {points[0,0]} --> end: {points[-1,0]}")
        if points[0, 0] > points[-1, 0]:
            self.points = np.flip(points, axis=0)
            # print(f"FLIPPED :: New Pts start: {self.points[0,0]} --> end: {self.points[-1,0]}")
        else:
            self.points = points

        if smoothing:
            self.apply_smoothing()

        self.el = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cs = np.insert(np.cumsum(self.el), 0, 0)

        self.tck = interpolate.splprep([self.points[:, 0], self.points[:, 1]], k=3, s=0)[0]

    def find_closest_point(self, pt, previous_maximum, string=""):
        dists = np.linalg.norm(self.points - pt, axis=1)
        closest_ind = np.argmin(dists)
        t_guess = self.cs[closest_ind] / self.cs[-1]

        closest_t = optimize.fmin(dist_to_p, x0=t_guess, args=(self.tck, pt), disp=False)
        if closest_t < 0:
            print(f"{string}: Guess, {t_guess:.3f}; Closest_t {closest_t}, Previous {previous_maximum}")
            return self.points[0], closest_t
        t_pt = max(closest_t, previous_maximum)

        print(f"{string}: Guess, {t_guess:.3f}; Closest_t {closest_t}, Previous {previous_maximum}")

        interp_return = interpolate.splev(t_pt, self.tck)
        closest_pt = np.array(interp_return).T
        if len(closest_pt.shape) > 1:
            closest_pt = closest_pt[0]

        return closest_pt, t_pt
        
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

def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
    s = interpolate.splev(t_glob, path, ext=3)
    s = np.concatenate(s)
    return spatial.distance.euclidean(p, s)

# @njit(cache=True)
def calculate_intersection(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # Calculate the slopes of the lines
    if x2 - x1 != 0:
        slope1 = (y2 - y1) / (x2 - x1)  
    else:
        slope1 = 1e9
    
    if x4 - x3!= 0:
        slope2 = (y4 - y3) / (x4 - x3)
    else:
        slope2 = 1e9

    # Check if the lines are parallel
    if slope1 == slope2:
        return None

    # Calculate the y-intercepts of the lines
    intercept1 = y1 - slope1 * x1
    intercept2 = y3 - slope2 * x3

    # Calculate the intersection point (x, y)
    if slope1 == 1e9:
        x = x1
        y = slope2 * x + intercept2
    if slope2 == 1e9:
        x = x3
        y = slope1 * x + intercept1
    else:
        x = (intercept2 - intercept1) / (slope1 - slope2)
        y = slope1 * x + intercept1 # can use either

    return np.array([x, y])


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

if __name__ == "__main__":
    pass