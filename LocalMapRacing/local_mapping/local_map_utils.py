import numpy as np 
from scipy.interpolate import splrep, BSpline
from scipy import interpolate
from scipy import spatial
import os
from numba import njit


def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    
    return xs, ys

def interp_nd_points(ss, xp, track):
    new_track = np.zeros((len(ss), track.shape[1]))
    for i in range(track.shape[1]):
        new_track[:, i] = np.interp(ss, xp, track[:, i])
    
    return new_track

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)

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

def calculate_track_direction(pt_1, pt_2):
    n_diff = pt_1 - pt_2
    heading = np.arctan2(n_diff[1], n_diff[0])
    theta = heading + np.pi/2

    if theta > np.pi:
        theta = theta - 2 * np.pi
    elif theta < -np.pi:
        theta = theta + 2 * np.pi

    return theta


def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
    s = interpolate.splev(t_glob, path, ext=3)
    s = np.concatenate(s)
    return spatial.distance.euclidean(p, s)

    
@njit(cache=True)
def calculate_offset_coords(pts, position, heading):
    rotation = np.array([[np.cos(heading), -np.sin(heading)],
                        [np.sin(heading), np.cos(heading)]])
        
    new_pts = (rotation @ pts.T).T + position

    return new_pts

@njit(cache=True)
def calculate_speed(delta, f_s=0.8, max_v=7):
    b = 0.523
    g = 9.81
    l_d = 0.329

    if abs(delta) < 0.03:
        return max_v
    if abs(delta) > 0.4:
        return 3

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    V = min(V, max_v)

    return V

def interpolate_track(points, n_points, s=10):
    el = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cs = np.insert(np.cumsum(el), 0, 0)
    ss = np.linspace(0, cs[-1], n_points)
    tck_x = splrep(cs, points[:, 0], s=s, k=min(3, len(points)-1))
    tck_y = splrep(cs, points[:, 1], s=s, k=min(3, len(points)-1))
    xs = BSpline(*tck_x)(ss) # get unispaced points
    ys = BSpline(*tck_y)(ss)
    new_points = np.hstack((xs[:, None], ys[:, None]))

    return new_points


def interpolate_track_new(points, n_points=None, s=0):
    order_k = min(3, len(points) - 1)
    tck = interpolate.splprep([points[:, 0], points[:, 1]], k=order_k, s=s)[0]
    if n_points is None: n_points = len(points)
    track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T

    return track

# def interpolate_track_weights(points, n_points, s=0):
#     ws = np.ones_like(points[:, 0])
#     ws[0:2] = 100
#     ws[-2:] = 100
#     tck = interpolate.splprep([track[:, 0], track[:, 1]], k=3, s=s)[0]
#     if n_pts is None: n_pts = len(self.track)
#     self.track[:, :2] = np.array(interpolate.splev(np.linspace(0, 1, n_pts), tck)).T


