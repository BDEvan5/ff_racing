import numpy as np 
from scipy.interpolate import splrep, BSpline
from scipy import interpolate
import os


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
    el = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cs = np.insert(np.cumsum(el), 0, 0)
    cs = cs / cs[-1]
    # print(f"Cs: {cs}")
    tck = interpolate.splprep([points[:, 0], points[:, 1]], u=cs, k=3, s=s)[0]
    if n_points is None: n_points = len(points)
    track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T

    new_distances = np.linalg.norm(np.diff(track, axis=0), axis=1)
    print(f"New distances: {new_distances}")

    return track

# def interpolate_track_weights(points, n_points, s=0):
#     ws = np.ones_like(points[:, 0])
#     ws[0:2] = 100
#     ws[-2:] = 100
#     tck = interpolate.splprep([track[:, 0], track[:, 1]], k=3, s=s)[0]
#     if n_pts is None: n_pts = len(self.track)
#     self.track[:, :2] = np.array(interpolate.splev(np.linspace(0, 1, n_pts), tck)).T


