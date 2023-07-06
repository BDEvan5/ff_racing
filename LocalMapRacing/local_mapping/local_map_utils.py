import numpy as np 
from scipy.interpolate import splrep, BSpline
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
