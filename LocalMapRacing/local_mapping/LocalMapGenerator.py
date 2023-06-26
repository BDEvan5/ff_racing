import numpy as np
import matplotlib.pyplot as plt
import LocalMapRacing.tph_utils as tph
from matplotlib.collections import LineCollection
np.set_printoptions(precision=4)
from LocalMapRacing.local_mapping.local_map_utils import *
from LocalMapRacing.local_mapping.LocalMap import LocalMap

DISTNACE_THRESHOLD = 1.6 # distance in m for an exception
TRACK_WIDTH = 1.8 # use fixed width
POINT_SEP_DISTANCE = 1.2


class LocalMapGenerator:
    def __init__(self, path) -> None:
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)
        self.sines = np.sin(self.angles)
        self.xs, self.ys = None, None 

        self.local_map_data_path = path + "LocalMapData/"
        # ensure_path_exists(self.local_map_data_path)
        self.counter = 0

    def generate_line_local_map(self, scan):
        self.counter += 1
        xs = self.coses[scan < 10] * scan[scan < 10] #? why are long beams excluded???? Try without this.
        ys = self.sines[scan < 10] * scan[scan < 10]

        pts, pt_distances, inds = self.extract_track_lines(xs, ys)

        long_side, n_pts, w = self.calculate_longest_line(pts, pt_distances, inds)

        track = self.project_side_to_track(long_side, w, n_pts)
        local_map = self.adjust_track_normals(track)

        return local_map

    def extract_track_lines(self, xs, ys):
        clip_xs, clip_ys = xs[180:-180], ys[180:-180] 

        pts = np.hstack((clip_xs[:, None], clip_ys[:, None]))
        pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        inds = np.where(pt_distances > DISTNACE_THRESHOLD)
        inds = np.delete(inds, np.where(inds[0] == 0)) 

        if len(inds) == 0:
            pts = np.hstack((xs[:, None], ys[:, None]))
            pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            inds = np.where(pt_distances > DISTNACE_THRESHOLD)
            inds = np.delete(inds, np.where(inds[0] == 0)) 

            if len(inds) == 0: 
                raise IOError("Problem with full scan, no gaps found")

        return pts, pt_distances, inds

    def calculate_longest_line(self, pts, pt_distances, inds):
        arr_inds = np.arange(len(pt_distances))[inds]
        min_ind = np.min(arr_inds) + 1
        max_ind = np.max(arr_inds) + 1

        l1_cs = np.cumsum(pt_distances[:min_ind-1])
        l2_cs = np.cumsum(pt_distances[max_ind:])

        if l1_cs[-1] > l2_cs[-1]:
            long_pts = pts[:min_ind]
            line_length = l1_cs[-1]
            w = 1
        else:
            long_pts = pts[max_ind:]
            long_pts = long_pts[::-1]
            line_length = l2_cs[-1]
            w = -1
        
        n_pts = int(line_length / POINT_SEP_DISTANCE)
        long_side = interpolate_track(long_pts, n_pts*2, 0)

        return long_side, n_pts, w

    def project_side_to_track(self, side, w, n_pts):
        side_lm = LocalMap(side)
        center_line = side + side_lm.nvecs * w * TRACK_WIDTH / 2
        center_line = interpolate_track(center_line, n_pts, 1)

        ws = np.ones_like(center_line) * TRACK_WIDTH / 2
        track = np.concatenate((center_line, ws), axis=1)

        return track

    def adjust_track_normals(self, track):
        lm = LocalMap(track)

        crossing_horizon = min(5, len(lm.track)//2 -1)
        i = 0
        while i < 20 and tph.check_normals_crossing.check_normals_crossing(lm.track, lm.nvecs, crossing_horizon):
            i += 1
            if np.mean(lm.kappa) > 0:
                lm.track[:, 2] *= 0.9
            else:
                lm.track[:, 3] *= 0.9
            lm.calculate_length_heading_nvecs()
            # print(f"{i}:: Normals crossed --> New width: {lm.track[0, 2:]}")

        return lm

        
        



if __name__ == "__main__":
    pass